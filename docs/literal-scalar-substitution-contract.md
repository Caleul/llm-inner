# Contrato de substituição escalar literal

## Objetivo

O produto final é um programa matemático autocontido. Um nó de alto nível como
`linear`, `rms_norm` ou `scaled_dot_product_attention` é útil para execução,
mas não basta como explicação auditável. Cada operação deve ter uma forma por
índice que permita acompanhar os dados desde `x[...]`, através de cada camada,
até os logits.

Para uma projeção linear com redução escalar ordenada, a forma obrigatória é:

```text
y[t,o] = F32(sum_{i=0..I-1, em ordem crescente}(F32(x[t,i] * W[o,i])) + b[o])
```

Na visualização literal, `W[o,i]` e `b[o]` são substituídos pelos números
exatos decodificados. Exemplo:

```text
y[0,0] = F32(F32(x[0,0] * 3.456812134) + F32(x[0,1] * -0.125) + 0.75)
```

## Regra de armazenamento versus regra de auditoria

Não é aceitável copiar todos os produtos escalares de uma E4B para o JSON
principal: isso repetiria dezenas de bilhões de pesos em cada uso e tornaria o
artefato inutilizável. O formato tem, portanto, duas representações da mesma
semântica:

1. **Programa de produção:** payload exato de cada tensor, dtype, shape,
   layout, byte order, packing/quantização e decoder; operações vetoriais com
   domínios e ordem de acumulação declarados.
2. **Vista de auditoria literal:** expansão finita solicitada por camada,
   operação, índice ou janela. Ela decodifica o valor do payload e o substitui
   na fórmula. Nunca inventa um peso e nunca deixa `weight[o,i]` sem caminho
   determinístico para o número correspondente.

As duas formas devem produzir a mesma política numérica. Um `F32` deve indicar
onde cada produto, soma, cast, softmax e dequantização arredonda; `BF16`/`F16`
deve expor sua conversão IEEE antes do uso.

Uma redução vetorial só é válida quando declara o mapeamento de cada índice de
entrada para a lane e a dobra horizontal. Por exemplo, uma agenda
`tile-contiguous-terms` com `laneCount=16` e `termsPerLane=2` declara, para
cada tile de 32 índices, que `i=0,1` alimenta a lane 0, `i=2,3` a lane 1 e
assim por diante; cada adição na lane e a árvore final também são `F32` e
auditáveis. Isso não pode ser abreviado como uma propriedade do hardware.

Uma redução de produto adjacente também não pode esconder sua fronteira. Uma
agenda `blocked-f32-terms` com `termsPerBlock=2`, `inputBlock=contiguous-terms`
e `productBoundary=separately-rounded-f32` declara
`partial[b] = F32(F32(x[2b] * W[o,2b]) + F32(x[2b+1] * W[o,2b+1]))` e depois
`y[t,o] = F32(sum_b em ordem crescente(partial[b]))`. A variante `fused-fma`
mantém cada produto exato até a soma F32 do parcial. Isto é diferente tanto de
uma soma escalar por termo quanto de acumular em lanes e dobrá-las ao final.

Uma agenda `blocked-tiled-f32-lanes` descreve outra fronteira observável:
para cada tile finito de `laneCount * termsPerLane` coordenadas, ela inicializa
lanes F32, aplica os produtos contíguos a cada lane, dobra as lanes na ordem
declarada e só então adiciona esse parcial ao acumulador F32 ordenado dos
tiles. As lanes não atravessam tiles. Portanto ela não é equivalente a
`tiled-f32-lanes`, que mantém as lanes vivas por toda a redução.

## Requisitos por operação

- `linear`: índices de batch/token/saída/entrada, ordem da soma, pesos e bias.
- `rms_norm`: eixo de redução, tamanho, epsilon, cada termo quadrático e peso.
- atenção: índices de cabeça/query/key/canal, escala, máscara, ordem de
  softmax e soma ponderada de `V`.
- RoPE, reshape e scatter: mapeamento índice-a-índice completo.
- quantização: endereço do código, escala/zero/bias/codebook e fórmula de
  dequantização antes de qualquer operação consumidor.
- cache KV e geração: estado anterior, escrita, leitura e predecessor nomeado.

O papel do tensor também é dado, não inferido. Cada consumidor aprendido deve
ligar explicitamente `weight`, `bias`, limites de clipping, tabela posicional,
kernel de convolução, escala de normalização ou escala por dimensão ao tensor,
ao decoder e à expressão de índices lógicos correspondente. Rank, shape,
ordem no array e sufixo de nome podem ser validados, mas não podem selecionar o
papel semântico durante a auditoria.

A fórmula também é dado do artefato, não comportamento secreto do leitor. O
Gemma 4 schema v5 inclui `scalarCalculations` com uma entrada para cada
definição composite, vision, audio e text. A entrada contém `orderedInputs`,
coordenadas nomeadas, fórmula indexada, papéis aprendidos, política de dtype e
o domínio/agenda completo de cada redução. A navegação instancia essas mesmas
entradas para imagem e vídeo e a vista escalar as acompanha enquanto substitui
os valores decodificados.

O schema v6 aplica o mesmo princípio às transições greedy. O artefato inclui
`generation.scalarCalculations` para as 12 definições de controle e
`generation.forwardCalculation.operationOrder` para a expansão instanciada do
forward. `generation.forwardCalculation.cacheTransitions` declara, por camada,
layout BHSD, operação de atenção, produtor, leitura compartilhada e a fórmula
de prefill ou concatenação incremental no eixo de sequência. Um leitor pode
substituir `step` e `max_new_tokens`, mas não pode inventar a fórmula de argmax,
EOS, posição, cache ou estado forward fora do JSON.

No schema v7, `calculationGraph.assignments` é a sequência forward canônica.
Cada entrada vincula a definição reutilizável ao seu `invocationId`, substitui
nomes locais por entradas/saídas instanciadas, carrega o domínio e cálculo
escalar correspondentes e declara as arestas de predecessor e consumidor.
Portanto um leitor não pode reconstruir a expansão de `vision-feature-program`,
`audio-feature-program` ou `text-core` a partir de convenções próprias. As
fórmulas de wrapper referem o intervalo instanciado no grafo; a navegação e a
geração consomem essa mesma sequência serializada e falham se outra ordem for
declarada.

No schema v8, cada token numérico presente nessas fórmulas e nas fórmulas
greedy também possui uma declaração em `numericLiterals.literals`. O token é
ligado aos seus consumidores e aos bits F64, F32 e BF16; o cast escrito na
fórmula seleciona a representação aplicável. BF16 declara explicitamente
round-to-nearest-ties-to-even. Constantes nomeadas como `pi` carregam os mesmos
bits, portanto um leitor não pode obter outro valor da biblioteca matemática
do host. A tabela inteira é derivada e validada fail-closed contra as fórmulas
serializadas.

No schema v9, `sourceIdentity` torna a autoridade do cálculo parte do JSON.
`modelId`, revisão imutável, adaptador, nomes, tamanhos e SHA-256 dos arquivos
do pacote são validados fail-closed. Os JSONs de configuração, geração,
processor e tokenizer permanecem incorporados como bytes Base64 com decoder e
digest exatos; os Safetensors são comprometidos como arquivos completos e seus
valores aprendidos continuam incorporados uma única vez em `constants`. Assim,
o leitor source-removed não depende de documentação externa para descobrir de
qual pacote vieram a topologia, os controles ou a tokenização declarados.

No schema v10, `formulaLanguage` incorpora a interpretação normativa de
`indexed-ieee754-expression-v1`. Seus JSON pointers ligam fórmulas forward e
greedy à ordem instanciada, domínios, bindings aprendidos, decoders e bits dos
tokens numéricos. O contrato fixa indexação zero-based, layout, ordem de
coordenadas, materialização de casts F64/F32/BF16, operadores, intrínsecos e
reduções. `runtime-defined` é explicitamente não executável: nenhum leitor pode
substituí-lo por uma redução do host e chamar o resultado de literal.

No schema v11, os endereços de `learnedOperands` deixam de ser strings livres.
Cada eixo lógico é uma AST `gemma4-learned-index-expression-v1`, cujas folhas
diferenciam coordenada de saída, índice de redução e escalar de input. Reduções
e inputs carregam bounds explícitos; `add`, `multiply` e `modulo` operam apenas
sobre inteiros seguros não negativos em ordem depth-first left-to-right. O
avaliador deve resolver a AST, validar cada dimensão contra o shape lógico e
só então aplicar o decoder e o offset row-major. Um binding ausente, overflow,
divisor não positivo ou índice fora do tensor é erro antes de qualquer leitura.

No schema v12, o decoder não deixa ao leitor a conversão desse índice em bytes.
`storageDecoder.address` declara o shape lógico, os strides elementares
row-major, o domínio de cada eixo, a largura do elemento e as equações de
element/byte offset. `storageDecoder.decode` declara a leitura little-endian e
a transformação exata dos bits de F32, BF16 ou F16 nos bits F32. Para F16, os
quatro casos zero, subnormal, normal e infinito/NaN, suas máscaras e a
normalização da fração fazem parte do JSON. A substituição deve executar esses
programas incorporados; recalcular strides ou selecionar uma conversão apenas
pelo nome do dtype é comportamento externo proibido.

Quando a autoridade disponível prova a equação, mas não a árvore de uma
redução nativa, a fórmula preserva o domínio matemático completo e a entrada
declara `reproducibility=fail-closed-runtime-reduction` com
`order=runtime-defined`. Alterar esse limite para uma ordem inventada invalida
o artefato. Assim, cobertura estrutural não é confundida com fidelidade
numérica ainda não demonstrada.

## Critério de aceite

Uma implementação só é candidata a fechar este requisito quando:

1. gera um exemplo pequeno completo como
   `docs/examples/literal-scalar-substitution.example.json`;
2. um leitor independente reproduz o `expectedOutputs` apenas de `inputs`,
   valores substituídos e operações declaradas;
3. a implementação rejeita uma constante, índice, ordem de redução ou
   arredondamento ausente; e
4. o próximo loop revisa independentemente o diff e as evidências antes de
   aceitar a alegação. O agente implementador nunca a certifica sozinho.

## Vista navegável do artefato Gemma 4

`npm run inspect:gemma4-literal -- --artifact <json> --list-operations`
expande o programa composite na ordem de dependência. As chamadas de imagem e
vídeo instanciam separadamente a mesma definição visual, a chamada de áudio é
instanciada no seu ponto de uso e o texto preparado começa nas camadas — o
prelude standalone não é executado novamente. Cada atribuição preserva output,
predecessores produtores, consumidores, vizinhos, definição e invocação.

Uma coordenada é expandida com `--operation <id> --output-coordinate
<i,j,...>`. O resultado `gemma4-literal-scalar-view` deve registrar:

- `sourceCheckpointAccessed: false`;
- bits de storage, dtype, decoder, índice row-major e literal F32 exato de
  toda constante aprendida referenciada;
- uma fórmula por termo na qual o número aparece diretamente, sem
  `weight[...]`;
- bounds completos e a agenda de redução serializada;
- casts de saída F32/BF16; e
- `complete`, `renderedWindow` e `omittedTerms`, para que uma janela diagnóstica
  nunca seja confundida com a equação completa.

Lineares expandem a redução inteira quando nenhuma janela é fornecida.
Embeddings exigem o token concreto, porque `input_ids` é uma variável do
programa e não pode receber valor implícito. A implementação suporta somente
storage denso row-major F32/F16/BF16 nesse caminho e rejeita outros decoders em
vez de inventar uma interpretação.

`--end-to-end-calculation --generation-max-new-tokens <n>` compõe essa mesma
slice forward com todas as fórmulas de controle greedy instanciadas e as
transições KV armazenadas. O forward é compartilhado uma vez entre prefill e
decode, mas cada invocação declara o modo e o estado de entrada. A vista deve
particionar todas as constantes incorporadas entre operandos alcançáveis do
logit e projeções K/V ou K-norm locais declaradas runtime-unreachable por
reutilizar o produtor compartilhado. A soma das duas classes precisa ser exatamente o
número de constantes do artefato; órfãos, sobreposição ou decoder ausente são
erro, nunca warning.

## Programa literal de geração

Logits não completam o produto de geração. O artefato Gemma 4 schema v2 deve
declarar também `max_new_tokens` e `eos_token_id`, além de uma máquina de estado
greedy ordenada. `requiredFor` diferencia inputs obrigatórios por modo, sem
tornar `max_new_tokens` um requisito de forward. A forma canônica registra,
sem depender de defaults de runtime:

1. o prefill pelo programa forward declarado;
2. a seleção da última linha de logits;
3. argmax por varredura crescente, com empate pelo menor token ID;
4. append do token antes de qualquer parada EOS;
5. avanço inteiro da posição;
6. entrada incremental `[1,1]` com o cache pós-RoPE anterior;
7. ausência explícita de máscaras/modais de prefill no decode;
8. nova execução do mesmo programa de cálculo declarado;
9. captura do cache resultante;
10. avaliação de EOS somente depois que logits e cache incrementais existem;
11. seleção dos logits do último forward realmente executado; e
12. seleção do cache pertencente ao mesmo estado terminal.

Cada passo nomeia dtype, shape, inputs e output. O leitor streaming deriva o
contrato canônico do próprio programa incorporado e rejeita uma atribuição,
ordem, semântica, input ou output divergente. `--show-generation-program`
expõe esse estado mesmo quando o checkpoint está indisponível.

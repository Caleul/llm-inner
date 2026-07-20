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

No schema v13, essas equações deixam de ser strings normativas. O artefato
incorpora `denseDecoderLanguage`, que fixa a ordem e a semântica das linguagens
`exact-safe-integer-expression-v1` e `u32-bit-expression-v1`. O programa de
endereço é uma redução AST por eixo com bindings explícitos para índice,
stride, element offset e element bytes. O programa IEEE é uma AST u32 com
leitura little-endian, bindings ordenados, operações bitwise, shifts,
`count-leading-zeros-u32` e `select-u32`; seus bits finais são bitcast para F32.
O leitor executa essas ASTs, não um ramo de conversão escolhido por `dtype`.
Shifts fora de `0..31`, bindings ausentes/duplicados, overflow inteiro, opcode
desconhecido ou uma alteração no contrato canônico falham antes de retornar o
literal. A cobertura F16 percorre exaustivamente os 65.536 words possíveis e a
cobertura BF16 prova `sourceBits << 16` para o mesmo domínio completo.

No schema v14, uma agenda FMA/dot não pode mais coexistir com uma fórmula que
materializa `F32(input*weight)` antes da acumulação. O intrínseco incorporado
`exact_product(a*b)` retém o produto matemático dos operandos já materializados
até a fronteira F32 imediatamente declarada pela agenda. A expansão linear
completa define `product[i]` para cada índice usando o literal aprendido real e
depois transcreve, sem atalhos, a região principal de registradores, a árvore
ARM, o fold horizontal, o vector tail e o scalar tail. `weight[o,i]`,
`decode(role)` e nomes opacos de kernel são inválidos numa vista escalar já
renderizada; termos faltantes também invalidam uma vista marcada `complete`.

No schema v15, nomes de schedules de normalização também deixam de ser
atalhos. `formulaLanguage.reductions.normalizationPrograms` incorpora as
transições normativas de `pytorch-cpu-f32-cascade-sum` e
`pytorch-cpu-bf16-welford`. A vista RMS declara cada square, unidade de 16
coordenadas, nível de cascade, merge/clear, fold de registrador e fold de lane.
A LayerNorm de canais declara updates low/high de média e M2, merge vetorial,
fold Welford de lanes, variância, sqrt, recíproco, bias nativo, segundo passe e
gamma. Conv2d e depthwise substituem todos os kernels aprendidos e aplicam a
agenda intercalada serializada; padding multiplica o literal real por zero em
vez de omiti-lo. `PYTORCH_CPU_F32_CASCADE_SUM`, `mean_channels`,
`variance_channels` e uma soma ordenada que contradiga a agenda são inválidos
numa vista escalar completa.

Quando a autoridade disponível prova a equação, mas não a árvore de uma
redução nativa, a fórmula preserva o domínio matemático completo e a entrada
declara `reproducibility=fail-closed-runtime-reduction` com
`order=runtime-defined`. Alterar esse limite para uma ordem inventada invalida
o artefato. Assim, cobertura estrutural não é confundida com fidelidade
numérica ainda não demonstrada.

Uma redução nesse estado ainda deve ser auditável até a fronteira desconhecida.
Para as cinco classes BMM Gemma 4, a vista de produtos enumera o par de
endereços de operandos para cada índice solicitado, resolve coordenadas de
head/chunk e explicita o predicado que transforma padding em zero. O termo
`REAL_PRODUCT(left * right)` descreve somente o produto matemático antes do
kernel; ele não autoriza arredondamento, FMA, lane, tile ou fold. A vista deve
terminar em `status=fail-closed-runtime-reduction`, nomear o provider e declarar
`productRounding` e `accumulationOrder` como
`unpublished-provider-boundary`. Ela nunca produz o escalar de saída nem
substitui a rejeição da vista executável estrita.

No schema v38, a ordem dos statements deixa também de esconder a navegação
entre intermediários internos. Cada cálculo forward incorpora
`statementDataflow` com uma entrada para cada ordinal de `scalarAssignments`.
`writes` nomeia o local ou output produzido; `reads` registra cada acesso local
distinto e o `producerStatementOrdinal` anterior; e
`consumerStatementOrdinals` fornece as arestas reversas. Um local indexado como
`score[key]` ou `acc[patch-1]` carrega `coordinates` e os mesmos
`coordinatePrograms` executáveis do grafo externo. A construção é genérica por
statement: leitura antes da produção, output terminal ausente/duplicado,
coordenada malformada, aresta reversa divergente ou binding livre falham antes
da abertura do artefato. A string continua sendo o rendering auditável nesse
schema; o dataflow elimina somente a necessidade de reparsá-la para descobrir
produtores, consumidores e coordenadas dos intermediários.

No schema v39, `statementPrograms` remove também a necessidade de reparsar a
string para descobrir a aritmética. Cada entry tem o mesmo ordinal e source de
`scalarAssignments`, targets estruturados e uma expressão em tagged union
fechada. A linguagem cobre literals, identifiers, arrays, unary/binary,
conditional, call, index, member, range-inclusive, named-argument,
filtered-domain, ordered-loop e evaluate-invocation. Ordem dos filhos,
short-circuit, branch selection, casts, binding de redução, filtro, loop e
invocação composite são declarados por `formulaLanguage.scalarPrograms`.
`source` é somente o rendering humano; a AST é a autoridade executável.
Alterar qualquer nó, manter uma string sem AST equivalente, introduzir um kind
desconhecido ou reinterpretar `/` de um ID instanciado como divisão falha
fechado.

No schema v40, a AST deixa de depender de um ambiente de nomes implícito.
`statementEnvironment` enumera as coordenadas de output, inputs posicionais,
produtores locais, índices/extent aliases ligados a `reduction.domains`, roles
aprendidas, intrinsics, valores especiais e acessos de membro registrados.
Esse ambiente é reconstruído após o binding de cada call site. Um identifier
que não pertença a uma dessas classes, um `decode(role)` sem binding aprendido,
um helper ou named argument não registrado, um `reductionStages[id]` ausente,
um `.shape` fora de tensor declarado ou um campo de STRUCT sem contrato falha
antes de qualquer output. Assim, sintaxe válida não é confundida com semântica
executável.

No schema v41, `payloadIntegrity` deixa de ser opcional e
`integrityManifest` compromete as 23 seções não-payload do programa, inclusive
metadados das constantes, todos os contratos executáveis e a própria tabela de
digests de payload. A canonicalização é declarada como bytes UTF-8 do
`JSON.stringify` ECMAScript por seção, em ordem fixa; cada entrada declara
tamanho e SHA-256, e uma raiz liga a sequência completa. O leitor streaming
recalcula a raiz sem materializar os payloads de 20 GiB, enquanto a verificação
source-removed percorre e rehasha cada byte aprendido. Ausência, alteração ou
reordenação não reconhecida falha antes da navegação ou replay.

No schema v42, `fidelityGate` é a 24a seção comprometida pelo manifesto. O gate
é reconstruído a partir de `calculationGraph` e `authoritativeExecution` e
enumera toda redução `runtime-defined` com classe, ordinal, IDs, output e JSON
Pointers para a semântica escalar e a coordenada navegável. Redução fora das
cinco classes autoritativas, marcação fail-closed incompleta ou qualquer
alteração da lista é rejeitada. Lista não vazia obriga
`exactReplayClaim=forbidden`; lista vazia declara somente elegibilidade para
uma certificação independente com replay source-removed e diferencial fresco.

No schema v43, a auditoria dos produtos de uma redução nativa não recebe mais
uma fórmula paralela codificada no leitor. Ela localiza o único `REDUCE` no
`statementPrograms` incorporado, resolve os locals escalares anteriores em
ordem, associa os dois acessos indexados aos `orderedInputs`, liga a coordenada
solicitada aos `outputCoordinates` e usa o `endExclusive` do domínio de redução
como única autoridade do extent. Predicados de padding vêm da própria AST
condicional. Extents fixos de arquitetura são constantes serializadas; extents
dependentes de tensor exigem uma janela explícita quando os shapes de runtime
não foram fornecidos. Alterar metadados redundantes da torre não pode mudar a
vista, enquanto alterar AST, inputs, domínio ou predicado invalida ou muda a
auditoria de modo observável. A saída continua indisponível quando a agenda do
provider é `runtime-defined`.

O trace autoritativo usado para investigar essa fronteira também precisa ser
fechado sobre os operandos reais do kernel. `runtimeReductionCoverage` enumera,
em ordem do programa, cada BMM `runtime-defined`, sua saída nativa e os dois
produtores de entrada com shapes capturados. A cobertura é reconstruída do
programa literal durante o replay source-removed; um checkpoint de Q/K anterior
ao RoPE, um contexto anterior ao blocking, uma entrada faltante ou um shape
alterado não pode substituir o operando declarado. O envelope do trace inclui
nonce de captura e SHA-256 de `config.json` e `model.safetensors`, que precisam
coincidir com `sourceIdentity`. Essa cobertura autoriza medir hipóteses de
redução; ela não transforma uma coincidência numérica em agenda do provider.

No schema v48, a preparação executável dessas reduções deixa de reconstruir o
ambiente por classe no host. Cada `invocationProgram` schema v2 declara o dtype
de runtime e uma lista fechada de bindings de torre com `source`, domínio
`safe-integer` e `minimumInclusive`. O provider materializa somente esses
bindings a partir do programa Gemma 4 e o adapter incorporado revalida as mesmas
declarações antes dos stages. Binding ausente, extra, duplicado, fora do domínio
ou com origem alterada falha antes de `torch.matmul`. A agenda escalar do SGEMM
permanece `runtime-defined`; esta versão fecha o ambiente de invocação sem
confundi-lo com a árvore de redução ainda não publicada.

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

No schema v30, a presença opcional de inputs também deixa de ser controle
secreto do leitor. `forwardControl` declara, em ordem, os grupos completos de
imagem, vídeo e áudio, as operações e invocações executadas quando presentes e
o alias identidade exato quando ausentes. Ele seleciona ainda o programa de
máscaras visual, a máscara aditiva do chamador ou a máscara causal/cache
textual; fixa inputs incompatíveis; e declara os estados explícitos para
`position_ids` ausente e `past_key_values` ausente/presente. Tanto replay quanto
navegação devem interpretar esse objeto. O grafo completo continua contendo
todas as definições, mas uma invocação avalia somente os ramos selecionados e
os aliases de ausência serializados — nunca uma decisão reconstruída por
convenção do host.

No schema v31, a lista humana de inputs é ligada a um `inputContract`
executável. O contrato fixa representação, dtype lógico, rank e eixos de todas
as 13 entradas; domínio de tokens/posições; shapes relativos de posições,
`mm_token_type_ids` e máscara aditiva; e propriedade/shape BHSD de cada cache
produtor. As modalidades não dependem de uma validação escondida na torre:
imagem e vídeo declaram largura patchificada, coordenadas/padding, pooling,
flatten e cardinalidade de placeholders; áudio declara máscara, dois
subsamplings stride 2, canais, largura da projeção e cardinalidade após os dois
slices. O leitor deve validar esse objeto canônico e executá-lo antes do
`forwardControl`. Adulterar o catálogo, rank, buffer, domínio, shape relacionado,
propriedade de cache ou cardinalidade é erro, nunca um default do host.

No schema v32, `outputContract` torna a fronteira pública do cálculo igualmente
executável. Para forward, ele liga `hidden_states_0`, `ple_inputs` e logits aos
produtores declarados, fixa dtype F32, layout row-major e shapes derivados dos
inputs/configuração, e exige que o estado KV contenha apenas produtores no
layout BHSD com comprimento `past_sequence + input_sequence`. Para geração,
ele declara tokens, logits de seleção, snapshots KV e estado terminal como
funções de `executed_steps`; zero passos deve devolver exatamente o prefill e
um ou mais passos deve devolver o cache do último forward incremental. Assim,
um leitor não pode aceitar um logit com vocabulário divergente, um cache de
consumer, um snapshot omitido ou um terminal selecionado por convenção do
host. O caminho síncrono e o paginado executam o mesmo objeto serializado.

No schema v33, a mesma regra cobre a cadeia inteira de geração, não apenas suas
extremidades. `generation_logits_append` preserva os logits F32 completos de
cada `forward_state[step+1]` em `step_forward_logits[step]`, alinhados ao
snapshot KV do mesmo estado. O contrato exige que o primeiro logit de seleção
seja o prefill, que cada seleção posterior seja bitwise idêntica ao snapshot
anterior, que `generated_token_ids[step]` seja o argmax crescente desse tensor
e que logits/cache terminais sejam os snapshots do último forward incremental.
Assim um leitor não pode combinar token, logits e cache vindos de estados
diferentes e ainda chamar a geração de reprodução literal.

No schema v34, a navegação de uma fórmula até seus predecessores deixa de
depender de procurar nomes dentro do texto. Cada predecessor do grafo inclui
`accesses` em ordem da primeira leitura, com uma destas formas:

- `tensor-element`: expressão completa e uma entrada por coordenada;
- `tensor-shape`: eixo exato consultado pelo programa; ou
- `whole-value`: leitura estruturada/de controle sem indexação tensorial.

Uma lista vazia só é válida com `scalarUse=shape-or-control-only`; ela declara
honestamente que o predecessor participa do domínio ou do controle, não da
expressão escalar. A vista de uma coordenada materializa os mesmos vínculos em
`predecessorCoordinates`, preserva o `producerOperationId` e distingue
`complete` de `windowed`. O parser de notação é único para todas as operações,
shapes, dtypes e layers; uma lista de IDs de atribuição seria incompatível com
este contrato.

No schema v35, a mesma regra é bidirecional. Toda atribuição declara
`outputCoordinate.write`, que precisa ser a única ocorrência tensorial do
output no lado esquerdo de `=`, e `shapeAssertions` para invariantes que leem o
shape da própria saída. `consumerCoordinates` repete os acessos downstream por
`operationId`; a lista de consumidores e a lista de vínculos coordenados devem
ter a mesma cardinalidade e ordem. Uma vista concreta materializa a escrita em
`renderedOutputCoordinate`, inclusive quando o resultado numérico de uma BMM
continua fail-closed.

O parser trata a notação incorporada `row_major_alias(x)[i,...]` como leitura
de `x[i,...]`. Isso não inventa semântica arquitetural: `row_major_alias` já é
um intrínseco layout-only do contrato de fórmulas. A regra vale para qualquer
atribuição e elimina a falsa classificação de reshape de heads como leitura do
tensor inteiro.

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

## Replay atestado de reduções nativas

Um replay que delega uma fronteira `runtime-defined` a um runtime autoritativo
também precisa ser auditável. O artefato fixa versão, commit de build, modo de
execução, device, plataforma e backend BLAS. Cada chamada deve corresponder a
uma única redução do programa e produzir um recibo que vincula ID/classe,
operandos ordenados e output por shape, bytes IEEE-F32 little-endian e
SHA-256. Runtime diferente, recibo alterado, valor não finito, shape divergente
ou cardinalidade diferente do grafo ativo falha fechado. Esse recibo prova qual
kernel reproduziu o valor; ele não substitui uma árvore escalar ainda não
publicada.

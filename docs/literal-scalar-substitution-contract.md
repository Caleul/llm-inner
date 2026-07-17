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
enumera o programa Gemma4Text na ordem de dependência e preserva, para cada
atribuição, seu output, predecessores produtores, consumidores e vizinhos.

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

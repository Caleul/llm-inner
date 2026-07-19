# Gemma 4 E4B: navegação de coordenadas predecessoras — 2026-07-19

## Fronteira candidata

O schema v33 já continha fórmulas escalares e arestas entre produtores e
consumidores, mas uma vista concreta não dizia estruturalmente qual coordenada
do produtor deveria ser aberta a seguir. O leitor precisava procurar nomes de
tensores dentro de strings. O schema v34 fecha essa lacuna para a classe
completa de atribuições Gemma 4 sem dispatch por layer ou assignment ID.

`calculationGraph.assignments[*].predecessors[*]` agora incorpora:

- `producerOperationId`, quando o input possui produtor no grafo;
- cada acesso `tensor-element`, com expressão e coordenadas separadas;
- cada acesso `tensor-shape`, com o eixo consultado;
- `whole-value` para estruturas e controles não indexados; e
- `scalarUse=shape-or-control-only` quando não há leitura escalar.

As vistas escalares e a auditoria BMM expõem `predecessorCoordinates` com os
templates incorporados, acessos concretos e cobertura `complete` ou
`windowed`. O parser preserva índices aninhados e a ordem da primeira leitura.
O mesmo fechamento revelou e removeu uma atribuição de placeholder sem
consumidores, a aresta falsa desse placeholder para a substituição por PAD e a
aresta falsa de vision blocks para a máscara full causal.

## Artefato real e integridade

Fonte imutável: `google/gemma-4-E4B`, revisão
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16/F32 denso não
quantizado. O comando de geração foi:

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Resultado final:

- schema do artefato `34`, linguagem de fórmulas `19` e grafo de cálculo `2`;
- `21.388.118.020` bytes;
- SHA-256 `08a2c87c30711f5eea3d7df7fa4a3caf55f547de2544c67c95b9d8479e4a9aad`;
- `2.130` constantes e `15.992.314.836` bytes aprendidos incorporados;
- `2.708` atribuições forward, `3.491` predecessores e `3.361` arestas com
  produtor explícito;
- `3.488` predecessores escalares endereçados, `4.101` acessos de elemento,
  `55` acessos de shape, `189` leituras de valor completo e três dependências
  somente de shape/controle.

A auditoria source-present comparou as 2.130 constantes e todos os bytes
aprendidos com os seis arquivos da fonte. Storage fonte e literal produziram
SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O relatório `/private/tmp/llm-inner-loop92-v34-source-audit-final.json` tem
SHA-256 `c2f790340b9aa7777b4af6a80eba9ea9660141584adda2bfe318e2ca92b409b7`.

## Prova source-removed e navegação concreta

`./gemma-4-E4B-dense` foi movido fisicamente sob restore trap. Enquanto o
caminho estava ausente, o reader reabriu somente o JSON e validou o grafo
schema 2 inteiro. O resumo source-removed registrou 2.708 atribuições, 4.144
statements escalares, 1.462 reduções, 256 estágios e exatamente 100 reduções
runtime-defined. Relatório
`/private/tmp/llm-inner-loop92-v34-navigation-summary-final.json`, SHA-256
`aa0284a0c6cd2c67a49b6b90acec7039ae2e35eb9233d8711304606ddb693385`.

Uma janela real de `layer_0_q_proj[0,0,0]` materializou
`layer_0_attn_norm[0,0,0..3]`, ligou todos os termos ao produtor
`layer_0_input_norm`, decodificou os quatro weights BF16 e marcou a cobertura
`windowed` sobre o domínio completo 0..2559. Relatório SHA-256
`f8c5317f53dd9b65ef203b4b73bd76e9502478d8af881dea1c350758c33f59a1`.

A auditoria real de vision score ligou 64 termos aos produtores Q-RoPE e
K-RoPE, com cobertura completa. A auditoria de vision value ligou uma janela
dinâmica de três key patches aos produtores de attention weights e V-norm e a
marcou `windowed`. Seus relatórios têm SHA-256, respectivamente,
`ce4c0b345f49d9f4d3837bfccd792eaab6e79a0510fae9b9606713bb53c9fcaf` e
`d4ce794a0d837e5f214201cb4017a15a9601f4d12d8a0967059cce5580abce2f`.

A vista end-to-end source-removed cobriu as 2.708 operações, as 2.130
constantes (`2.076` alcançáveis e `54` explicitamente runtime-unreachable), 13
operações greedy para um passo e as mesmas 100 BMM fail-closed. Relatório
SHA-256
`f06aece04d676809b544b3d56481daea98e8472118c34e714f67057d78c3beb0`.

## Diferencial das três modalidades e limite preservado

A suite source-removed de imagem, vídeo e áudio passou novamente com tolerância
absoluta/relativa zero: três prefills e três gerações
`lossless-within-dtype`, nenhuma primeira divergência e 24 caches KV por
modalidade. Ela contou 32 + 32 + 36 BMM runtime-defined. O relatório
`/private/tmp/llm-inner-loop92-v34-composite-modality-suite.json` tem SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

Isso é evidência candidata da navegação de coordenadas e da remoção de
dependências mortas, não do checkpoint. A agenda escalar interna das 100 BMM
Apple Accelerate continua `unpublished-fail-closed`; portanto nenhuma saída
BMM literal foi inventada e `.agent-loop/checkpoints/gemma4-dense-lossless/`
não foi criado.

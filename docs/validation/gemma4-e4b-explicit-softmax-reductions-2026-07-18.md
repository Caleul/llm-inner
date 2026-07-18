# Gemma 4 E4B: reduções softmax explícitas — 2026-07-18

## Fronteira fechada

A revisão independente do schema v19 confirmou os programas unários
incorporados, mas encontrou uma declaração matemática ambígua: uma atenção de
texto armazenava um único campo `reduction` embora executasse quatro reduções
distintas, e as fórmulas de visão/áudio escondiam máximo e soma do softmax em
helpers como `score-max-*`, `max_k` e `max_context`.

O schema v20 resolve a classe inteira, sem dispatch por layer id:

- cada `scaled_dot_product_attention` declara, em ordem, `score-dot`,
  `softmax-maximum`, `softmax-exponential-sum` e `context-dot`;
- o caminho BF16 nativo usa as agendas ARM NEON BF16/F32 já publicadas nos dois
  dots e o fold PyTorch F32 de quatro lanes no máximo e na soma;
- `masked-softmax` de visão e `chunked-relative-attention-softmax` de áudio
  declaram estágios F32 ordenados, domínios completos, identidades e predicados;
- `formulaLanguage.reductions.softmaxPrograms` incorpora os programas escalares
  de máximo/soma ordenados e os programas PyTorch de inicialização, lanes,
  cauda e fold; programa ausente, alterado ou compactação vetorial implícita por
  predicado falha fechado;
- fórmulas e vistas escalares rejeitam todos os antigos helpers opacos.

Na exportação real há 42 atenções de texto com quatro estágios cada, 32
softmaxes de visão e 12 de áudio com dois estágios cada. Os programas aparecem
84 vezes como `ARM_NEON_BF16_DOT_F32`, 42 vezes cada como
`PYTORCH_F32_VECTOR_REDUCE_MAX/SUM` e 44 vezes cada como
`ORDERED_F32_REDUCE_MAX/SUM`. A contagem de helpers opacos é zero.

## Artefato real

Fonte imutável: `google/gemma-4-E4B`, revisão
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
quantizado. Comando de geração:

```bash
npm run build
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Resultado:

- schema do artefato: `20`; schema da linguagem de fórmulas: `6`;
- bytes: `21,384,709,153`;
- SHA-256: `62264f7fd990061cdb1d2de521f21cc2cc40ee04743eb199c853eba5ad3db68c`;
- constantes/decoders: `2,130/2,130`;
- bytes aprendidos incorporados: `15,992,314,836`;
- atribuições forward instanciadas: `2,709`.

A auditoria source-present verificou os 2.130 payloads contra os seis arquivos
da fonte: o SHA-256 concatenado de storage nos dois lados foi
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
Relatório `/private/tmp/llm-inner-loop76-v20-source-audit.json`, SHA-256
`1e9dfa9d504816d1008835c60112d0088ba7fef945eae8bec8f1109f6240d6ad`.

## Evidência source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente para fora do caminho
declarado sob um restore trap durante cada inspeção e replay. A vista
end-to-end verificou todos os payloads e bytes, registrou
`sourceCheckpointAccessed=false`, preservou o SHA-256 de storage acima e
materializou 2.709 operações forward e 12 operações de geração. Relatório
`/private/tmp/llm-inner-loop76-v20-source-removed.json`, SHA-256
`60a01834b7e34d28016f3b960d8df34daa0ff9f9d353b2d3ae053f74a475af28`.

Três vistas escalares source-removed demonstram os contratos instanciados:

- texto, `layer_0_attention[0,0,0]`, com quatro estágios e os programas ARM e
  PyTorch: `/private/tmp/llm-inner-loop76-v20-text-softmax.json`, SHA-256
  `920ac4ff62f82bef59856679b127c4a6022fab354326c65c9b226b0a0f13ab2f`;
- visão, com predicado derivado dos pares de `image_position_ids` e máximo/soma
  ordenados: `/private/tmp/llm-inner-loop76-v20-vision-softmax.json`, SHA-256
  `83c4d6f366de0896297750ac1e82224cfb6252b6aa069079f4c58b870d3dfcab`;
- áudio, com máximo/soma ordenados sobre o contexto completo:
  `/private/tmp/llm-inner-loop76-v20-audio-softmax.json`, SHA-256
  `b09ff2eac3c9914502f999e63d4fb223802029da6e23671a7356b2f06be520f6`.

O replay do trace composite de áudio autoritativo, ainda sem a fonte, usou
tolerâncias absoluta e relativa zero. Prefill e geração permaneceram
`lossless-within-dtype`, sem primeira divergência; token 184 na posição 2,
logits e os 24 caches KV passaram. Relatório
`/private/tmp/llm-inner-loop76-v20-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Validação e limite preservado

`npm run typecheck` passou. `npm test` passou 234 testes, zero falhas.

Esta é uma alegação candidata de fechamento da semântica de softmax, não do
checkpoint Gemma 4. A vista end-to-end permanece
`fail-closed-runtime-reduction`: 64 BMMs de visão e 36 BMMs de áudio ainda não
possuem árvore escalar publicada pelo Apple Accelerate. O diretório
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

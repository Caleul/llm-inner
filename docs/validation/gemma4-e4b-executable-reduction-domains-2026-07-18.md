# Gemma 4 E4B: domínios executáveis de redução — 2026-07-18

## Fronteira candidata

O loop 83 revisou independentemente o commit
`adf79ccd2a1584aa26ba5febb9e2d20fd2772161` e repetiu
`npm run typecheck && npm test`: os 234 testes passaram. Isso aceita a
fronteira de dimensões executáveis do schema v26 dentro do escopo declarado
pelo loop 82; não aceita o checkpoint Gemma 4.

A revisão encontrou a mesma dependência interpretativa nos domínios de
redução. As fórmulas diziam, por exemplo, `0..in_features-1`,
`0..patches-1`, `0..context-1` e `0..K-1`, mas o JSON não ligava esses aliases
a um eixo concreto no call site. O bound matemático era visível, porém ainda
exigia que um leitor inferisse o significado do nome.

O schema v27 incorpora `gemma4-literal-reduction-domain-language` e liga cada
índice de redução a uma destas duas expressões fechadas:

- `constant`: inteiro positivo e seguro incorporado no JSON;
- `tensor-axis`: `tensorShapes[tensor][axis]`, depois que o grafo instanciado
  liga o nome local ao `orderedInput` real do call site.

Cada domínio declara `startInclusive=0`, `endExclusive` e ordem ascendente. A
construção e o leitor recusam índice duplicado, extent zero/inseguro, eixo ou
tensor ausente, referência fora de `orderedInputs`, linguagem alterada e
divergência entre o índice humano e seu AST. A regra cobre em um único dispatch
por classe:

- linear/clipped-linear e RMSNorm;
- Conv2d, LayerNorm e convolution depthwise;
- pooling vision;
- score/value BMM e softmax vision;
- os três BMM e softmax chunked de áudio;
- os quatro estágios de atenção de texto, incluindo o eixo `K` obtido do mask
  já ligado ao cache.

Nenhum ID de camada participa do contrato. As 100 BMM nativas recebem bounds
executáveis, mas permanecem `runtime-defined` porque o schema não inventa uma
árvore de acumulação Apple Accelerate não publicada.

## Artefato real e integridade

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16/F32 denso não
quantizado.

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Resultado:

- schema do artefato `27`, fórmula schema `13`, cálculo escalar schema `2`;
- bytes: `21.385.561.858`;
- SHA-256 do artefato:
  `b4e3fe3cfd5d544fa2ceed0b78d533e6bbf19a0fb39391ac9f0c4f9c0b225293`;
- constantes/decoders: `2.130/2.130`;
- bytes de payload incorporados: `15.992.314.836`.

O audit source-present:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop83-v27-source-audit.json \
  --verify-gemma4-payloads
```

Todos os 2.130 payloads compararam iguais. Source e literal produziram o mesmo
SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`
para `15.992.314.836` bytes. O relatório tem SHA-256
`7ca8d04a9cc3976b0b930619364cd1c5b9bf96c6e1da945f9facc5be4302c145`.

## Leitura e avaliação source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente sob restore trap. Com
o caminho ausente, o leitor correto do artefato composto executou:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop83-v27-source-removed.json
```

O relatório, SHA-256
`9e0ba53fd3d12216d7b90a6179a3db0c4081e1f1ef66a27fb365128e2e43dce8`,
registrou:

- `sourceCheckpointAccessed=false`;
- 2.130 payloads e 15.992.314.836 bytes verificados;
- 2.709 operações forward e 20 operações de geração;
- 1.462 reduções, 256 estágios e 1.722 domínios executáveis;
- exatamente 100 reduções `runtime-defined`: 64 vision e 36 audio.

O evaluator do contrato foi exercitado com shapes reais e também falha para
tensor/eixo ausente, linguagem alterada e extent inválido. Slices e vistas
escalares carregam o mesmo `formulaLanguage.reductions.domainLanguage` e os
domínios ligados do grafo, sem consultar o checkpoint.

## Regressão autoritativa

Ainda com a fonte fisicamente ausente, o artefato foi comparado em tolerância
absoluta e relativa zero com a captura
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`:

```bash
node dist/src/gemma4-literal-composite-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop83-v27-composite-audio.json \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --allow-unverified-fidelity
```

O relatório tem SHA-256
`30b4b49783f542ab2540269cd7530391c4eaf37d0f5ce31b4bfb68b21e32f241`.
Prefill e geração foram `lossless-within-dtype`: erro absoluto/relativo máximo
zero, cosine/top-k `1`, argmax agreement, token greedy `184` na posição `2`,
todos os 24 caches produtores e nenhuma primeira divergência. Durante essa
execução, o evaluator resolveu 1.138 domínios de 942 reduções e 192 estágios em
996 atribuições ativas. A cobertura inclui as 36 BMM de áudio, que continuam
explicitamente `runtime-defined` quanto à ordem escalar.

## Limite preservado

Esta mudança remove inferência dos extents, não a ausência de uma especificação
da árvore SGEMM nativa. As 64 BMM vision e 36 BMM audio permanecem
`fail-closed-runtime-reduction`; o uso de `--allow-unverified-fidelity` continua
explícito. Este loop implementa o schema v27 e portanto registra apenas
evidência candidata. `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi
criado.

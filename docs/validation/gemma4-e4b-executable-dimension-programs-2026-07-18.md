# Gemma 4 E4B: programas executáveis de dimensão — 2026-07-18

## Fronteira candidata

O loop 82 revisou independentemente o commit `5a70b1beed7d767b776821a63b8c9f7bd812d96e`
e repetiu `npm run typecheck && npm test`: 234 testes passaram. Isso aceita a
clausura de operandos do schema v25 dentro do escopo declarado pelo loop 81;
não aceita o checkpoint Gemma 4.

A inspeção seguinte encontrou uma dependência interpretativa no próprio
artefato. `calculationDomains.dimensionDefinitions` ainda descrevia shapes com
prosa, incluindo “past key length plus S”, `ceil(AT0/2)`, divisão de pooling e
“number of true rows”. As fórmulas eram navegáveis, mas um leitor independente
precisava transformar frases em aritmética antes de enumerar todos os valores
de cada saída.

O schema v26 substitui essa tabela pela linguagem incorporada
`gemma4-safe-integer-dimension-expression-v1` e por 26 programas AST. O
contrato cobre, para todas as definições composite/vision/audio/text:

- leitura explícita de eixo de tensor e comprimento KV opcional;
- dependências entre dimensões avaliadas uma única vez;
- contagem BOOL row-major para `strip-padding`;
- soma, multiplicação, ceil-division e divisão exata com inteiro seguro;
- shapes compostos formados somente por literal, dimensão registrada e
  multiplicação exata;
- avaliação lazy de uma dimensão, sem exigir inputs de modalidades não usadas;
- rejeição de ciclo, nome/tensor ausente, eixo inválido, valor negativo,
  overflow, divisor zero ou divisão não exata.

Slices, vistas escalares e o cálculo end-to-end carregam a linguagem e os
programas diretamente do JSON. Nenhum dispatch por layer/assignment foi
introduzido.

## Artefato real e integridade

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
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

- schema do artefato `26`, fórmula schema `12`, dimensão schema `1`;
- bytes: `21.385.039.328`;
- SHA-256 do artefato:
  `862b12576e87025c1d2f01d6ee9612e9315e7db6137c0b235342fcc42eed173c`;
- constantes/decoders: `2.130/2.130`;
- bytes aprendidos incorporados: `15.992.314.836`.

O audit source-present comparou todos os payloads:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop82-v26-source-audit.json \
  --verify-gemma4-payloads
```

Storage source e literal produziram o mesmo SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`
para `15.992.314.836` bytes. O relatório tem SHA-256
`a67398010f0a31b2efb15ce84fe710eb5d127bf0f322c9308e881b7bb66abe07`.

## Prova source-removed e dimensões reais

O diretório `./gemma-4-E4B-dense` foi movido fisicamente sob restore trap. Com
o caminho ausente, o leitor executou `--verify-payloads`,
`--end-to-end-calculation` e dois passos de navegação greedy. O relatório
`/private/tmp/llm-inner-loop82-v26-source-removed.json` tem SHA-256
`7ac91e8f9116b4f7c1ef3be86a8c0a4a77e7feac2efb0316033d164432621373` e
registrou:

- `sourceCheckpointAccessed=false`;
- 2.130 compromissos e 15.992.314.836 bytes verificados;
- 2.709 operações forward, 20 operações de geração e 42 transições KV;
- 2.609 operações literais e as mesmas 100 BMM fail-closed;
- os 26 programas de dimensão dentro do slice de logits/end-to-end.

Usando apenas esse relatório e o input de áudio registrado
`input_ids.shape=[1,2]`, `input_features.shape=[1,1,128]` e
`audio_output_mask=[true]`, o executor de dimensões obteve:

```json
{
  "B": 1,
  "S": 2,
  "K": 2,
  "AB": 1,
  "AT0": 1,
  "AF0": 128,
  "AT1": 1,
  "AF1": 64,
  "AT2": 1,
  "AF2": 32,
  "ABLOCKS": 1,
  "AVALID": 1,
  "audioFlattenedWidth": 4096
}
```

Sem cache, `K` usa o `whenAbsent=0` declarado e resulta em `S`; inputs ausentes
de imagem/vídeo não são consultados durante essa avaliação lazy.

## Regressão autoritativa

Ainda com a fonte fisicamente ausente, o schema v26 foi comparado em tolerância
absoluta e relativa zero com a captura
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`:

```bash
node dist/src/gemma4-literal-composite-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop82-v26-composite-audio.json \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --allow-unverified-fidelity
```

Prefill e geração foram `lossless-within-dtype`: as cinco fronteiras composite,
logits, token `184` na posição `2` e todos os 24 caches produtores passaram com
erro absoluto/relativo máximo zero, cosine/top-k `1`, argmax agreement e nenhuma
primeira divergência. O relatório tem SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

As validações finais configuradas também passaram: `npm run typecheck` saiu
com código zero e `npm test` executou 234 testes, com 234 passes e zero falhas.

## Limite preservado

Os programas de dimensão removem prosa dos bounds e shapes, mas não publicam
uma árvore numérica inexistente. As 64 BMM de visão e 36 BMM de áudio continuam
`unpublished-fail-closed`; o diferencial de áudio continua evidência candidata
sob reconhecimento explícito de fidelidade ainda não verificada para essas
classes. Este commit implementa o schema v26 e portanto não pode certificá-lo
independentemente. `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi
criado.

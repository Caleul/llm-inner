# Gemma 4 E4B: classes nativas explícitas do áudio — 2026-07-18

## Classificação

Esta é evidência candidata de avanço do artefato matemático Gemma 4, não uma
aceitação do checkpoint. O replay source-removed produz a saída terminal de
áudio exatamente para a entrada registrada, mas 11 scores internos ainda
divergem e a agenda nativa das reduções F32 de score/value permanece
deliberadamente `runtime-defined`. Por isso a classificação do ramo continua
`approximate` e `.agent-loop/checkpoints/gemma4-dense-lossless/` não existe.

## Identidade reproduzível

- modelo: `google/gemma-4-E4B`
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`
- formato: Safetensors BF16 denso, sem quantização
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`
- runtime autoritativo:
  `transformers-5.5.0/torch-2.12.1-Gemma4Audio-CPU-eager`
- build PyTorch: `7269437d655783a26cba32aa88195b741ff496aa`, CPU macOS
  ARM com Accelerate BLAS, `USE_MKLDNN=OFF`

O profiler autoritativo despachou os dois Conv2d e as 12 depthwise para
`aten::_slow_conv2d_forward`, as LayerNorm de canal para
`aten::native_layer_norm`, os lineares sem bias para `mm` e o linear de saída
com bias para `addmm`. O executor aplica esses contratos a todas as atribuições
compatíveis, nunca por ID de camada:

- slow-convolution BF16: quatro acumuladores F32 intercalados e fold ascendente;
- LayerNorm BF16: Welford vetorial, merge low/high, fold de lanes e segundo
  passe `x*scale+bias`, seguido de gamma;
- lineares BF16 sem bias: dot ARM NEON BF16/F32 já registrado;
- linear BF16 com bias: dot ARM seguido de soma F32 do bias;
- Q: `F32(F32(q * F32(q_scale)) * BF16(softplus(per_dim_scale)))`;
- K: multiplicação F32 pelo escalar registrado;
- posição relativa: casts BF16 explícitos em inverse-timescale, produto de
  posição e saídas SLEEF sin/cos;
- atenção: score, softmax, value e cast BF16 do contexto são atribuições
  navegáveis distintas; score/value F32 falham fechados no scalar view até que
  sua redução nativa seja comprovada.

## Integridade do artefato regenerado

```bash
npm run build
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop51-final-audit.json \
  --verify-gemma4-payloads
```

Resultado real:

- artefato: `artifacts/gemma4-e4b-dense.literal.json`
- bytes: `21,328,116,225`
- SHA-256: `4a95d53e04941e125e06529614aadacc2f3261c070396248599c2f83d3bfdab8`
- constantes incorporadas: `2,130`
- bytes de payload comparados: `15,992,314,836`
- SHA-256 agregado da origem e do literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`
- caminho proibido da origem serializado: `false`

## Diferencial autoritativo expandido

Entrada determinística: `input_features` F32 `[1,1,128]`, linearmente espaçada
de `-0.25` a `0.25`, com máscara verdadeira. A captura executa primeiro um
forward não instrumentado, depois o forward instrumentado, e exige igualdade
bit a bit de `pooler_output` e `attention_mask` antes de aceitar os 372
checkpoints.

```bash
npm run capture:gemma4-audio-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop51-audio-qk-scaled.json \
  --frames 1 --feature-start=-0.25 --feature-end=0.25 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Com `gemma-4-E4B-dense` renomeado e indisponível, foram executados
`--verify-payloads --assert-source-unavailable` e:

```bash
node dist/src/gemma4-literal-audio-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop51-audio-qk-scaled.json \
  --report /tmp/gemma4-loop51-final-audio-source-removed.json \
  --max-tensor-mib 16 --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

Resultado real:

- `sourceCheckpointAccessed: false`;
- 372 operações presentes, 361 exatas em tolerância zero;
- todos os 12 Q escalados e 12 K escalados exatos;
- todos os 12 softmax, contexts, casts BF16 e consumidores posteriores exatos;
- score da camada 2 exato;
- 11 scores divergentes, erro absoluto máximo
  `5.7220458984375e-6`;
- primeira divergência: `audio_layer_0_attention_scores`;
- saída `audio_features [1,2560]`: erro absoluto `0`, erro relativo `0`,
  cosseno `1`, top-k overlap `1`, argmax agreement verdadeiro;
- fidelidade global: `approximate`.

O resultado terminal exato não permite apagar a divergência interna. O limite
restante é uma única classe semântica: as reduções F32 nativas usadas pelos
batched matmuls de score/value. O artefato registra esse limite como
`runtime-defined` em todas as camadas e o scalar view o rejeita, em vez de
substituí-lo por uma soma ascendente não comprovada.

## Regressão de imagem

O mesmo artefato regenerado reproduziu novamente a captura image eager inteira:
294/294 operações, `lossless-within-dtype`, tolerâncias absoluta e relativa
zero e nenhuma primeira divergência. Isso preserva a evidência candidata do
ramo visual, sem transformar esta sessão na autoridade que aceita seu próprio
checkpoint.

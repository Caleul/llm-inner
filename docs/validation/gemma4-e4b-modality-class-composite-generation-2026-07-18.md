# Gemma 4 E4B: geração composite por classe de modalidade — 2026-07-18

## Classificação

Esta sessão revisou independentemente o commit `15164ea` com typecheck e a
suíte existente antes de ampliar a fronteira composite. A captura e a
comparação antes codificadas somente para imagem agora usam um único contrato
discriminado para `image`, `video` e `audio`. Cada perfil fixa token type,
input, feature output, scatter e as mesmas fronteiras posteriores de PLE,
logits e geração. Inputs mistos, shapes incompatíveis e roteamento contraditório
falham antes de qualquer acesso ao modelo.

O resultado é evidência candidata `lossless-within-dtype` para uma invocação
real de vídeo desde pixels até geração cached, e a primeira localização
end-to-end da divergência de áudio no próprio programa composite. O marcador
de checkpoint continua proibido: os matmuls nativos internos de vídeo e áudio
continuam sem agenda escalar geral, e a invocação de áudio é aproximada.

## Identidade vinculada

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- runtime: `transformers-5.5.0/torch-2.12.1`, CPU eager BF16;
- pacote: Safetensors BF16 denso, sem quantização;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- artefato literal: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,328,161,289`;
- SHA-256 verificado nesta sessão:
  `1bd3ef0638b85cefc92346a5993e301228302133021a6bdf24d81cd3bc2adc07`.

## Contrato de captura

O helper autoritativo executa primeiro um forward nativo não instrumentado.
Depois captura `composite_text_embeddings`, o feature output da modalidade,
`hidden_states_0` após scatter, `ple_inputs` e `softcapped_logits`. A captura é
recusada se instrumentação alterar logits ou cache. O mesmo trace carrega
logits de seleção, token e posição greedy, 24 caches produtores pós-decode,
logits terminais e 24 caches terminais.

O perfil é aplicado como classe completa:

| perfil | token type | feature | scatter |
|---|---:|---|---|
| image | 1 | `image_features` | `composite_image_scatter` |
| video | 2 | `video_features` | `composite_video_scatter` |
| audio | 0 | `audio_features` | `composite_audio_scatter` |

Imagem foi recapturada pelo novo contrato com `[1,9,768]`, posições 3x3,
prompt `[258880,2]` e `max_new_tokens=1`:

```bash
npm run capture:gemma4-composite-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop54-image-composite-trace.json \
  --modality image --modality-token-id 258880 --text-token-id 2 \
  --max-new-tokens 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Vídeo usou um frame `[1,1,9,768]`, posições 3x3, prompt
`[258884,2]`, posições `[0,1]` e `max_new_tokens=1`:

```bash
npm run capture:gemma4-composite-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop54-video-composite-trace.json \
  --modality video --modality-token-id 258884 --text-token-id 2 \
  --frames 1 --max-new-tokens 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Áudio usou um frame `[1,1,128]`, mask verdadeira, prompt `[258881,2]`
e features F32 entre `-0.25` e `0.25`:

```bash
npm run capture:gemma4-composite-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop54-audio-composite-trace.json \
  --modality audio --modality-token-id 258881 --text-token-id 2 \
  --frames 1 --max-new-tokens 1 \
  --feature-start -0.25 --feature-end 0.25 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Os traces de imagem, vídeo e áudio têm `8,232,709`, `8,232,719` e `8,102,087`
bytes, com SHA-256
`ed9f2a16fbf913285a932249f29ce6ce85222f2313b1d32103a8f6720c340912`,
`a82eef33459c21b9a41416c504cb7ce76f7ab4eb67a74522b5eb662169ba6a28`
e `b8a7fa5732de6245d914a80fa41283f7aa5c0357a624cab90bffff040ac11586`.

## Replay com fonte ausente

`gemma-4-E4B-dense` foi renomeado durante as duas comparações; o CLI exigiu
`--assert-source-unavailable` e ambos os relatórios declaram
`sourceCheckpointAccessed=false`:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop54-<video|audio>-composite-trace.json \
  --report /tmp/gemma4-loop54-<video|audio>-composite-source-removed-report.json \
  --max-read-mib 16 --max-tower-tensor-mib 64 \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

### Imagem e vídeo

Nos dois perfis, as cinco fronteiras passaram com erro absoluto e relativo
zero: embedding, `image_features`/`video_features [1,2560]`, scatter,
`ple_inputs [1,2,42,256]` e logits
`[1,2,262144]`. Cosseno, top-k e argmax são `1` em todas. O programa selecionou
o mesmo token `184` na posição `2`; logits de seleção, todos os `24/24` caches
pós-decode, logits terminais `[1,1,262144]` e todos os `24/24` caches terminais
também têm erro zero. Prefill e geração são `lossless-within-dtype` para essa
entrada. Os relatórios de imagem e vídeo têm SHA-256
`b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e`
e `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d`.

### Áudio

Embedding textual é exato, mas a primeira divergência é
`composite_audio_features [1,2560]`: erro absoluto máximo `0.08203125`, relativo
`20.28301886792453`, cosseno `0.9999220339355607`, top-k `1` e argmax igual.
Scatter preserva esse máximo; PLE chega a `0.375`; logits de seleção chegam a
`2`, com cosseno `0.9998388707060447`, top-k `0.9` e argmax igual. O token
greedy ainda é `184` na posição `2`, mas nenhum dos 24 caches é bitwise exato.
Logits terminais têm erro máximo `0.3125`, cosseno `0.9998730821941509`, top-k
`1` e argmax igual. A classificação permanece `approximate`. O relatório tem
SHA-256
`7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4`.

## Limite preservado

O vídeo fecha a antiga ausência de evidência composite ampla, mas a igualdade
terminal de uma entrada não declara a agenda escalar dos dois batched matmuls
internos anteriormente divergentes. O áudio confirma em uma fronteira ampla
que a divergência antecede scatter, PLE e texto e começa no feature encoder;
isso é compatível com os 36 `sgemm` de áudio ainda `runtime-defined`. Nenhuma
agenda foi escolhida por layer ID, shape ou resultado terminal, e as vistas
escalares dessas classes continuam fail-closed.

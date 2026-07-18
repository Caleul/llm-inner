# Gemma 4 E4B: visão eager explícita e geração composite sem checkpoint — 2026-07-18

## Classificação

Esta é evidência candidata de fidelidade `lossless-within-dtype` para uma
invocação real de imagem e para prefill + um passo greedy do programa composite
real, executados somente do JSON literal. Ela não autoriza ainda
`.agent-loop/checkpoints/gemma4-dense-lossless/`: uma invocação real de vídeo
possui duas divergências internas de batched matmul e áudio continua aproximado.
Uma sessão posterior também precisa revisar esta implementação antes de aceitar
qualquer gate, conforme o contrato de revisão independente.

## Identidade vinculada

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- pacote: Safetensors BF16 denso, sem quantização;
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- runtime: `transformers-5.5.0/torch-2.12.1` em CPU com
  `attn_implementation="eager"` exigido pelo helper;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,328,083,855`;
- SHA-256: `daa029dd3e635b6c7757a1db6979ea7f81d2b8c8b49a193b6574a58ea5d13cb4`.

O audit fonte-para-literal comparou os 2.130 payloads e todos os
`15,992,314,836` bytes incorporados. Fonte e literal produziram o mesmo digest
de storage:
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
Com o diretório fonte renomeado, a verificação interna repetiu esse digest e
declarou `sourceCheckpointAccessed=false`.

## Correção da autoridade de captura

A captura anterior se identificava como eager, mas `from_pretrained` não
recebia `attn_implementation="eager"`; a configuração efetiva do vision tower
era `sdpa`. O helper agora seleciona eager explicitamente, valida
`vision_config._attn_implementation`, instrumenta a função eager real e falha
se qualquer uma das fronteiras esperadas estiver ausente.

Cada uma das 16 atenções foi decomposta no programa de 427 atribuições em:

1. `attention-score-matmul`, com saída BF16 e redução nativa ainda
   `runtime-defined`;
2. `masked-softmax`, com máximo, exponencial, soma de chaves ascendente e cast
   BF16 explícitos;
3. `attention-value-matmul`, com saída BF16 e redução nativa ainda
   `runtime-defined`.

O pooling espacial foi separado e recebeu a agenda comprovada
`ordered-fma`: peso `F32(1/F32(9))`, patches em ordem ascendente e cast BF16.
Todas as RMSNorm compartilham a transcrição de `torch.pow(x,-0.5)` como
`F32(1/F32(sqrt(x)))`. Essas decisões são por classe de operação e atingem
todas as camadas compatíveis; nenhum ID de camada foi colocado em allowlist.

## Comparação vision de 294 fronteiras

As entradas determinísticas foram imagem `[1,9,768]` e vídeo `[1,2,9,768]`,
com valores F32 de `0` a `1` e posições 3x3. As capturas registraram 18
fronteiras por camada, além de patch embedding, pooling, strip, norma e
projeção finais.

```bash
npm run capture:gemma4-vision-checkpoints -- \
  --source ./gemma-4-E4B-dense --output /tmp/image.json \
  --invocation image --frames 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a

npm run compare:gemma4-literal-vision -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/image.json --report /tmp/image-report.json \
  --max-read-mib 16 --max-tensor-mib 64 \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --allow-unverified-fidelity
```

- imagem: 294/294 fronteiras passam; primeiro desvio `null`; erro máximo zero;
  `image_features [1,2560]` exato; `lossless-within-dtype`;
- vídeo: 292/294 fronteiras passam e `image_features [2,2560]` é exato; somente
  `vision_layer_8_attention` (`5.960464477539063e-8`) e
  `vision_layer_9_attention_scores` (`0.0000152587890625`) divergem;
  classificação `approximate`.

Agendas escalares simples com e sem produto arredondado produziram os mesmos
dois valores candidatos. Sem evidência para transcrever o kernel batched nativo
em toda a classe, score e value continuam fail-closed na vista escalar. Softmax
e pooling são navegáveis sem fonte; exemplos renderizados:

```text
attention_weights[0,0,0,0] = BF16(F32(exponential[0]/denominator))
denominator = F32(sum_k_ascending(exponential[k]))
vision_pooled[0,0,0] = BF16(ordered_F32_FMA_patches(source,F32(1/F32(9))))
```

Solicitar `vision_layer_0_attention_scores` no artefato BF16 falha com:

```text
vista escalar falha fechada porque a redução pytorch-native-batched-matmul
ainda não possui agenda literal comprovada
```

## Replay composite source-removed

O trace usa prompt `[258880,2]`, uma imagem `[1,9,768]`, posições `[0,1]` e um
token greedy. Seu SHA-256 é
`3fd047cf0530625751baa39eec1a77c4045800eba0fd46fbca333ceaf9a1747b`.
Durante verificação de payload e execução composite, o diretório
`gemma-4-E4B-dense` permaneceu renomeado e o CLI exigiu sua ausência:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop50-composite-trace.json \
  --report /tmp/gemma4-loop50-composite-source-removed-report.json \
  --max-read-mib 16 --max-tower-tensor-mib 64 \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

O relatório (`b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e`)
declarou `sourceCheckpointAccessed=false`. Embedding, `image_features`, scatter,
PLE e logits de prefill passaram com erro zero. O token greedy foi `184` na
posição `2`; logits de seleção, todos os caches KV produtores, logits terminais
`[1,1,262144]`, top-k e argmax também passaram com erro zero. Prefill e geração
foram classificados `lossless-within-dtype`.

## Limites preservados

- áudio ainda diverge em `audio_subsample_0_relu` e termina com erro absoluto
  `0.0625`; Conv2d/LayerNorm/depthwise/linear com bias continuam nativos e
  `runtime-defined`;
- vídeo preserva duas divergências internas de batched matmul, embora a saída
  terminal seja exata;
- score/value BF16 recusam expansão escalar até uma agenda nativa por classe
  ser comprovada;
- não foi criado marcador de checkpoint e esta sessão registra somente uma
  alegação candidata para revisão independente.

# Gemma 4 E4B: fronteira F32 direta do contexto de áudio — 2026-07-18

## Classificação

Esta sessão revisou independentemente o commit anterior com `npm run
typecheck` e `224/224` testes, preservou o dispatch `runtime-defined` dos 36
matmuls F32 e fechou uma lacuna de evidência em toda a classe de value matmul.
As 12 saídas F32 agora são capturadas diretamente depois de `attn_weights @
value_states` e antes do cast para o dtype BF16 da projeção `post`. A inspeção
source-removed também revelou e corrigiu nas 12 atribuições um contrato
inconsistente que declarava o input do cast como BF16; o artefato regenerado
agora declara `F32 -> BF16`, sem redução.

O checkpoint continua proibido. Uma entrada autoritativa prova todas as 12
saídas F32 e seus 12 casts BF16, mas não estabelece a árvore interna geral do
Apple Accelerate `sgemm`; a vista escalar continua falhando fechada nas 36
reduções nativas.

## Fonte e artefato

- modelo: `google/gemma-4-E4B`;
- revisão declarada na captura:
  `411aa17b749aa952df1359d2dcea73917a544d9a`;
- runtime: `transformers-5.5.0/torch-2.12.1`, CPU eager;
- pacote: Safetensors BF16 denso, sem quantização;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`;
- artefato literal: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes do artefato: `21,328,161,289`;
- SHA-256 do artefato:
  `1bd3ef0638b85cefc92346a5993e301228302133021a6bdf24d81cd3bc2adc07`.

O programa já continha as atribuições separadas
`audio_layer_{0..11}_attention_context` em F32 e
`audio_layer_{0..11}_attention_context_bf16`. Esta sessão regenerou o artefato
para corrigir o `dtypePolicy` de toda essa classe. A auditoria contra a fonte
comparou as `2,130` constantes e `15,992,314,836` bytes com o mesmo SHA-256
agregado `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
o relatório tem SHA-256
`c1f4057a2415ce7c88138c016d69dd7344d0f950de65a8af008988f40b3e243a`.
Com a fonte ausente, os mesmos payloads foram verificados apenas do JSON; o
relatório source-removed tem SHA-256
`e20b3229dcdfbb2188c8c213a8f838760fbee1be1964c2a9b96baa53519e15ed`.

## Captura autoritativa

```bash
npm run capture:gemma4-audio-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop53-audio-context-boundary.json \
  --frames 1 --feature-start=-0.25 --feature-end=0.25 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

A atenção instrumentada mantém o guard que compara `pooler_output` e
`attention_mask` bit a bit com um forward não instrumentado. O trace resultante
tem `444` operações, incluindo 12 contextos F32 diretos e 12 consumidores
BF16, e SHA-256
`ad762c44f758e44a17675640766827c7b18620a138dd3b571cc642fda5ec8f79`.

## Replay com fonte ausente

Com `gemma-4-E4B-dense` temporariamente renomeado:

```bash
npm run compare:gemma4-literal-audio -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop53-audio-context-boundary.json \
  --report /tmp/gemma4-loop53-audio-source-removed.json \
  --max-tensor-mib 16 --absolute-tolerance 0 --relative-tolerance 0 \
  --top-k 10 --assert-source-unavailable ./gemma-4-E4B-dense
```

O relatório tem SHA-256
`a9f5b6423b06fb8d9103ecc05f57eccaf0ac07c47ce60ccbb9fdabfff57d1dac`
e registra:

- `444` fronteiras comparadas, `374` exatas e `70` divergentes;
- erro absoluto máximo `0.0000457763671875` e relativo máximo
  `0.000012681605515882787`;
- primeira divergência em
  `audio_layer_0_attention_position_scores`;
- `12/12` contextos F32 diretos exatos, com erros absoluto e relativo zero;
- `12/12` contexts BF16 end-to-end exatos;
- `12/12` casts BF16 source-anchored exatos a partir dos contextos F32
  autoritativos;
- `48/48` estágios source-anchored de relative shift, add, softcap e mask
  exatos;
- `audio_features [1,2560]` exato, com cosseno `1`, top-k overlap `1` e
  argmax agreement verdadeiro.

A regressão de imagem contra o mesmo artefato regenerado, também com a fonte
ausente, permaneceu `294/294` em tolerância zero (`lossless-within-dtype`). O
relatório `/tmp/gemma4-loop53-image-source-removed-regression.json` tem SHA-256
`db409a3a5963a808959060473e88cedb26d570146255eb3545514785d44d0451`.

## Navegação escalar source-removed

Ainda com a fonte indisponível, a vista da atribuição
`composite_audio_features/audio_layer_0_attention_context_cast` na coordenada
`[0,0,0]` foi renderizada somente do artefato:

```text
composite_audio_features/audio_layer_0_attention_context_bf16[0,0,0]
  = BF16(composite_audio_features/audio_layer_0_attention_context[0,0,0])
```

O navegador liga o predecessor à atribuição
`composite_audio_features/audio_layer_0_attention` e o consumidor a
`composite_audio_features/audio_layer_0_self_attn.post`. O relatório
`/tmp/gemma4-loop53-audio-context-cast-scalar-regenerated.json` tem SHA-256
`7b04e50af7de1eedf0eb6e6cf09ff77c70ad0e71fba9978caa8611065da1dc1a`
e declara `inputDtype=F32`, `accumulationDtype=none` e `outputDtype=BF16`.
Pedir a vista escalar do próprio value matmul encerra com o erro explícito
`vista escalar falha fechada porque a redução pytorch-cpu-f32-matmul ainda não
possui agenda literal comprovada`, em vez de publicar uma soma fictícia.

Os 12 value matmuls coincidiram nesta entrada, mas permanecem corretamente
`runtime-defined`: uma amostra não transforma uma chamada BLAS opaca em uma
agenda escalar geral. Os 12 AC e 12 BD também permanecem fail-closed. Vídeo
ainda tem duas divergências internas de batched matmul, e geração composite
ampla com áudio/vídeo não está estabelecida. Por isso
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

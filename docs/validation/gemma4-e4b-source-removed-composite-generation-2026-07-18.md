# Gemma 4 E4B: composição imagem → logits → geração sem checkpoint — 2026-07-18

## Classificação

Esta é evidência candidata de execução do programa composite real somente a
partir do artefato literal. Ela fecha a ausência de um caminho que conectasse
uma torre multimodal incorporada ao PLE, às 42 camadas de texto, aos logits e
ao estado de geração serializado. Não é evidência de fidelidade exata e não
autoriza `.agent-loop/checkpoints/gemma4-dense-lossless/`: vision continua
aproximado e a divergência propaga para PLE, logits e caches.

## Identidade vinculada

- modelo: `google/gemma-4-E4B`
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`
- pacote: Safetensors BF16 denso, sem quantização
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- runtime autoritativo:
  `transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager`
- artefato literal: `artifacts/gemma4-e4b-dense.literal.json`
- bytes do artefato: `21,328,054,627`
- SHA-256 do artefato:
  `b33178b4d138367753ca29fa6a5936f64f9cb95d9f85407e2495cf87939be0b2`

## Fronteira implementada

`executeGemma4LiteralCompositeF32` não recebe catálogo nem mapa de tensores.
Ele usa exclusivamente o índice do JSON e executa, em ordem:

1. substituição dos IDs de imagem/vídeo/áudio pelo PAD declarado;
2. embedding textual e identidade PLE por ranges incorporados;
3. torre image/video e/ou audio incorporada;
4. scatter modal por cardinalidade exata;
5. projeção, escala, reshape, RMSNorm e combinação PLE **depois** do scatter;
6. máscaras full/sliding do bloco visual;
7. as 42 camadas textuais preparadas, logits e caches produtores.

O executor textual foi dividido em três contratos coesos: embeddings antes do
scatter, projeção PLE após o scatter e camadas/epílogo com prelude preparado.
Assim, a composição não repete o embedding standalone e não contém uma segunda
cópia das políticas BF16 de embedding, linear, RMSNorm ou texto.

`generateGemma4LiteralCompositeF32` injeta esse forward no interpretador das
doze atribuições de geração já serializadas no artefato. Imagem e
`mm_token_type_ids` existem somente no prefill; cada decode usa o token
selecionado, posição avançada e cache pós-RoPE, como o programa declara.

## Captura autoritativa

O prompt é `[258880, 2]`: um placeholder de imagem e um token textual. A
imagem sintética declarada tem shape `[1,9,768]`, valores F32 de `0` a `1` e
posições 3x3. `mm_token_type_ids=[1,0]`, posições `[0,1]` e
`max_new_tokens=1`.

O helper compara o prefill instrumentado com um forward nativo sem hooks e
recusa a captura se logits ou cache mudarem. Ele persiste cinco fronteiras:
embedding PAD, `image_features`, embeddings após scatter, `ple_inputs` e
logits softcapped; também persiste logits de seleção, token/posição, 24 caches
pós-decode, logits terminais e 24 caches terminais.

```bash
npm run capture:gemma4-composite-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop49-composite-trace.json \
  --image-token-id 258880 --text-token-id 2 \
  --max-new-tokens 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

O runtime autoritativo selecionou token `184` na posição `2`.
O trace tem SHA-256
`3fd047cf0530625751baa39eec1a77c4045800eba0fd46fbca333ceaf9a1747b`.

## Replay com a fonte ausente

Durante toda a execução candidata, `gemma-4-E4B-dense` esteve renomeado e o
CLI exigiu sua ausência:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop49-composite-trace.json \
  --report /tmp/gemma4-loop49-composite-source-removed-report.json \
  --max-read-mib 16 --max-tower-tensor-mib 64 \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

O relatório declarou `sourceCheckpointAccessed=false`. Em tolerância zero:
seu SHA-256 é
`d282960e59b1afa5e36103a1b620da732e3bf1c825d568ac9423cf1ea616b983`.

| fronteira de prefill | status | erro abs. máximo | cosseno | top-k | argmax |
|---|---|---:|---:|---:|---|
| `composite_text_embedding` | pass | 0 | 1 | 1 | igual |
| `composite_image_features` | diverged | 0.044921875 | 0.9998542606687334 | 1 | igual |
| `hidden_states_0` após scatter | diverged | 0.044921875 | 0.999954502339819 | 1 | igual |
| `ple_inputs` | diverged | 0.375 | 0.9999814032502666 | 1 | igual |
| `softcapped_logits` | diverged | 1.03125 | 0.999358797360333 | 1 | igual |

A geração preservou prompt, token `184`, posição `2`, top-k `1` e argmax dos
logits de seleção. Os logits de seleção herdaram erro máximo `1.03125`. Os
logits terminais `[1,1,262144]` tiveram erro máximo `0.28125`, cosseno
`0.9999271922285432`, top-k `0.9` e argmax igual. Todos os 24 caches produtores
tinham shape correto, mas divergiram numericamente; portanto a classificação
correta de prefill e geração é `approximate`.

## Limites preservados

- A comparação aceita que a execução seja diagnóstica somente com
  `--allow-unverified-fidelity`; sem isso o executor falha fechado.
- A divergência começa na saída vision, não no embedding textual incorporado.
- Nenhuma agenda por layer/assignment foi adicionada. As classes nativas de
  vision e audio sem transcrição continuam `runtime-defined` e suas vistas
  escalares continuam fail-closed.
- Este loop implementou a fronteira e registra apenas evidência candidata; uma
  revisão posterior deve inspecionar e reproduzir o diff antes de aceitá-la.
- O marcador de checkpoint Gemma 4 continua proibido.

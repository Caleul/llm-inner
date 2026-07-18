# Gemma 4 E4B: image/video real sem o checkpoint — 2026-07-18

## Classificação

Esta é evidência candidata de execução literal e diferencial da torre vision
real compartilhada por imagem e vídeo. Não é evidência de fidelidade exata e
não autoriza `.agent-loop/checkpoints/gemma4-dense-lossless/`: as duas
invocações divergem pela primeira vez na atenção nativa da camada 0, e o áudio
real também permanece aproximado.

## Identidade

- modelo: `google/gemma-4-E4B`
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`
- fonte: pacote Safetensors BF16 denso local `gemma-4-E4B-dense`
- runtime autoritativo: `transformers-5.5.0/torch-2.12.1-Gemma4Vision-CPU-eager`
- artefato: `artifacts/gemma4-e4b-dense.literal.json`
- bytes do artefato: `21,328,054,627`
- SHA-256 do artefato:
  `b33178b4d138367753ca29fa6a5936f64f9cb95d9f85407e2495cf87939be0b2`

## Limite implementado

`executeGemma4LiteralVisionF32` resolve 659 referências densas exclusivamente
das constantes incorporadas no JSON. Ele executa as 394 atribuições da torre
de features (todas as 395 definições vision menos o scatter composite final)
para a instância de imagem ou para os frames de vídeo achatados em ordem
`video,frame`, sem catálogo nem fallback para Safetensors.

A política numérica é despachada por fonte, dtype e classe de operação em todo
o programa vision BF16:

- 114 lineares/clipped-linears bias-free usam a árvore ARM de 32 lanes;
- 113 RMSNorms usam `pytorch-cpu-f32-cascade-sum`;
- todos os resultados observáveis da torre são estreitados para BF16 antes do
  consumidor seguinte, exceto a entrada F32 e a máscara BOOL;
- as 16 atenções bidirecionais e o pooling espacial declaram
  `accumulationDtype=runtime-defined` porque são matmuls nativos batched cuja
  agenda não é a GEMV ARM já comprovada no ramo textual.

A tentativa de aplicar a agenda eager BF16 textual à classe batched vision
`9x64 · 64x9` não aumentou as fronteiras exatas em imagem ou vídeo. O artefato
final portanto preserva a fronteira fail-closed: a vista escalar de uma atenção
recusa a expansão até existir uma agenda geral comprovada, enquanto lineares
continuam substituindo bits BF16, números aprendidos, quatro bounds de clipping,
redução ARM completa e cast de saída.

## Capturas autoritativas

Imagem usa nove patches `[1,9,768]` nas posições espaciais 3x3, com valores
F32 de `0` a `1`. Vídeo usa dois frames `[1,2,9,768]`, as mesmas posições por
frame e valores de `-0.5` a `0.5`. Cada captura registra 245 fronteiras: patch
embedding, 15 fronteiras por camada nas 16 camadas, pooling/strip e projeção de
linguagem.

```bash
npm run capture:gemma4-vision-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop48-image-checkpoints-full.json \
  --invocation image --frames 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a

npm run capture:gemma4-vision-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop48-video-checkpoints-full.json \
  --invocation video --frames 2 --pixel-start=-0.5 --pixel-end 0.5 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

## Replay sem fonte

Durante cada comparação o diretório `gemma-4-E4B-dense` foi renomeado e a
asserção de ausência permaneceu ativa:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-vision-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop48-image-checkpoints-full.json \
  --report /tmp/gemma4-loop48-image-source-removed-final-report.json \
  --max-tensor-mib 64 --absolute-tolerance 0 --relative-tolerance 0 \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

Uma verificação integral separada confirmou os 2.130 compromissos do artefato,
15.992.314.836 bytes de storage incorporado e SHA-256 lógico
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`, também
com `sourceCheckpointAccessed=false`.

Resultados em tolerância zero:

| invocação | fronteiras | exatas | primeiro desvio | erro abs. no primeiro desvio | saída | erro abs. terminal | cosseno terminal | top-k | argmax |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| imagem | 245 | 8 | `vision_layer_0_o` | 0.125 | `image_features [1,2560]` | 0.03173828125 | 0.9999172406954236 | 1 | igual |
| vídeo (2 frames) | 245 | 8 | `vision_layer_0_o` | 0.125 | `image_features [2,2560]` | 0.029296875 | 0.9999457241229366 | 1 | igual |

As oito fronteiras bit-exatas em ambas as invocações são patch embedding,
input RMSNorm da camada 0 e Q/K/V projection + RMSNorm. A primeira divergência
é a saída da atenção já projetada por `o_proj`; isso delimita a classe batched
attention sem criar allowlist por camada.

## Navegação escalar sem fonte

O índice final contém 2.549 operações composite instanciadas. Uma janela de
oito termos de `composite_image_features/vision_layer_0_q` decodificou oito
pesos BF16 reais e os quatro bounds, declarou a redução completa `0..767`, a
árvore ARM e `outputDtype=BF16`. Já
`composite_image_features/vision_layer_0_attention` falhou como esperado:

```text
vision_layer_0_attention: vista escalar falha fechada porque a redução
pytorch-native-matmul ainda não possui agenda literal comprovada.
```

O checkpoint Gemma 4 continua proibido: vision e áudio ainda não possuem
fidelidade exata, e a composição multimodal completa de logits/geração não foi
demonstrada contra o runtime autoritativo.

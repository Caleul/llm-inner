# Gemma 4 E4B: execução de áudio real sem o checkpoint — 2026-07-18

> **Artefato substituído.** A execução source-removed foi repetida após a
> transcrição compartilhada de `torch.pow(x,-0.5)`, usando o artefato
> `daa029dd3e635b6c7757a1db6979ea7f81d2b8c8b49a193b6574a58ea5d13cb4`.
> A primeira divergência permanece `audio_subsample_0_relu`; a saída terminal
> mantém erro absoluto `0.0625` e agora tem cosseno `0.9999595279745614`.
> A evidência consolidada está em
> [`gemma4-e4b-explicit-eager-vision-composite-2026-07-18.md`](gemma4-e4b-explicit-eager-vision-composite-2026-07-18.md).

## Classificação

Esta é evidência candidata de execução literal e diferencial do tower de áudio
real. Não é evidência de fidelidade exata e não autoriza
`.agent-loop/checkpoints/gemma4-dense-lossless/`: a primeira divergência ocorre
no primeiro Conv2d e imagem, vídeo e composição multimodal completa ainda não
possuem comparação real equivalente.

## Identidade

- modelo: `google/gemma-4-E4B`
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`
- fonte: pacote Safetensors BF16 denso local `gemma-4-E4B-dense`
- runtime autoritativo: `transformers-5.5.0/torch-2.12.1-Gemma4Audio-CPU-eager`
- artefato: `artifacts/gemma4-e4b-dense.literal.json`
- bytes do artefato: `21,327,893,565`
- SHA-256 do artefato:
  `0ec0c382e09af1bc28805e59778c9ede69e9aa1133bdbd04edbb8b781f1d8483`

## Limite implementado

`executeGemma4LiteralAudioF32` resolve todas as referências densas a partir das
constantes incorporadas no JSON e não recebe catálogo ou diretório de
checkpoint. O materializador aceita tensores F32/F16/BF16 de rank arbitrário,
valida shape e tamanho exatos e impõe um orçamento por tensor antes de alocar.
No pacote real, ele decodifica 752 referências usadas pelas 523 atribuições
nomeadas do programa de áudio.

A política numérica é aplicada por classe de operação em todo o programa:

- 134 lineares BF16 sem bias: dot ARM BF16/F32 de 32 lanes;
- 109 RMSNorm BF16: `pytorch-cpu-f32-cascade-sum`;
- 223 operações BF16/F32 sem redução;
- 38 operações F32/F32 sem redução;
- 2 operações BOOL;
- 17 operações BF16 nativas com acumulação ainda `runtime-defined`: Conv2d,
  LayerNorm de canais, depthwise convolution e lineares com bias.

Todo resultado declarado BF16 é estreitado antes do consumidor seguinte. A
vista escalar substitui os números aprendidos e mostra bits, decoder, bounds,
árvore ARM, agenda de RMS e casts; ela falha fechada quando a acumulação
permanece `runtime-defined`.

## Captura autoritativa

Entrada determinística: `input_features` F32 `[1,1,128]`, linearmente espaçada
de `-0.25` a `0.25`, com máscara verdadeira. A captura registra 20 fronteiras:
dois subsamplers, projeção de entrada, posição relativa, 12 camadas, projeção e
norma finais, projeção de embedding e stripping final.

```bash
npm run capture:gemma4-audio-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop47-audio-checkpoints-bf16.json \
  --frames 1 --feature-start=-0.25 --feature-end=0.25 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

## Replay candidato sem fonte

Durante o comando abaixo, `gemma-4-E4B-dense` foi renomeado e portanto estava
indisponível. A asserção de ausência faz o comparador falhar se o caminho
existir.

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-audio-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop47-audio-checkpoints-bf16.json \
  --report /tmp/gemma4-loop47-audio-source-removed-final-report.json \
  --max-tensor-mib 16 --absolute-tolerance 0 --relative-tolerance 0 \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

Resultado real:

- `sourceCheckpointAccessed: false`;
- 20/20 operações presentes, sem mismatch de shape ou valor não finito;
- `fidelityClass: approximate`;
- primeira divergência: `audio_subsample_0_relu`;
- saída terminal: `audio_features [1,2560]`;
- erro absoluto máximo terminal: `0.0625`;
- erro relativo máximo terminal: `4.680672268907563`;
- similaridade cosseno terminal: `0.9999528084610771`;
- top-k overlap terminal: `1`;
- argmax terminal: acordo verdadeiro.

O erro inicial prova que a redução nativa de Conv2d/LayerNorm não pode ser
substituída por uma agenda ordenada genérica e chamada de exata. O contrato
permanece `runtime-defined` e a classificação permanece aproximada até que a
classe nativa completa tenha evidência reproduzível.

## Revisão independente recebida nesta sessão

A alegação candidata anterior de RoPE full-range foi inspecionada e repetida
com token `184`, posição absoluta `125` e o checkpoint ausente. As 1.229
atribuições textuais, logits e caches KV produtores passaram com erro absoluto
e relativo zero. Essa evidência textual foi aceita independentemente, mas não
altera a classificação aproximada do áudio nem autoriza o checkpoint composto.

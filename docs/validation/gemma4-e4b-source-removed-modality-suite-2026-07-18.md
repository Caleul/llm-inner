# Gemma 4 E4B: suite source-removed das três modalidades — 2026-07-18

## Fronteira candidata

O loop 85 revisou independentemente o commit
`e6fede07cb2a4fb80303cfc23b753200efde37d6`: `npm run typecheck` passou e os
234 testes existentes passaram. Isso aceita, dentro de seu escopo, a ordem dos
4.146 statements escalares do schema v28; não aceita o checkpoint Gemma 4.

A evidência autoritativa mais recente cobria somente áudio sob o contrato
fixado `torch.inference_mode` + eager. Os traces de imagem e vídeo eram
anteriores a essa fronteira de execução, embora o artefato já carregasse as
três rotas. A nova suite transforma essa lacuna numa única condição
fail-closed por classe de modalidade:

- requer `image`, `video` e `audio`, exatamente uma vez e nessa ordem;
- requer modelo, revisão, runtime de referência e runtime candidato iguais;
- liga `modelId`, revisão e checksums de `config.json`/Safetensors de cada trace
  aos compromissos de source incorporados no artefato;
- exige tolerância absoluta e relativa zero;
- exige cinco fronteiras de prefill `pass` por modalidade;
- exige prompt, token/posição, logits terminais e todos os caches de geração;
- rejeita qualquer resultado diferente de `lossless-within-dtype`; e
- exige que as 32 + 32 + 36 BMM ainda desconhecidas totalizem exatamente 100,
  sem promovê-las a uma agenda escalar inventada.

O comando `compare:gemma4-literal-composite-suite` exige tanto
`--assert-source-unavailable` quanto `--allow-unverified-fidelity`. O primeiro
prova a fronteira source-removed; o segundo impede que a igualdade tensorial
seja confundida com publicação da árvore Apple Accelerate ainda ausente.

## Capturas autoritativas

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors denso BF16/F32 sem
quantização.

Imagem `[1,9,768]` e vídeo `[1,1,9,768]` foram recapturados pelo helper atual:

```bash
npm run capture:gemma4-composite-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --modality image --modality-token-id 258880 --text-token-id 2 \
  --max-new-tokens 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a

npm run capture:gemma4-composite-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --modality video --modality-token-id 258884 --text-token-id 2 \
  --frames 1 --max-new-tokens 1 --pixel-start 0 --pixel-end 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Os traces têm SHA-256:

- imagem: `48a7270be75d44ae31809ae04c77dd419b22937bd3ec5fdc1e4e705f031af68b`;
- vídeo: `10618977609781f65a7c325c7cbff3018cfb9b2ad280cd1a9bdde898bdaa7e9d`;
- áudio registrado: `d07227accbccdd911f2e5d66b0152e12f8d908ad480503e45b55d8344fac0610`.

Cada trace declara
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`,
`torch.inference_mode`, atenção `eager` e execução CPU.

## Suite com a fonte ausente

`./gemma-4-E4B-dense` foi movido fisicamente e restaurado por trap. Enquanto o
caminho estava ausente, foi executado:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-modality-suite-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --image-trace /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --video-trace /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --audio-trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop85-composite-modality-suite.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --top-k 10 --allow-unverified-fidelity
```

Resultado:

| modalidade | prefill | geração | token/posição | caches | BMM runtime-defined |
| --- | --- | --- | --- | --- | ---: |
| imagem | 5/5, erro zero | `lossless-within-dtype` | `184` / `2` | 24/24 | 32 |
| vídeo | 5/5, erro zero | `lossless-within-dtype` | `184` / `2` | 24/24 | 32 |
| áudio | 5/5, erro zero | `lossless-within-dtype` | `184` / `2` | 24/24 | 36 |

O relatório tem 169.857 bytes e SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.
Não houve primeira divergência. A fonte foi restaurada após o comando.

## Limite preservado

A suite fecha a cobertura ampla e coerente das rotas imagem/vídeo/áudio sob o
mesmo runtime autoritativo. Ela não resolve a ordem escalar interna das 100 BMM
Apple Accelerate: o relatório as conta e exige, mas mantém
`candidateFidelityAcknowledged=true` e
`fidelityClaim=candidate-modality-coverage-with-runtime-defined-reductions`.
Por isso `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

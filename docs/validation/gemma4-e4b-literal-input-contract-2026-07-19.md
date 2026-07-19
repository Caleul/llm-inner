# Gemma 4 E4B: contrato literal executável de inputs — 2026-07-19

## Fronteira candidata

O loop 88 revisou independentemente o schema v30 do loop 87 e repetiu
`npm run typecheck` e os 235 testes antes de alterar o formato. A base passou
integralmente. O checkpoint Gemma 4 continuou não aceito porque 100 BMM nativas
permanecem sem agenda escalar autoritativa.

O schema v30 já escolhia branches pelo `forwardControl`, mas os shapes,
domínios, cache ownership e cardinalidades que tornavam a presença de um input
válida ainda eram inferidos pelos executores/torres. O schema v31 incorpora
`inputContract`, um único programa para a classe completa de inputs Gemma 4:

- 13 declarações ligadas ao catálogo humano, com representação, rank e eixos;
- `input_ids` retangular, domínio de vocabulário/modal tokens e posições I32;
- `mm_token_type_ids` no mesmo shape e conflitos com máscara/cache;
- máscara aditiva `[batch,1|8,query,past+query]`, finita ou `-Infinity`;
- exatamente 24 caches produtores BHSD, com heads/head-dim por layer e sem
  propriedade duplicada dos 18 consumidores shared-KV;
- imagem/vídeo com 768 features por patch, posições `<10240`, padding somente
  `[-1,-1]`, pooling 3x3 e cardinalidade de células válidas igual aos
  placeholders 258880/258884;
- vídeo flatten em ordem video/frame row-major; e
- áudio `[batch,frames,features]`, máscara BOOL, dois strides 2, 32 canais,
  projeção de largura 1024 e cardinalidade pós-subsampling igual aos
  placeholders 258881.

O validador reconstrói o contrato somente do programa Gemma 4 registrado e
rejeita qualquer alteração. Os executores literal em memória e paginado o
executam antes do roteamento e de qualquer torre. O leitor e
`--end-to-end-calculation` expõem o mesmo objeto; a linguagem de fórmulas
schema 17 liga sua autoridade a `/inputContract`.

## Artefato e storage reais

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors denso BF16/F32.

```bash
npm run build
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Resultado:

- artifact schema `31`, formula schema `17`, input contract schema `1`;
- bytes: `21.387.310.472`;
- SHA-256:
  `e4dda53013c20cfe6d0afeff53e51b1859f830e7039c5cc673c5d74b7e16b85d`;
- constantes/decoders: `2.130/2.130`;
- payload incorporado: `15.992.314.836` bytes e `21.323.089.296`
  caracteres Base64.

O audit source-present:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop88-v31-source-audit.json \
  --verify-gemma4-payloads
```

comparou todos os `15.992.314.836` bytes. Source e literal produziram o mesmo
SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O relatório tem SHA-256
`746b35bc546176f84e97afbf390c48b5d355ce2ddc6a3e256e8ef692f048be0b`.

## Navegação e replay com a fonte ausente

`./gemma-4-E4B-dense` foi movido sob trap de restauração. Com o caminho
ausente:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop88-v31-source-removed-end-to-end.json
```

produziu `65.398.012` bytes, SHA-256
`2165d81ba3154b36b0700934dd77e3396d63f7e4259a982a923083ec55d79050`.
O relatório registra `sourceCheckpointAccessed=false`, o contrato completo,
2.709 operações/4.146 statements forward, 20 operações greedy e cobertura
`2.076 reachable + 54 runtime-unreachable = 2.130` constantes.

Ainda sem a fonte:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-modality-suite-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --image-trace /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --video-trace /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --audio-trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop88-v31-composite-modality-suite.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --top-k 10 --allow-unverified-fidelity
```

passou imagem, vídeo e áudio: 3/3 prefills e 3/3 gerações em tolerância zero,
token 184 na posição 2, logits terminais e 24 caches produtores por modalidade.
O relatório de `169.857` bytes é byte-idêntico à evidência v30, SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`;
isso mostra que o novo contrato remove comportamento host sem mudar a função
aceita para os três inputs reais.

## Validação e limite

```text
npm run typecheck  # exit 0
npm test           # 235/235, exit 0
```

Os testes cobrem execução/tamper do contrato, ranks e buffers, largura de
áudio, pooling/cardinalidade multimodal, catálogo alinhado, leitura streaming,
replay e geração.

Esta é evidência candidata do schema v31. As 100 BMM Apple Accelerate (32
imagem, 32 vídeo, 36 áudio) continuam
`fail-closed-runtime-reduction`; portanto o marcador
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

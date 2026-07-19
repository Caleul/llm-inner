# Gemma 4 E4B: controle estruturado do forward — 2026-07-19

## Fronteira candidata

O loop 87 inspecionou o diff do schema v29, repetiu os 235 testes e executou
novamente a suite real de imagem, vídeo e áudio. O relatório permaneceu
byte-idêntico, com SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.
Isso aceita o controle greedy estruturado do loop 86 dentro de seu escopo; não
aceita o checkpoint Gemma 4.

O grafo forward já serializava as três torres, mas os executores ainda
escolhiam em `if` do host quais inputs opcionais ativavam imagem, vídeo, áudio
e máscaras visuais. Quando um ramo estava ausente, o alias identidade entre
embeddings consecutivos também era comportamento do executor, não dado do
artefato.

O schema v30 incorpora `forwardControl`, um programa estruturado que declara:

- `input_ids` como input obrigatório;
- ordem modal única `image -> video -> audio`;
- os pares de input `all-present-or-all-absent` de cada modalidade;
- operações top-level e `activeInvocationId` de cada ramo presente;
- aliases `identity-alias-no-cast` exatos de cada ramo ausente;
- seleção entre máscaras visual full/sliding serializadas, máscara aditiva do
  chamador e causal/cache textual;
- incompatibilidade de `mm_token_type_ids` com `attention_mask` ou
  `past_key_values`;
- `position_ids[batch,sequence]=I32(sequence)` quando posições não são
  fornecidas; e
- cache vazio versus cache pós-RoPE fornecido.

O executor síncrono do objeto em memória e o executor paginado do artefato
aberto interpretam o mesmo programa antes de entrar nas torres. O validador
reconstrói esse programa do composite registrado e rejeita qualquer alteração
de grupo, ordem, operação, invocação, alias, máscara, posição ou cache. A
linguagem de fórmulas avançou para schema 16 e liga a autoridade a
`/forwardControl`.

## Artefato real e storage

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors denso BF16/F32 sem
quantização.

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

- schema do artefato `30`, linguagem de fórmulas `16` e controle forward `1`;
- bytes: `21.387.304.872`;
- SHA-256:
  `a338c7900e67fbb56b285c7f82ca4773cf0cfc6b48b4b75d0b6ba67866542838`;
- constantes/decoders: `2.130/2.130`;
- payload incorporado: `15.992.314.836` bytes e `21.323.089.296`
  caracteres Base64.

O audit source-present:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop87-v30-source-audit.json \
  --verify-gemma4-payloads
```

comparou todos os `15.992.314.836` bytes. Source e literal produziram o mesmo
SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
o relatório tem SHA-256
`74e2365220795d930eebbfeafe839f8b301dd317cb6088f2046b2082d79e3a9e`.

## Navegação e replay com a fonte ausente

`./gemma-4-E4B-dense` foi movido fisicamente sob trap de restauração. Enquanto
o caminho estava ausente, o leitor executou:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop87-v30-source-removed-end-to-end.json
```

O relatório de `65.378.484` bytes tem SHA-256
`cf9f1e56a857f48ca8975d9b9aeb5384624adcf5017bf8abe6e95adc61830860`
e registrou `sourceCheckpointAccessed=false`, o `forwardControl` completo,
2.709 atribuições forward, 4.146 statements escalares, 20 operações de geração
e cobertura de storage `2.076 reachable + 54 runtime-unreachable = 2.130`.
Continuaram explícitos 1.462 reduções, 256 estágios, 1.722 domínios e exatamente
100 reduções `runtime-defined`.

Ainda sem a fonte, a suite executou:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-modality-suite-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --image-trace /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --video-trace /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --audio-trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop87-v30-composite-modality-suite.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --top-k 10 --allow-unverified-fidelity
```

Imagem, vídeo e áudio passaram as 15/15 fronteiras de prefill com erro
absoluto/relativo zero. Cada rota selecionou token `184` na posição `2`,
reproduziu logits terminais e os 24 caches produtores sem primeira divergência.
A suite preservou 32 + 32 + 36 = 100 BMM `runtime-defined` e manteve o mesmo
SHA-256 do relatório aceito no loop anterior:
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

## Validação e limite preservado

```text
npm run typecheck  # exit 0
npm test           # 235/235, exit 0
```

Os testes cobrem seleção modal completa, aliases de ausência, conflitos de
máscara/cache, pares incompletos, adulteração do programa, replay síncrono e
leitura streaming.

Esta é evidência candidata do schema v30. As 100 BMM nativas de visão/áudio
continuam `fail-closed-runtime-reduction` porque a agenda escalar interna de
Apple Accelerate SGEMM não é publicada nem foi derivada por um contrato geral.
Por isso `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

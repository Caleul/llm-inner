# Gemma 4 E4B: controle estruturado de geração — 2026-07-18

## Fronteira candidata

O loop 86 revisou independentemente a evidência aceita no handoff 85. Na
revisão, `npm run typecheck` passou, os 235 testes passaram e a suite
source-removed de imagem, vídeo e áudio reproduziu o mesmo relatório com
SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.
Isso aceita a fronteira multimodal do schema v28 dentro do seu escopo, mas não
o checkpoint Gemma 4.

O artefato anterior descrevia as doze transições greedy em assignments e
fórmulas escalares, porém o executor ainda escolhia e ordenava essas transições
num loop do host. O schema v29 incorpora agora
`generation.controlProgram`, um programa estruturado, validado integralmente e
executado por todos os caminhos literais. Ele fixa:

- bindings e restrições do prefill;
- origem da posição inicial;
- domínio completo do vocabulário, ordem ascendente, comparação estrita,
  desempate pelo menor token e rejeição de logits não finitos;
- append de token e incremento inteiro exato da posição;
- bindings do forward incremental e a lista exata de inputs omitidos;
- snapshot do cache depois do forward incremental;
- avaliação de EOS somente depois desse forward e snapshot; e
- seleção dos logits e caches terminais pelo número de passos concluídos.

O executor síncrono do artefato e o interpretador assíncrono paged/composite
consomem o mesmo objeto. O validador reconstrói o contrato inteiro a partir do
programa forward incorporado e rejeita qualquer divergência; não existe
fallback para `generate`, Transformers ou outro loop genérico.

## Artefato real regenerado

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

O artefato schema v29 tem 21.387.302.065 bytes e SHA-256
`0169f2de1aee13d8bc7f8d44063d5a823fa6b2fada4ab86b422f8160d54a2c07`.
Ele incorpora 2.130 constantes, 15.992.314.836 bytes de payload e
21.323.089.296 caracteres Base64. O caminho de origem não aparece no JSON.

O audit de todos os bytes foi executado com:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop86-v29-source-audit.json \
  --verify-gemma4-payloads
```

Os 15.992.314.836 bytes comparados produziram o mesmo hash de storage na
fonte e no literal:
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O audit ligou ainda seis arquivos de identidade, totalizando
16.024.773.810 bytes, ao modelo e revisão declarados.

## Replay real com a fonte ausente

`./gemma-4-E4B-dense` foi movido fisicamente e restaurado por trap. Enquanto o
caminho estava ausente, o interpretador v29 executou:

```bash
node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-modality-suite-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --image-trace /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --video-trace /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --audio-trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop86-v29-composite-modality-suite.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --top-k 10 --allow-unverified-fidelity
```

| modalidade | prefill | geração | token/posição | caches | BMM runtime-defined |
| --- | --- | --- | --- | --- | ---: |
| imagem | 5/5, erro zero | `lossless-within-dtype` | `184` / `2` | 24/24 | 32 |
| vídeo | 5/5, erro zero | `lossless-within-dtype` | `184` / `2` | 24/24 | 32 |
| áudio | 5/5, erro zero | `lossless-within-dtype` | `184` / `2` | 24/24 | 36 |

As três rotas passaram prefill e geração com tolerâncias absoluta e
relativa zero, sem primeira divergência. O hash do relatório permaneceu
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`,
demonstrando que a nova representação estruturada preserva os resultados
aceitos ao retirar a decisão de ordem do loop de host.

## Validação de contrato

Os testes cobrem o programa incorporado, o desempate exato do argmax, a
rejeição de não finitos, o replay síncrono equivalente, o momento do EOS, a
rejeição de controle adulterado e a exposição pelo leitor streaming. Os gates
configurados passaram:

```text
npm run typecheck  # exit 0
npm test           # 235/235, exit 0
```

## Limite preservado

Esta é uma reivindicação candidata para revisão independente. A mudança
fecha o controle de geração implícito, mas não publica a ordem de acumulação
escalar interna das 100 BMM Apple Accelerate. A suite continua exigindo e
contando essas reduções como `runtime-defined`, com
`candidateFidelityAcknowledged=true`. Por isso
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

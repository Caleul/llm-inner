# Gemma 4 E4B: identidade imutável e metadata incorporada — 2026-07-18

## Resultado candidato

O programa literal schema v9 agora carrega a autoridade exata da qual seu
cálculo foi derivado. `sourceIdentity` contém o model ID, revisão imutável,
adaptador semântico, tamanho e SHA-256 dos seis arquivos do pacote. Os cinco
JSONs top-level são incorporados como bytes Base64 com decoder e digest; o
Safetensors permanece incorporado como 2.130 constantes nomeadas e também
recebe um compromisso do arquivo-container inteiro.

O gerador exige `--model-id` e `--revision`. O auditor source-present recalcula
todos os compromissos e rejeita qualquer diferença antes da remoção da fonte.
O leitor source-removed valida novamente os bytes de metadata incorporados e
os compromissos de cada payload aprendido. Esta é uma alegação candidata de
proveniência autocontida; não altera a classificação numérica dos batched
matmuls nativos e não autoriza o marcador de checkpoint.

## Identidade incorporada

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- adaptador: `gemma4-composite-v1`;
- `model.safetensors`: `15.992.595.884` bytes, SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- `config.json`: `5.105` bytes, SHA-256
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`;
- `generation_config.json`: `181` bytes, SHA-256
  `de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c`;
- `processor_config.json`: `1.689` bytes, SHA-256
  `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c`;
- `tokenizer_config.json`: `881` bytes, SHA-256
  `6a9383197f000d2723684efd2210f5bf217bc29fa2cf360ae1574201b47af060`;
- `tokenizer.json`: `32.170.070` bytes, SHA-256
  `12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`.

Os seis arquivos totalizam `16.024.773.810` bytes comprometidos. O programa
não serializa nenhum caminho local do checkpoint.

## Regeneração real

```bash
npm run build
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

- artefato: `21.382.303.443` bytes;
- SHA-256:
  `595603bedc33e27e3e7d270df45985c00eb7873196ac932aaf717fa0be0c6eee`;
- constantes aprendidas: `2.130`;
- payload aprendido: `15.992.314.836` bytes.

## Auditoria com a fonte presente

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop65-source-audit.json \
  --verify-gemma4-payloads
```

Resultado:

- os seis arquivos e `16.024.773.810` bytes de source identity foram
  recalculados e coincidiram;
- os 2.130 payloads e todos os `15.992.314.836` bytes foram comparados por
  nome, dtype, shape e range;
- digest agregado da fonte e do literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho fonte encontrado no JSON: `false`.

O auditor foi corrigido para delimitar estruturalmente o array `constants`.
Assim, os cinco `payloadBase64` de metadata não podem ser contados como pesos.

## Verificação source-removed e navegação end-to-end

O diretório `gemma-4-E4B-dense` foi movido para fora do caminho, restaurado por
`trap`, e o leitor executou no mesmo comando `--verify-payloads` e
`--end-to-end-calculation --generation-max-new-tokens 2`. O relatório declarou:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --verify-payloads \
  --end-to-end-calculation \
  --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop65-final-source-removed.json
```

- `schemaVersion: 9`, model ID e revisão acima;
- `sourceCheckpointAccessed: false`;
- 2.130 compromissos de payload e `15.992.314.836` bytes verificados;
- digest incorporado
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- 2.709 atribuições forward, das quais 2.609 literais e 100 fail-closed;
- 20 atribuições greedy, 42 transições KV;
- 2.076 constantes aprendidas alcançáveis e 54 runtime-unreachable, cobrindo
  exatamente as 2.130 constantes incorporadas.

O relatório transitório possuía `56.814.405` bytes e SHA-256
`f4ecb6ce037bbc57e12e5dfdb6dbcb9f3c026fb3409d7920b6bed3cb0ad678ba`.

## Revisão independente e limites

Antes da mudança, o commit
`039f94f3ab5720459ac96bb87838c11b9973c0bb` foi revisado com typecheck, 227
testes e uma nova inspeção source-removed. As 2.709 atribuições forward, 20 de
controle, 42 transições KV e a partição `2.076 + 54 = 2.130` foram reproduzidas;
portanto a alegação candidata de composição/navegação do loop 64 é aceita sem
elevar sua fidelidade.

Permanecem 100 operações em cinco classes vision/audio de batched matmul cuja
árvore interna do Apple Accelerate `sgemm` não é declarada. Áudio continua
numericamente aproximado nos diferenciais preservados. O diretório
`.agent-loop/checkpoints/gemma4-dense-lossless/` permanece ausente.

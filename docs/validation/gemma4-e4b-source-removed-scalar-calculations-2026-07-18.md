# Gemma 4 E4B: cálculos escalares serializados — 2026-07-18

## Classificação

Esta sessão produz uma alegação candidata de formato/navegação, não aceita o
checkpoint que ela própria modificou. O schema v5 incorpora fórmulas indexadas,
inputs ordenados, coordenadas, casts, papéis aprendidos e reduções para todas as
definições do forward. As cinco classes nativas sem árvore de redução
autoritativa continuam fail-closed e o áudio continua aproximado; portanto
`.agent-loop/checkpoints/gemma4-dense-lossless/` permanece ausente.

## Identidade e integridade

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- source: Safetensors BF16 denso, sem quantização;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,332,185,460`;
- SHA-256: `b5ef211948435030f3f90e5e13b0157d54b3fe6dc648fa5587f62b9152d36367`;
- constantes: `2.130`;
- payload incorporado/comparado: `15.992.314.836` bytes;
- SHA-256 agregado do storage source e literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho do checkpoint presente no JSON: `false`.

O audit integral foi executado com:

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop58-scalar-calculations-audit.json \
  --verify-gemma4-payloads
```

## Cobertura source-removed

Com `gemma-4-E4B-dense` fisicamente renomeado, o leitor verificou novamente
todos os payloads e listou `2.709` operações instanciadas a partir de `2.283`
definições em `70` pares scope/operação. Toda operação navegável contém a
entrada `scalarCalculation` lida do próprio JSON.

- `2.609` instâncias possuem `reproducibility=literal`;
- `100` instâncias, correspondentes a `68` definições, preservam
  `fail-closed-runtime-reduction`;
- as cinco classes são vision score/value BF16 e audio content-score,
  position-score e value F32;
- o relatório source-removed tem SHA-256
  `6235eb378ea8415fc9386d8eee5d1a2f7a22fe2567c4b87478f31dad9e912efb`.

Uma vista de `composite_image_features/vision_layer_0_q[0,0,0]` carregou do
contrato serializado os papéis `weight`, `input-min`, `input-max`, `output-min`
e `output-max`. A fórmula mantém `input_feature=0..in_features-1`; uma janela
de quatro termos decodificou os pesos BF16 `0xbc40`, `0xbbc0`, `0xbbc0` e
`0xbc61` como `-0.01171875`, `-0.005859375`, `-0.005859375` e
`-0.01373291015625`. Os quatro bounds foram substituídos como `-6.375`,
`6.3125`, `-11.3125` e `11.1875`. A agenda completa permaneceu ARM NEON BF16
dot de 32 lanes com fold pairwise e `764` termos omitidos somente da janela.

O comando source-removed foi:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-operations --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --operation composite_image_features/vision_layer_0_q \
  --output-coordinate 0,0,0 --input-start 0 --input-count 4 \
  --output /private/tmp/gemma4-loop58-source-removed-scalar-calculations.json
```

## Diferencial autoritativo source-removed

Os traces imutáveis da runtime
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager`
foram repetidos a tolerância zero contra o schema v5:

| modalidade | prefill | geração | logits terminais | token | SHA-256 do relatório |
|---|---|---|---|---:|---|
| imagem | `lossless-within-dtype` | `lossless-within-dtype` | erro abs./rel. zero | 184 | `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e` |
| vídeo, 1 frame | `lossless-within-dtype` | `lossless-within-dtype` | erro abs./rel. zero | 184 | `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d` |
| áudio, 1 frame | `approximate` | `approximate` | abs. `0.3125`, rel. `2073.9473684210525` | 184 | `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4` |

O áudio diverge primeiro em `composite_audio_features`; seus caches também
divergem. Igualdade do token terminal não altera a classificação.

## Validação fail-closed

Os testes exigem uma entrada escalar para cada domínio de atribuição e rejeitam
uma fórmula opaca, uma agenda removida, uma ordem alterada, um papel aprendido
alterado ou um decoder divergente. O leitor streaming exige
`scalarCalculations` na cauda estrutural e revalida o contrato inteiro antes de
expor qualquer navegação.

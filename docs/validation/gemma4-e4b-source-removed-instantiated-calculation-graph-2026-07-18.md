# Gemma 4 E4B: grafo de cálculo instanciado source-removed — 2026-07-18

## Classificação

Esta sessão produz uma alegação candidata de formato e navegação. Ela aceita a
evidência candidata de geração do commit
`54ce0f550d24d458ab1fe803215c6b49a76c8d34` após rerodar `npm run typecheck`
e os `227/227` testes sem alteração. Ela não pode aceitar o schema v7 que ela
própria implementou nem criar o checkpoint Gemma 4: áudio e cinco classes BMM
nativas continuam incompletos.

## Resultado

`calculationGraph.assignments` substitui a expansão reconstruída pelo leitor.
Cada operação forward instanciada contém:

- ordinal, `operationId`, `definitionId`, scope e `invocationId`;
- entradas ordenadas e saída já vinculadas ao call site;
- shape, eixos, layout, dtype e política numérica da saída;
- fórmula escalar vinculada e operandos aprendidos com decoder;
- predecessores produtores e consumidores explícitos.

O gerador possui um único construtor desse grafo. A ordem forward de geração é
derivada dele, e o leitor source-removed navega as entradas serializadas em vez
de reinstanciar vision/audio/text em TypeScript. As fórmulas de wrapper agora
referem `calculationGraph` e nenhum cálculo instanciado contém
`declared_subprogram`, `inline_dependency_ordered_assignments` ou
`generic_decoder`.

O novo vínculo também expôs e corrigiu o nome textual real dos masks:
`attention_mask:full_attention` e `attention_mask:sliding_attention` são
vinculados, respectivamente, a `full_attention_mask` e
`sliding_attention_mask`.

## Artefato real e integridade

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- source: Safetensors BF16 denso, sem quantização;
- schema: `7`;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21.339.103.316`;
- SHA-256: `f0facbe87cc091d97e9aebbc9a34820f4fbde48881f1049d7792f1813c1b4fd1`;
- constantes: `2.130`;
- payload incorporado: `15.992.314.836` bytes;
- SHA-256 do storage source e literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho absoluto do checkpoint presente no JSON: `false`.

O audit source-backed tem SHA-256
`e9f347399b19d44d120da7f86d3aea082a56310bc03b498e3eaad15604010d44`.

```bash
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop60-calculation-graph-audit.json \
  --verify-gemma4-payloads
```

## Navegação source-removed

Com `gemma-4-E4B-dense` fisicamente movido para fora do caminho declarado, o
leitor verificou todo o payload e abriu:

- `2.709` atribuições, de `composite_placeholder_masks` a
  `final_logit_softcap`;
- `3.363` arestas explícitas de produtor;
- quatro invocações: imagem, vídeo, áudio e text core;
- `0` operações com os três marcadores opacos acima;
- `68` definições / `100` instâncias fail-closed nas cinco classes BMM já
  conhecidas.

O relatório completo de grafo tem `10.055.071` bytes e SHA-256
`bf57d6df792eaef13896088cedc7b5d86f6c51f336257d676c47c801031b52c1`.
Exemplos verificados incluem:

- `composite_image_features/vision_layer_0_q` lendo
  `composite_image_features/vision_layer_0_attn_norm`, produzido por
  `composite_image_features/vision_layer_0_input_norm`;
- `composite_video_features/vision_patch_projection` lendo a saída vinculada
  da normalização de pixels da invocação de vídeo;
- `composite_audio_features/audio_subsample_0_conv` lendo a saída vinculada do
  unsqueeze de áudio;
- `layer_0_attention` lendo `sliding_attention_mask`, produzido por
  `composite_sliding_attention_mask`;
- `final_logit_softcap` lendo `logits`, produzido por `lm_head`.

A vista escalar e a navegação de dois passos de geração têm SHA-256
`66b0cbc5a65c70ad9d6eabbc9ccbf840a32dc224d0b2ce7e76d0f83e780cdf38`.
Os quatro pesos BF16 `0xbc40`, `0xbbc0`, `0xbbc0` e `0xbc61` continuaram
decodificando como `-0.01171875`, `-0.005859375`, `-0.005859375` e
`-0.01373291015625`; os quatro bounds foram `-6.375`, `6.3125`, `-11.3125`
e `11.1875`. A geração expandiu as mesmas `2.709` entradas e as `42`
transições KV serializadas.

## Diferencial autoritativo source-removed

Os traces imutáveis da runtime
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager`
foram comparados a tolerância zero enquanto o source estava indisponível:

| modalidade | prefill | geração | logits terminais | SHA-256 |
|---|---|---|---|---|
| imagem | `lossless-within-dtype` | `lossless-within-dtype` | abs./rel. `0`; cosseno/top-k/argmax `1` | `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e` |
| vídeo, 1 frame | `lossless-within-dtype` | `lossless-within-dtype` | abs./rel. `0`; cosseno/top-k/argmax `1` | `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d` |
| áudio, 1 frame | `approximate` | `approximate` | abs. `0.3125`; rel. `2073.9473684210525`; cosseno `0.9998730821941509`; top-k/argmax `1` | `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4` |

Todos geraram o token `184` e declaram `sourceCheckpointAccessed=false`. Os
três hashes são idênticos aos relatórios do schema v6, provando que a mudança
de ownership/navegação não alterou os valores executados.

## Limites preservados

- Vision score/value: `32` definições e `64` instâncias (imagem + vídeo) ainda
  dependem da árvore nativa não publicada.
- Audio content/position/value: `36` definições e `36` instâncias continuam
  fail-closed.
- Áudio diverge a partir de `composite_audio_features`, inclusive logits e
  caches, apesar do argmax terminal igual.
- Esta sessão não cria nem autoriza
  `.agent-loop/checkpoints/gemma4-dense-lossless/`.

# Gemma 4 E4B: cálculos de geração serializados — 2026-07-18

## Classificação

Esta sessão produz uma alegação candidata de formato/navegação e não aceita o
checkpoint que ela própria modificou. O schema v6 move as fórmulas de controle
greedy, a ordem completa do forward e as transições KV para dentro do JSON.
Imagem e vídeo continuam lossless-within-dtype; áudio e cinco classes de BMM
continuam sem fidelidade literal completa. Portanto
`.agent-loop/checkpoints/gemma4-dense-lossless/` permanece ausente.

## Identidade e integridade

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- source: Safetensors BF16 denso, sem quantização;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21.332.629.091`;
- SHA-256: `e3dc6695e6bcdfe1949506c015c22925c0f9877d846a75cbd43f361a555acd44`;
- constantes: `2.130`;
- payload incorporado/comparado: `15.992.314.836` bytes;
- SHA-256 agregado do storage source e literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho do checkpoint presente no JSON: `false`.

O audit source-backed foi:

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop59-generation-calculations-audit.json \
  --verify-gemma4-payloads
```

## Contrato source-removed

Com `gemma-4-E4B-dense` fisicamente renomeado, o leitor verificou os `2.130`
payloads e todos os `15.992.314.836` bytes sem abrir a fonte. O schema v6
expôs:

- `2.709` operações forward instanciadas, de
  `composite_placeholder_masks` a `final_logit_softcap`;
- `12` definições escalares de controle greedy armazenadas no JSON;
- `42` transições KV textuais, sendo `24` produtoras e `18` reutilizações;
- layer 0 como produtora, com concatenação explícita da chave e do valor BHSD
  no eixo de sequência `2` durante decode;
- layer 24 como consumidora do cache já atualizado da layer produtora 22, sem
  criar uma segunda entrada de cache.

A navegação de dois passos materializou `20` operações de controle. A vista de
`generation_incremental_forward[1]` veio do contrato serializado e declarou:

```text
evaluate generation.forwardCalculation.operationOrder[0..2708] ...
forward_state[2].logits = softcapped_logits
forward_state[2].past_key_values = apply generation.forwardCalculation.cacheTransitions[*].incremental in layer order
forward_state[2] = STRUCT(logits=softcapped_logits,past_key_values=declared_incremental_cache_outputs)
```

Ela não contém `declared_cached_incremental_forward` nem `generic_decoder`. O
relatório com verificação integral tem SHA-256
`19003ca1978f60368a9c059136a641a9063cc2f9633a8de9d81f827a359b2e2a`; o
relatório que inclui o programa completo tem SHA-256
`65eda0606f987a60492d27907fff4e91814b05cecc24eccf0ad026e0bf7603a8`.

A substituição aprendida source-removed também foi repetida para
`composite_image_features/vision_layer_0_q[0,0,0]`. Os quatro pesos BF16
`0xbc40`, `0xbbc0`, `0xbbc0` e `0xbc61` decodificaram para
`-0.01171875`, `-0.005859375`, `-0.005859375` e `-0.01373291015625`; os
limites foram `-6.375`, `6.3125`, `-11.3125` e `11.1875`. O relatório tem
SHA-256 `bed54d8be08291dd5b56fdda4744d24ece06f003960ef230a9a74e08735e9040`.

## Diferencial autoritativo source-removed

Os traces imutáveis da runtime
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager`
foram repetidos a tolerância zero contra o schema v6:

| modalidade | prefill | geração | logits terminais | SHA-256 do relatório |
|---|---|---|---|---|
| imagem | `lossless-within-dtype` | `lossless-within-dtype` | abs./rel. `0`, cosseno/top-k/argmax `1` | `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e` |
| vídeo, 1 frame | `lossless-within-dtype` | `lossless-within-dtype` | abs./rel. `0`, cosseno/top-k/argmax `1` | `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d` |
| áudio, 1 frame | `approximate` | `approximate` | abs. `0.3125`, rel. `2073.9473684210525`, cosseno `0.9998730821941509`, top-k/argmax `1` | `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4` |

Os três relatórios declaram `sourceCheckpointAccessed=false`.

## Validação fail-closed

Os testes exigem uma definição escalar para cada atribuição de geração,
rejeitam uma fórmula prefill substituída por `generic_decoder`, rejeitam a
remoção de uma transição KV e comparam a ordem forward inteira serializada com
a navegação instanciada antes de renderizar qualquer cálculo. `step` e
`max_new_tokens` são as únicas substituições feitas pelo leitor; as fórmulas
de argmax, posição, append, EOS, seleção terminal e estado forward vêm do JSON.

## Limites preservados

- As cinco classes nativas de batched matmul ainda mantêm `68` definições e
  `100` instâncias fail-closed porque a árvore Accelerate não é autoritativa.
- Áudio continua aproximado desde `composite_audio_features`, inclusive logits
  e caches, apesar do argmax terminal igual.
- Esta sessão não cria nem autoriza o marcador de checkpoint Gemma 4.

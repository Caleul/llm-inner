# Gemma 4 E4B: literais numéricos source-removed — 2026-07-18

## Classificação

Esta sessão produz uma alegação candidata de formato e navegação, não aceita o
checkpoint Gemma 4. O schema v8 fecha a dependência do parser decimal para
constantes escalares: todas as fórmulas forward e greedy apontam para uma tabela
validada de tokens e bits F64/F32/BF16. As cinco classes BMM nativas e o ramo de
áudio continuam incompletos, portanto nenhum marker de checkpoint foi criado.

## Contrato

`numericLiterals.literals` é construída de todas as fórmulas serializadas e
declara, por token:

- identidade estável e usos por scope/definition;
- token decimal ou constante matemática nomeada;
- bits IEEE-754 binary64, binary32 e bfloat16;
- arredondamento BF16 `round-to-nearest-ties-to-even`.

O cast circundante da fórmula escolhe a representação. O leitor reconstrói a
tabela do programa e rejeita bit, uso ou token divergente. A CLI permite lookup
direto com `--numeric-literal`; no E4B real foram encontrados 70 tokens. Por
exemplo, `0.5` declara F64 `0x3fe0000000000000`, F32 `0x3f000000` e BF16
`0x3f00`, além de todos os consumidores vision/audio/text.

## Artefato real

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- formato: Safetensors BF16 denso, sem quantização;
- schema: `8`;
- bytes: `21.339.397.767`;
- SHA-256: `d25f9e2c7cbdd05d50de65c93911537057498b139ae0d7a5fe7a85480244b9b3`;
- constantes incorporadas: `2.130`;
- payload incorporado: `15.992.314.836` bytes;
- SHA-256 do storage source e literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho do checkpoint serializado: `false`.

O audit source-backed foi gravado em
`/private/tmp/gemma4-loop62-numeric-literals-audit.json` com SHA-256
`96eee422e82e59a57303768ead869b9adb687fb63c42780fcc90af42b0fc5921`.

## Evidência source-removed

Com `gemma-4-E4B-dense` fisicamente fora do caminho declarado, uma única
inspeção verificou todos os 2.130 payloads, os 15.992.314.836 bytes e o digest
agregado acima, abriu o token `0.5` e substituiu quatro pesos e bounds de
`composite_image_features/vision_layer_0_q`. O relatório tem SHA-256
`ae3b6d6e8150a0b5c24a11ed873ca43f1f6c1ebee60ecf19826e34007eec7969`
e declara `sourceCheckpointAccessed=false`.

Os diferenciais autoritativos source-removed preservaram exatamente os hashes
anteriores:

| modalidade | prefill | geração | logits terminais | SHA-256 |
|---|---|---|---|---|
| imagem | `lossless-within-dtype` | `lossless-within-dtype` | abs./rel. `0`; cosseno/top-k/argmax `1` | `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e` |
| vídeo, 1 frame | `lossless-within-dtype` | `lossless-within-dtype` | abs./rel. `0`; cosseno/top-k/argmax `1` | `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d` |
| áudio, 1 frame | `approximate` | `approximate` | abs. `0.3125`; rel. `2073.9473684210525`; cosseno `0.9998730821941509`; top-k/argmax `1` | `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4` |

Todos geraram o token `184` e acessaram somente o artefato.

## Limites preservados

- 64 instâncias vision score/value e 36 instâncias audio AC/BD/value continuam
  `fail-closed-runtime-reduction` porque a árvore F32 interna do Apple
  Accelerate `sgemm` não é publicada.
- Áudio diverge desde `composite_audio_features`, inclusive logits e caches.
- A tabela resolve bits de constantes formulares; ela não transforma uma
  redução nativa desconhecida em uma operação literal.
- `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

# Gemma 4 E4B: operandos aprendidos explícitos sem fonte — 2026-07-18

## Resultado candidato

O artefato literal schema v4 liga cada consumidor de storage a papéis
semânticos e expressões de índice lógico antes da decodificação row-major. A
vista escalar deixou de selecionar peso, bias, bounds de clipping, tabela de
posição, kernel, gamma ou escala por rank, shape, posição no array ou sufixo do
nome. O leitor valida o contrato inteiro contra o programa incorporado e falha
se um papel, tensor, decoder ou índice for alterado.

O contrato real contém 1.152 atribuições com 2.081 referências de operandos e
2.076 nomes de tensor únicos. Os outros 54 tensores do pacote são os K/V locais
explicitamente inatingíveis dos consumidores shared-KV. Portanto os 2.130
tensores do checkpoint estão particionados sem omissão nem sobreposição entre
operandos alcançáveis e constantes inatingíveis declaradas.

Papéis serializados no programa real:

| papel | referências |
|---|---:|
| `weight` | 598 |
| `normalization-scale` | 485 |
| `position-table` | 1 |
| `input-min` / `input-max` | 232 / 232 |
| `output-min` / `output-max` | 232 / 232 |
| `convolution-kernel` | 14 |
| `per-dimension-scale` | 12 |
| `bias` | 1 |
| `tensor-scale` | 42 |

## Identidade e payload

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- fonte: Safetensors BF16 denso sem quantização;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21.330.293.031`;
- SHA-256:
  `63880103e26346b3302144d21eec7c0e652828de255c46afc39af6efd3b2e958`;
- constantes: 2.130;
- bytes de payload: 15.992.314.836;
- SHA-256 agregado da origem e do literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho proibido da origem serializado: `false`.

O audit fonte-para-literal foi gravado em
`/private/tmp/gemma4-loop57-learned-operands-audit.json`, SHA-256
`76525503f4d17d0136b77bc6cdd2034138eeb9df78aba2c20c321844d4aed7d4`.

## Navegação e substituição com o checkpoint ausente

O diretório `gemma-4-E4B-dense` foi movido para fora do caminho durante a
verificação e restaurado por `trap`. A verificação interna dos payloads, a
navegação e a vista escalar declararam `sourceCheckpointAccessed=false`.

A navegação expandida contém 2.709 operações. Destas, 1.359 instâncias expõem
operandos aprendidos, correspondendo a 1.148 definições navegáveis; as quatro
definições text-prelude restantes estão presentes no contrato integral, mas o
forward composite entra no texto já preparado e não as reinstancia.

Para `composite_image_features/vision_layer_0_q[0,0,0]`, a navegação expôs os
papéis `weight`, `input-min`, `input-max`, `output-min` e `output-max`. Uma
janela de quatro termos decodificou os bits BF16 `0xbc40`, `0xbbc0`, `0xbbc0`
e `0xbc61` como `-0.01171875`, `-0.005859375`, `-0.005859375` e
`-0.01373291015625`. Os bounds foram substituídos como `-6.375`, `6.3125`,
`-11.3125` e `11.1875`, todos com seus próprios bits e decoder IDs. A redução
continua completa em `0..767` com a agenda ARM NEON BF16 de 32 lanes; a janela
marca 764 termos omitidos da renderização, não do cálculo.

Hashes dos relatórios source-removed:

- payloads:
  `1719aabd1aee4d6b07e3fefdae174de91d3549060fe4f5116c19e9f0b486d17f`;
- navegação:
  `45919c98df57731fa810ef63e7bb84fd4f88dc2d894c62b07134b20ee9f4107b`;
- substituição escalar:
  `237de9be666da20116b003215f935c77a95f7d82182488ed6713bede11afaecc`.

## Replay composite source-removed

As capturas autoritativas imutáveis do loop 54 foram repetidas contra schema
v4 com tolerâncias zero:

| modalidade | prefill | geração | terminal logits | token |
|---|---|---|---|---:|
| imagem | `lossless-within-dtype` | `lossless-within-dtype` | erro absoluto/relativo zero | 184 |
| vídeo de um frame | `lossless-within-dtype` | `lossless-within-dtype` | erro absoluto/relativo zero | 184 |
| áudio de um frame | `approximate` | `approximate` | abs. 0,3125; rel. 2073,9473684210525 | 184 |

Os relatórios permaneceram byte-idênticos aos anteriores:

- imagem: `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e`;
- vídeo: `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d`;
- áudio: `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4`.

## Limite numérico preservado

O source pinned do PyTorch despacha BMMs abaixo do limiar de 400 termos para
um loop C++ ordenado, mas as shapes reais de atenção excedem esse limiar e
caem em `addmm` por batch. Esta build declara `BLAS_INFO=accelerate`; a árvore
interna do Apple Accelerate não é exposta pelo modelo nem pelo source do
PyTorch. Por isso nenhuma agenda foi ajustada a partir das amostras.

Continuam fail-closed cinco classes: score/value vision BF16 (32 definições,
64 instâncias imagem/vídeo) e content-score, position-score e value audio F32
(36 definições). Áudio continua aproximado e o marcador
`.agent-loop/checkpoints/gemma4-dense-lossless/` permanece proibido.

## Comandos principais

```bash
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop57-learned-operands-audit.json \
  --verify-gemma4-payloads

node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-operations \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop57-source-removed-learned-navigation.json
```

Esta sessão registra uma alegação candidata de formato/substituição; ela não
aceita o checkpoint que a própria sessão modificou.

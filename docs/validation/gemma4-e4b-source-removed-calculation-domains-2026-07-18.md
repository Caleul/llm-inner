# Gemma 4 E4B: domínios completos de cálculo sem fonte — 2026-07-18

## Resultado

O artefato literal schema v3 agora serializa um contrato de coordenadas para
cada valor nomeado, em vez de exigir que o leitor deduza shapes a partir do
nome da atribuição. Cada domínio contém dtype, layout, shape simbólico, eixos,
bounds de índice e a política de compute/acumulação/cast da operação.

O contrato possui 2.297 definições validadas: 22 composite, 427 vision, 619
audio, 8 text-prelude, 1.218 text-layer e 3 text-epilogue. A expansão
composite instancia vision separadamente para imagem e vídeo, produzindo 2.709
operações navegáveis, todas com `outputDomain`. Dimensões dinâmicas são
expressões de inputs declarados (`B`, `S`, `K`, `IMAGE_PATCHES`,
`VIDEO_BATCH*VIDEO_FRAMES`, `AT2`, `ABLOCKS`); dimensões de arquitetura são
literais, como hidden `2560`, 12 heads de visão e vocab `262144`.

Os domínios não escondem as fronteiras numéricas pendentes. Os scores vision
continuam `BF16/pytorch-native-batched-matmul/runtime-defined`, e os scores
audio continuam `F32/pytorch-cpu-f32-matmul/runtime-defined`. A vista escalar
permanece fail-closed para essas classes.

## Correção das vistas do prelude textual

As operações composite compatíveis do prelude textual antes reutilizavam os
bytes corretos, mas suas vistas didáticas locais apresentavam fórmulas F32
genéricas. O dispatch por classe de operação agora reutiliza as definições e
políticas autoritativas do programa text para embedding, identidade PLE,
projeção contextual, escala, reshape, RMSNorm e soma. Ele falha fechado quando
não existe uma definição compatível única.

Com a origem ausente, o token `2`, feature `0`, produziu:

```text
composite_text_embeddings[0,0,0] =
  BF16_RNE(F32(-0.0294189453125 * 50.5))
```

O peso foi decodificado do payload BF16 `0xbcf1`, row-major index `5120`, pelo
decoder `ieee-bf16-to-f32`. Assim, número aprendido, bits, decoder, escala
BF16 e cast de saída estão ligados na mesma vista.

A projeção contextual também deixou de apresentar uma redução genérica. A
coordenada `[0,0,0]` declara redução completa `0..2559`, produtos BF16,
acumulação `reduce_arm-neon-bf16-dot-fma`, fold pareado de 32 lanes e cast
BF16 de saída. A janela auditada `0..3` substituiu os quatro valores aprendidos
por seus números e bits reais e marcou explicitamente os 2.556 termos omitidos
da visualização, sem truncar o domínio matemático.

## Identidade e integridade

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- pacote: Safetensors BF16 denso sem quantização;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,329,687,350`;
- SHA-256:
  `51aa4d605060c2b69e3245f3196876199e8ff6757cba8b49150cf87161b93d3f`;
- constantes: `2.130`;
- payload incorporado: `15.992.314.836` bytes;
- digest de storage da origem e do literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho da origem serializado: `false`.

O audit fonte-para-literal comparou todos os bytes. Depois, com
`gemma-4-E4B-dense` renomeado e inexistente, a verificação interna dos 2.130
compromissos passou e declarou `sourceCheckpointAccessed=false`.

Os relatórios source-removed finais ligam o mesmo artefato às vistas e à
navegação: payloads
`3e2abb574e6f51fe5081eb61f47868553096f8442874db3307b74855a2a76721`,
navegação
`ccf1679e84f38bac63d822e97a02993cedf03de5248e2d2ab929d2d3d8c281d0`,
embedding
`f34705ab4c5064a50b9ab539bc09ef36aecadfd75adbed22b77401bdde2d7371`
e projeção contextual
`51af9166df03d20f15103e116ceff54bcb25adf35f28f696ca21d24f73dafb0e`.
O relatório de navegação declarou 2.709 operações forward, todas com domínio,
além de 20 operações concretas de controle/geração.

## Replay composite sem checkpoint

As capturas autoritativas do loop 54 foram repetidas contra o schema v3 com a
origem ausente:

- imagem: prefill e um passo greedy `lossless-within-dtype`; logits, 24 caches
  produtores, token `184`/posição `2` e terminal com erro zero;
- vídeo de um frame: o mesmo resultado `lossless-within-dtype` e erro zero;
- áudio de um frame: `approximate`; primeira divergência composite em
  `audio_features` (`maxAbsoluteError=0.08203125`), logits de seleção com erro
  máximo `2`, terminal com erro máximo `0.3125`, token final ainda `184`, e
  caches divergentes.

Os relatórios reproduziram byte a byte os hashes anteriores apesar do novo
índice de domínios:

- imagem: `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e`;
- vídeo: `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d`;
- áudio: `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4`.

## Comandos reproduzidos

```bash
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop56-final-domain-audit-v2.json \
  --verify-gemma4-payloads

node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-generation-operations --generation-max-new-tokens 2 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop56-final-source-removed-navigation-v2.json
```

Os três replays usaram
`gemma4-literal-composite-differential-cli.js`, as capturas
`gemma4-loop54-{image,video,audio}-composite-trace.json`, tolerâncias zero e
`--assert-source-unavailable ./gemma-4-E4B-dense`.

## Limite preservado

Esta sessão fecha uma fronteira de formato/navegação e registra uma alegação
candidata; não aceita o checkpoint Gemma 4. Os 36 matmuls F32 nativos de áudio
e os batched matmuls internos BF16 de visão/vídeo ainda não possuem agenda
escalar geral comprovada. O marcador
`.agent-loop/checkpoints/gemma4-dense-lossless/` continua proibido.

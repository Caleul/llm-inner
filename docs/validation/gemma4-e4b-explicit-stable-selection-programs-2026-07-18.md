# Gemma 4 E4B: seleção estável explícita — 2026-07-18

## Fronteira fechada

A revisão independente do schema v21 confirmou os programas de indexação e
RoPE e manteve o checkpoint não aceito. A inspeção das fórmulas revelou outra
classe de opacidade navegacional: cinco scatters multimodais usavam
`next_feature_row`/`placeholder_at`, e os dois `strip-padding` usavam
`stable_batch_major_true_mask_row`. Esses nomes não declaravam o índice que
liga a linha modal projetada ao token textual nem a coordenada que sobrevive à
remoção de padding.

O schema v22 fecha a classe completa sem dispatch por assignment ID:

- cada assignment de scatter serializa `placeholderTokenId` obtido do contrato
  Gemma 4 autoritativo;
- `STABLE_TRUE_COUNT(mask)` percorre a matriz BOOL completa em ordem
  batch-major;
- `STABLE_TRUE_PREFIX_RANK(mask,batch,sequence)` retorna o rank exclusivo da
  coordenada verdadeira, ou `-1` para uma coordenada falsa;
- `STABLE_TRUE_COORDINATE_AT_RANK(mask,rank)` faz a inversa e retorna o par
  fonte `[batch,sequence]` da linha válida;
- todos os três algoritmos são programas finitos em
  `formulaLanguage.indexing.programs`, possuem executores independentes e
  recusam programa alterado, matriz irregular, coordenada ou rank inválido;
- as cinco fórmulas de scatter declaram o token concreto, o prefix rank, a
  linha de feature e a igualdade entre contagem de placeholders e linhas;
- os dois `strip-padding` declaram o rank de saída, a coordenada fonte e a
  igualdade entre domínio de saída e contagem verdadeira;
- ausência do token modal autoritativo falha fechado antes da serialização.

## Artefato real

Fonte imutável: `google/gemma-4-E4B`, revisão
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
quantizado. O mesmo comando streaming documentado no README gerou:

- schema do artefato: `22`; schema da linguagem de fórmulas: `8`; schema dos
  programas de indexação: `2`;
- bytes: `21,384,805,251`;
- SHA-256: `3d6e18cc4f920cd03c455598d5967b056155eb314ebf78978347114ca5a5a770`;
- constantes/decoders: `2,130/2,130`;
- bytes aprendidos incorporados: `15,992,314,836`;
- atribuições forward instanciadas: `2,709`; arestas produtoras: `3,363`;
- cinco fórmulas de scatter e duas de strip-padding sem qualquer ocorrência
  dos três placeholders opacos anteriores.

A auditoria source-present comparou os 2.130 payloads e todos os
15.992.314.836 bytes com os seis arquivos da fonte. O SHA-256 concatenado do
storage foi
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`
nos dois lados. Relatório
`/private/tmp/llm-inner-loop78-v22-source-audit.json`, SHA-256
`0c5bccfa0a1ca665bc17928184a904fa68490c6689cd2f064820b9db4c9f73a5`.

## Evidência source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente para fora do caminho
declarado sob restore trap. Sem a fonte, a inspeção verificou os 2.130
payloads, 15.992.314.836 bytes, 2.709 operações forward, 12 operações de
geração, cobertura completa de 2.076 constantes alcançáveis e 54 constantes
KV locais explicitamente inalcançáveis. Relatório
`/private/tmp/llm-inner-loop78-v22-source-removed.json`, SHA-256
`38ca32b8beb12a3ec4ebff77063a3453004907468df5ea36859f47a5405a8f62`.

A vista `composite_image_scatter[0,0,0]` materializou o token real `258880`,
o prefix rank e a seleção de `image_features[feature_row,0]`. A vista
`composite_audio_features/audio_strip_padding[0,0]` materializou a inversão do
rank por `STABLE_TRUE_COORDINATE_AT_RANK`. Ambas registraram
`sourceCheckpointAccessed=false`; seus relatórios têm SHA-256
`3718191b720a5b9757705b393d19fafbb9cd010f187178e085f3184738121b73`
e `12c008f8a4dffd998662b03b9621ca6e9db09e89f207a931fd4b9771d23eeba8`.

O replay composite de áudio permaneceu `lossless-within-dtype` com tolerâncias
absoluta e relativa zero: prefill sem primeira divergência, token 184 na
posição 2, logits e 24 caches KV concordantes. Relatório
`/private/tmp/llm-inner-loop78-v22-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Validação e limite preservado

`npm test` passou 234 testes, zero falhas, antes da documentação final. O teste
focado passou 32 casos e cobre contagem, ranks direto/inverso, corrupção do
programa, token ausente e todas as sete fórmulas da classe.

Esta é uma alegação candidata de fechamento da seleção estável, não do
checkpoint Gemma 4. A vista end-to-end permanece
`fail-closed-runtime-reduction` para 100 BMMs nativos: 64 de visão e 36 de
áudio. A árvore escalar do Apple Accelerate continua sem contrato publicado e
não foi aproximada. `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi
criado.

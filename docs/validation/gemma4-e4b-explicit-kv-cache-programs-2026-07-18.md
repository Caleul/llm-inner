# Gemma 4 E4B: programas escalares explícitos de cache KV — 2026-07-18

## Fronteira candidata

A revisão independente começou no commit
`0e6abcae9669fa0f7e08ec282cb8539618988b2c`, inspecionou o diff e o handoff do
schema v23 e repetiu `npm run typecheck` e `npm test`: os 234 testes passaram.
Isso aceita a fronteira de coordenadas do loop anterior dentro do seu escopo;
não aceita o checkpoint Gemma 4.

O artefato ainda representava cada transição de cache por texto livre:
`past_key_values[layer] = {...}` para produtores e "reads producer cache" para
consumidores. O schema v24 substitui a classe inteira por programas indexados,
derivados da topologia `scaled_dot_product_attention` já validada:

- as 24 camadas produtoras copiam K pós-RoPE e V normalizado no prefill sobre
  todo o domínio `[batch,kv_head,sequence,head_feature]`;
- no decode, cada produtor declara `previous_sequence_length`,
  `current_sequence_length` e `next_sequence_length`, preserva cada elemento
  anterior quando `sequence < previous_sequence_length` e lê o elemento atual
  em `sequence-previous_sequence_length` no restante;
- os 18 consumidores shared-KV exigem `producerLayer < layer`, leem K/V do
  cache completo que o produtor já publicou e declaram explicitamente que não
  emitem uma entrada própria;
- shape, dtype F32/BF16, layout BHSD, eixo de concatenação 2, domínio completo e
  ordem lexicográfica são dados do programa;
- as fórmulas de prefill e incremental entram na validação de helpers e na
  tabela de bits numéricos. Alterar um programa é rejeitado pela reconstrução
  canônica, inclusive para shared-KV.

Não há dispatch por assignment ID: o construtor percorre toda operação de
atenção textual e escolhe `write-producer` ou `reuse-producer` somente pelo
contrato explícito `kvSharing`.

## Artefato real e storage lossless

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
quantizado. A geração streaming produziu:

- schema do artefato `24`, linguagem de fórmulas `10` e cálculo forward de
  geração schema `2`;
- bytes: `21,384,958,828`;
- SHA-256:
  `9f8ba30f420d0b8cacea01f0fa09ece95362832054c309cab19735e6df6f7564`;
- constantes/decoders: `2,130/2,130`;
- bytes aprendidos incorporados: `15,992,314,836`;
- atribuições forward: `2,709`;
- transições KV: `42`, sendo `24` produtoras e `18` reutilizações, todas com
  programas de prefill e incremental schema `1`.

`npm run audit:literal -- --verify-gemma4-payloads` comparou todos os 2.130
payloads e cada um dos 15.992.314.836 bytes com os seis arquivos da fonte. Os
dois lados produziram SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O relatório é `/private/tmp/llm-inner-loop80-v24-source-audit.json`, SHA-256
`57210f9b0de701b3b5dc5eb1f08b78176254711e46dbcf9582d2f7e30b2f622d`.

## Prova source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente sob restore trap. Com
o caminho ausente, o reader reabriu somente o JSON, validou os 42 contratos de
cache, verificou os 2.130 compromissos e leu todos os 15.992.314.836 bytes.
O relatório registra `sourceCheckpointAccessed=false`, 2.709 operações forward
e 100 operações ainda fail-closed. Ele está em
`/private/tmp/llm-inner-loop80-v24-source-removed.json`, SHA-256
`4561a196908f29b633c2d9aef4f2eddb9458afd113cef08ed6105bef713baf0a`.

O replay composite de áudio usou a captura autoritativa que declara
`torch.inference_mode` e atenção eager. Uma captura antiga sem esses dois
campos foi corretamente recusada com
`executionMode='undefined', attentionImplementation='undefined'`; nenhum
contexto foi inferido. Com a captura válida e o checkpoint ausente, prefill e
um passo greedy permaneceram `lossless-within-dtype` em tolerância absoluta e
relativa zero: cinco fronteiras compostas, token `184` na posição `2`, logits
e todos os 24 caches produtores passaram sem primeira divergência. O relatório
é `/private/tmp/llm-inner-loop80-v24-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Limite preservado

Esta implementação é evidência candidata do fechamento das transições KV
literais, não do checkpoint. As 100 instâncias Apple Accelerate BMM — 64 de
visão e 36 de áudio — continuam com agenda escalar
`unpublished-fail-closed`. O schema v24 não inventa essa árvore privada e a
vista end-to-end continua listando exatamente esses 100 IDs. Por isso
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

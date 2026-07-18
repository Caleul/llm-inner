# Gemma 4 E4B: programas explícitos de coordenadas — 2026-07-18

## Fronteira candidata

A revisão independente começou no commit
`5b9d0c3b5717bd5dbd524296456145a4d64f2b2e` e repetiu `npm run typecheck` e
`npm test`: os 234 testes passaram. Isso aceita somente a seleção estável do
schema v22 como evidência corrente; não aceita o checkpoint Gemma 4.

A inspeção das fórmulas encontrou uma classe implementável que não depende da
árvore privada do Apple Accelerate: pooling vision, relative shift audio,
índices chunked de áudio e aliases de heads ainda conservavam descrições como
"patches mapped to pool_cell", "any non-padding patch" ou
"after declared pad-flatten-slice-reshape". Essas frases descreviam o efeito,
mas não eram programas finitos suficientes para replay independente.

O schema v23 fecha a classe completa por operação e forma, sem dispatch por
assignment ID:

- `VISION_POOL_SLOT` reproduz `max_x`, colunas, padding e o slot derivado de
  `[x,y]`; `VISION_POOL_CELL_HAS_PATCH` percorre todo o domínio de patches em
  ordem ascendente;
- o cálculo de pooling declara `acc[-1]=F32(0)` e cada atualização
  `F32_FMA`, enquanto a máscara usa o programa booleano incorporado;
- `AUDIO_RELATIVE_SHIFT_SOURCE` reproduz o `pad/view/slice/view` e retorna
  `valid`, `query_in_block` e `relative_index`;
- content score, position score, eager mask e value BMM declaram
  `query_index`, `key_index`, sequência, bounds, head, block e query local;
- os 90 `reshape_heads` usam `row_major_alias` e o operador `%` ganhou
  semântica I32 explícita;
- o construtor rejeita qualquer operação dessa classe que volte a omitir seu
  programa de coordenadas, e os executores recusam programas serializados
  alterados.

No artefato real há uma definição de pool, uma de máscara pool, 12 de cada
operação chunked/relative de áudio e 90 definições de reshape. O grafo possui
duas instâncias de pool/máscara (imagem e vídeo), 12 instâncias de cada classe
de áudio e 90 instâncias de reshape. Nenhuma fórmula contém os quatro trechos
opacos anteriores.

## Artefato real e integridade

Fonte imutável: `google/gemma-4-E4B`, revisão
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
quantizado. A geração streaming produziu:

- schema do artefato `23`, linguagem de fórmulas `9`, programas de indexação
  `3`;
- bytes: `21,384,850,237`;
- SHA-256:
  `90ba8dfbde7d7fcc8e27f027d760062531e327b205141a563e4473b2e2936f6d`;
- constantes/decoders: `2,130/2,130`;
- bytes aprendidos incorporados: `15,992,314,836`;
- atribuições forward: `2,709`.

A auditoria source-present comparou os 2.130 payloads com os seis arquivos da
fonte. Os dois lados produziram o SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O relatório é `/private/tmp/llm-inner-loop79-v23-source-audit.json`, SHA-256
`a3a2f1dda933470f187550215584c643b587130d2d6aa894d3f3c39138778876`.

## Prova source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente sob restore trap.
Enquanto o caminho estava ausente, o reader reabriu somente o JSON e verificou
os 2.130 compromissos e todos os `15,992,314,836` bytes, com
`sourceCheckpointAccessed=false`. Relatório
`/private/tmp/llm-inner-loop79-v23-source-removed.json`, SHA-256
`b97d70546abec1b33531f3a99bffdd019a414aba4129bba7bb1f22bee76d7353`.

As vistas source-removed materializaram:

- `vision_pool[0,0,0]`: slot por `VISION_POOL_SLOT`, acumulador inicial F32,
  loop completo por patch, `F32_FMA` e cast BF16; SHA-256
  `c441cbff90915fc05211003e2f321742cee0894e8d3e4ce9f79f1ff14f2d8601`;
- `vision_pool_mask[0,0]`: `VISION_POOL_CELL_HAS_PATCH` sobre a linha completa;
  SHA-256
  `16de792cf208179ff287bd0a1dbc4ce02fd36101a115085237745024744f04ca`;
- `audio_layer_0_attention_relative_shift[0,0,0,0,0]`: programa com contexto
  24, relative length 13 e coordenada fonte concreta `[0,0]`; SHA-256
  `52c767d858084fb28de7bf871ae16c1d42f1af6983072d311950fd44b38fc48c`.

A navegação end-to-end source-removed preservou as 2.709 operações forward,
12 operações para um passo greedy e controle de geração literal. Seu relatório
tem SHA-256
`6aa3abb7d0aabd9c1660c6f38c7cf23b9ccc9ff690a2dabf5ac9711ef6769196`.

O replay composite de áudio registrado permaneceu
`lossless-within-dtype` em tolerância absoluta e relativa zero: prefill sem
primeira divergência, token 184 na posição 2, logits e todos os 24 caches KV
passaram. Relatório
`/private/tmp/llm-inner-loop79-v23-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Limite preservado

Esta implementação é evidência candidata do fechamento das coordenadas
autoritativas, não do checkpoint. As 100 instâncias Apple Accelerate BMM — 64
de visão e 36 de áudio — continuam com agenda escalar
`unpublished-fail-closed`; o end-to-end declara exatamente esses 100 IDs como
`fail-closed-runtime-reduction`. Nenhuma ordem escalar foi inferida a partir do
replay terminal. Por isso `.agent-loop/checkpoints/gemma4-dense-lossless/` não
foi criado.

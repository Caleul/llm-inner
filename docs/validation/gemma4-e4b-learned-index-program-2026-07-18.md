# Gemma 4 E4B: programa de índices aprendidos — 2026-07-18

## Resultado candidato

O artefato dense BF16 real `google/gemma-4-E4B` foi promovido ao schema v11.
Os índices lógicos de todo peso, bias, limite de clipping, tabela posicional,
kernel, escala de normalização e escala por dimensão deixaram de ser strings
interpretadas pelo renderer. `learnedOperands.indexLanguage` incorpora a AST
`gemma4-learned-index-expression-v1`, e todos os scalar views usam o mesmo
avaliador genérico antes do decoder row-major.

A AST distingue:

- coordenadas nomeadas do domínio de saída;
- índices de redução com `minInclusive` e `endExclusive`;
- escalares explícitos do caller, também com bounds;
- constantes inteiras; e
- `add`, `multiply` e `modulo` em ordem depth-first left-to-right.

O avaliador exige inteiros seguros não negativos, valida bounds em cada folha,
valida o índice final contra cada dimensão do tensor e só então calcula o
offset do decoder. Binding ausente, overflow, módulo inválido e out-of-bounds
falham fechados.

## Cobertura do contrato real

O leitor bounded-memory validou no JSON:

- `1.152` assignments que consomem storage aprendido;
- `2.081` operandos aprendidos;
- `1.113` folhas de coordenada de saída;
- `612` folhas de índice de redução;
- `6` folhas de input scalar;
- `68` constantes, `2` adds, `2` multiplies e `12` módulos; e
- todos os `decoderId` iguais a `decode_<tensor-name>`.

Os `2.081` operandos cobrem `598` weights, `1` bias, `485` escalas de
normalização, `1` tabela posicional, `14` kernels de convolução, `12` escalas
por dimensão, `42` tensor scales e `928` limites de clipping.

## Identidade e payloads

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21.382.503.979`;
- SHA-256: `e89442dff45852fd954a03fee9171d80ea26319bd5067c6b6ede92eec1acab0b`;
- constantes: `2.130`;
- payload aprendido: `15.992.314.836` bytes.

`npm run audit:literal -- --artifact ./artifacts/gemma4-e4b-dense.literal.json
--source ./gemma-4-E4B-dense --output
/private/tmp/llm-inner-loop67-source-audit.json --verify-gemma4-payloads`
recalculou os seis arquivos e `16.024.773.810` bytes da identidade imutável.
Todos os payloads comparados produziram o mesmo digest source/literal:
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

## Prova source-removed

O diretório `gemma-4-E4B-dense` foi movido para um diretório `mktemp`, com
restauração por `trap`, enquanto o leitor executava `--verify-payloads`,
`--end-to-end-calculation --generation-max-new-tokens 2` e cinco scalar views.
O relatório principal registrou `sourceCheckpointAccessed: false`, validou os
`2.130` payloads, as `2.709` assignments forward, `20` assignments greedy,
`42` transições KV e a partição completa `2.076 + 54 = 2.130` do storage. O
relatório tem `57.333.668` bytes e SHA-256
`83c341c87278d6178dde84b90eff56a2e75e85313d0a32810b83b3cf9009bc3c`.

As vistas source-removed provaram substituição do literal decodificado para:

- `layer_0_q_proj`: `[output_feature,input_feature]` -> `[0,0..3]`;
- `composite_ple_identity`: `token_id` e
  `layer*per_layer_width+ple_feature` -> `[1,258]`;
- `vision_position_embedding`: axis/position/hidden -> `[0,1,2]` e `[1,0,2]`;
- `audio_subsample_0_conv`: os nove índices
  `[out,in,kernel_time,kernel_feature]`; e
- `audio_layer_0_q_scale`: `hidden modulo head_dim` -> `[1]`.

Cada vista marcou `sourceCheckpointAccessed: false`, carregou bits/decoder do
JSON e continha todo literal decodificado na fórmula escalar retornada.

## Validação e limite preservado

`npm run typecheck` passou. `npm test` passou `227/227` testes, incluindo
execução da AST em todos os renderers de storage, rejeição de binding ausente,
escrita in-memory/streaming schema v11 e replay source-removed de fixture.

Esta mudança não inventa as árvores das cinco classes de batched matmul nativo.
As `100` instâncias vision/audio continuam
`fail-closed-runtime-reduction`, e a fidelidade audio continua aproximada. O
marcador `.agent-loop/checkpoints/gemma4-dense-lossless/` não é justificado.

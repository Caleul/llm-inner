# Gemma 4 E4B: programas escalares em ordem de dependência — 2026-07-18

## Fronteira candidata

O loop 84 revisou independentemente o commit
`9b03541c149abde91e75ee31b9264743e60048b0` e repetiu os comandos configurados:
`npm run typecheck` passou e os 234 testes passaram. Isso aceita a fronteira de
domínios de redução executáveis do schema v27 dentro de seu escopo; não aceita
o checkpoint Gemma 4.

A revisão encontrou uma lacuna distinta na execução do programa escalar. A
vista `formula` era intencionalmente output-first, porém fórmulas multistep
declaravam depois da saída os valores dos quais ela dependia. Por exemplo,
atenção declarava `context = ... probability[...]` antes de `score`, `maximum`,
`exponential`, `total` e `probability`; RoPE declarava a saída antes de `pair`,
`angle`, `cosine` e `sine`. Um leitor conseguia auditar a expressão, mas ainda
precisava inventar uma reordenação para executá-la.

O schema v28 preserva `formula` como renderização humana compacta e acrescenta
`scalarAssignments` a toda atribuição forward. Esse array é o programa
autoritativo em ordem de dependência:

1. coordenadas e temporários locais são materializados na ordem declarada;
2. precondições `require` são avaliadas antes da saída que protegem;
3. a atribuição final ao `output` é sempre o último statement;
4. o construtor falha se qualquer local for lido antes de sua declaração; e
5. a instanciação vision/audio/text liga nomes de call site em todos os
   statements, não apenas na vista `formula`.

O contrato é por classe de fórmula e cobre todos os layers compatíveis de uma
vez. Não adiciona allowlist de assignment IDs e não altera as reduções cuja
agenda nativa continua desconhecida. `scalarCalculations` avançou para schema
3, `formulaLanguage` para schema 14 e o artefato composto para schema 28.

## Artefato real e integridade

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors denso BF16/F32 sem
quantização.

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Resultado:

- artefato schema `28`, linguagem de fórmulas schema `14`, cálculos escalares
  schema `3`;
- bytes: `21.387.299.594`;
- SHA-256:
  `e2933fe31255035342b7d2fdf220987c50c3b5b6999a6865affe78017251a1b6`;
- constantes/decoders: `2.130/2.130`;
- bytes de payload incorporados: `15.992.314.836`;
- atribuições/statements escalares: `2.709/4.146`;
- `1.437` passos locais ou precondições em `288` atribuições multistep;
- `8` statements `require` explícitos.

O audit source-present:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop84-v28-source-audit.json \
  --verify-gemma4-payloads
```

Todos os 2.130 payloads e os 15.992.314.836 bytes compararam iguais. Source e
literal produziram o mesmo SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O relatório tem SHA-256
`bbbdc84078c4e48db102a41de21d0ea57fc0f57203c1b457cbb718857bb82e89`.

## Leitura source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente e restaurado por trap.
Enquanto o caminho estava ausente, o leitor executou:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop84-v28-source-removed.json
```

O relatório, SHA-256
`3784fee705411bc236d715a20906522e4677fdd9f177b266e8e7522b4bb5b2e9`,
registrou `sourceCheckpointAccessed=false`, verificou todos os payloads e
reconstruiu 2.709 operações forward, 20 operações de geração e os 4.146
statements escalares na ordem incorporada. O contrato de reduções permaneceu
inalterado: 1.462 reduções, 256 estágios, 1.722 domínios e exatamente 100
reduções `runtime-defined`.

## Regressão autoritativa

Ainda com a fonte ausente, o artefato foi comparado a zero tolerância com
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`:

```bash
node dist/src/gemma4-literal-composite-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop84-v28-composite-audio.json \
  --absolute-tolerance 0 --relative-tolerance 0 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --allow-unverified-fidelity
```

Prefill e geração passaram como `lossless-within-dtype`: todas as cinco
fronteiras forward tiveram erro absoluto/relativo zero, cosine/top-k 1 e
argmax agreement; o token greedy foi `184` na posição `2`, com todos os caches
KV iguais e nenhuma primeira divergência. O relatório tem SHA-256
`30b4b49783f542ab2540269cd7530391c4eaf37d0f5ce31b4bfb68b21e32f241`
e avaliou 1.138 domínios ativos em 996 atribuições, incluindo as 36 BMM de
áudio.

## Limite preservado

Esta mudança resolve a ordem dos statements com semântica estabelecida; ela
não inventa a árvore escalar interna de Apple Accelerate SGEMM. As 64 BMM de
visão e 36 BMM de áudio permanecem `fail-closed-runtime-reduction`, e o uso de
`--allow-unverified-fidelity` continua explícito. O schema v28 é evidência
candidata deste loop; `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi
criado.

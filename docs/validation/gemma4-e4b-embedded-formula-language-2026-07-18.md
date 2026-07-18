# Gemma 4 E4B: linguagem de fórmulas incorporada — 2026-07-18

## Resultado candidato

O artefato real dense BF16 `google/gemma-4-E4B` agora usa schema v10 e carrega
o contrato normativo de `indexed-ieee754-expression-v1` dentro do próprio JSON.
Antes, `scalarCalculations` e `generation.scalarCalculations` nomeavam essa
linguagem, mas a ordem de avaliação, os casts, a resolução de valores
aprendidos e os intrínsecos ainda precisavam ser entendidos pelo código ou pela
documentação do repositório.

`formulaLanguage` passa a declarar:

- os JSON pointers normativos para fórmulas forward/greedy, grafo instanciado,
  domínios, bindings aprendidos, decoders e bits numéricos;
- ordem de dependências, coordenadas row-major e bindings posicionais;
- tipos e materializações F64, F32, BF16, I32 e BOOL;
- operadores, indexação e treze famílias de intrínsecos;
- execução de reduções ascendentes e agendas por operação; e
- rejeição obrigatória de qualquer redução `runtime-defined`.

O leitor reconstrói o contrato canônico e rejeita qualquer palavra alterada,
inclusive quando a estrutura restante do programa e seus payloads permanecem
válidos. Esta mudança não inventa a árvore interna dos cinco batched matmuls
nativos ainda bloqueados e portanto não autoriza o marcador de checkpoint.

As regressões de escrita in-memory e streaming, abertura bounded-memory,
corruptela e replay de fixture passaram em `npm run typecheck` e `npm test`:
227 testes, 227 sucessos e zero falhas.

## Identidade e artefato

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- formato: Safetensors BF16 denso, sem quantização;
- artefato: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21.382.310.312`;
- SHA-256:
  `72fb8eaa2a74bd98739543e44a5f4571a5c0ec2c7d0121aaa5736eb290f7d0c7`;
- constantes incorporadas: `2.130`;
- payload aprendido: `15.992.314.836` bytes.

## Verificação source-present

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop66-source-audit.json \
  --verify-gemma4-payloads
```

O auditor recalculou os seis arquivos e `16.024.773.810` bytes da identidade
imutável. Comparou os 2.130 payloads e todos os `15.992.314.836` bytes
aprendidos; origem e literal produziram o mesmo digest agregado
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
Nenhum caminho local da fonte foi serializado.

Isso também revalida independentemente a alegação candidata de identidade do
schema v9: modelo, revisão, seis compromissos de arquivo e metadata Base64
permanecem íntegros depois da evolução de formato.

## Verificação source-removed

Depois do build, `gemma-4-E4B-dense` foi movido para um diretório temporário e
restaurado por `trap` somente após o comando terminar:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --verify-payloads \
  --end-to-end-calculation \
  --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop66-source-removed.json
```

Resultado:

- `schemaVersion: 10` e `sourceCheckpointAccessed: false`;
- linguagem `indexed-ieee754-expression-v1`, cinco tipos escalares, seis
  classes de operadores e treze famílias de intrínsecos;
- 2.130 payloads e `15.992.314.836` bytes verificados pelo digest agregado
  acima;
- 2.709 atribuições forward em ordem, das quais 2.609 literais e 100
  fail-closed;
- 20 atribuições greedy e 42 transições KV;
- cobertura de storage completa: 2.076 constantes alcançáveis e 54 tensores KV
  locais explicitamente inatingíveis pelo compartilhamento autoritativo;
- relatório: `56.822.384` bytes, SHA-256
  `048ae28688d56904dd9038dc367b4f322f089ffc7aacef433b3f53dd609661ca`.

## Limite preservado

As 64 instâncias vision e 36 instâncias audio das cinco classes de batched
matmul continuam `fail-closed-runtime-reduction`. O PyTorch fixado delega as
formas reais ao Apple Accelerate, cuja árvore escalar não é publicada. Uma
checagem class-wide das capturas existentes também não encontrou uma agenda
simples única de lanes/fold capaz de explicar as camadas de áudio. A linguagem
incorporada torna esse limite executável como erro, não como aproximação
silenciosa. Áudio composto continua numericamente aproximado, portanto
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

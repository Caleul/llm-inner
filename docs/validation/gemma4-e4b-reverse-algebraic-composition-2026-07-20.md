# Gemma 4 E4B: composição reversa algébrica

## Implementação

Há dois níveis complementares:

- `gemma4-reverse-algebraic-composer.ts` substitui definições escalares e roda
  o simplificador real exato depois de cada predecessor;
- `gemma4-parametric-reverse-composer.ts` faz o mesmo sobre as funções e somas
  paramétricas do artefato completo, preservando dimensões ainda não abertas.

As regras incluem folding racional, termos semelhantes, fatoração, paridade de
`sin`, `cos`, `tan` e `tanh`, valores em zero, `sin(x)/cos(x)=tan(x)`,
`sin(x)^2+cos(x)^2=1` e `sqrt(x^2)=abs(x)`. A última forma preserva valores
negativos. `x/x=1` foi removida porque não é exata em `x=0` sem hipótese de
domínio.

## Execução no modelo real

```bash
npm run compose:gemma4-reverse -- \
  --family terminal_logit --dimension 0 \
  --batch 0 --sequence 5 --steps 8 \
  --output /tmp/gemma4-reverse-logit-0-step8.json
```

| etapa | operação expandida | chamadas | nós antes | nós normalizados |
|---:|---|---:|---:|---:|
| 0 | `final_logit_softcap` | 1 | 3 | 7 |
| 1 | `lm_head` | 1 | 7 | 12 |
| 2 | `final_norm` | 1 | 12 | 25 |
| 3 | `layer_41_scalar` | 2 | 25 | 27 |
| 4 | `layer_41_ple_residual` | 2 | 27 | 29 |
| 5 | `layer_41_post_ple_norm` | 2 | 29 | 41 |
| 6 | `layer_41_ple_project` | 3 | 41 | 50 |
| 7 | `layer_41_ple_gated_multiply` | 1 | 50 | 52 |

Após oito substituições restaram três famílias de predecessores. O resultado
serializado contém 52 nós e representa o mesmo `terminal_logit[0]`, com
`batch=0` e `sequence=5`.

## Crescimento observado

O mesmo comando foi executado sem serializar o grafo nas janelas de 50, 200 e
500 etapas:

| etapas | operação alcançada | nós normalizados | famílias predecessor restantes |
|---:|---|---:|---:|
| 50 | `layer_40_q_norm` | 584 | 6 |
| 200 | `layer_34_q_norm` | 4.670 | 6 |
| 500 | `layer_22_gate_proj` | 24.330 | 4 |

Os marcos intermediários da execução de 200 etapas foram 584 nós na etapa 49,
1.538 na 99, 2.900 na 149 e 4.670 na 199. Essa progressão mede a composição
incremental real e substitui a extrapolação anterior baseada em expansão
sintática sem simplificação. Ela ainda não prova o tamanho da closure completa,
mas demonstra que a normalização entre operações evita a explosão imediata.

## Closure completa de `terminal_logit[0]`

A execução com `batch=0`, `sequence=5` e limite de 5.000 etapas terminou após
1.221 substituições. O grafo normalizado possui 1.643.895 nós, não contém mais
nenhuma `function-call` e foi escrito incrementalmente para evitar o limite de
comprimento de uma única string do V8:

- arquivo: `artifacts/gemma4-terminal-logit-0.reverse-composed.json`;
- tamanho: 615.043.978 bytes;
- SHA-256: `dba5254d78e3b2df316abf74dfa539a3e17fbf8ec590012ce23188800f882fd5`;
- raiz: `sha256:4df042bc38634ae973c5840e1ac9233003a9a06c8e1141e0a5a4019f537e84d5`;
- chamadas de função restantes: zero.

A operação matemática externa da dimensão é:

```text
calc_final_0 = 30 * tanh((sum(input_feature=0..2559, body_0(input_feature))) / 30)
```

`body_0` não é uma função opaca: é o nó
`sha256:3f6e9105c2b1b5b7f05a234f43653c83ca2fbebcb774ca84e7d3ccd917299293`
do mesmo grafo e sua closure contém todas as 42 camadas já substituídas. A forma
de DAG mantém subexpressões comuns compartilhadas; imprimir a árvore duplicaria
essas subexpressões e perderia justamente a redução de tamanho obtida.

## Diferencial sem arredondamentos intermediários

Com o prompt `The capital of France is` (IDs
`[2,818,5279,529,7001,563]`), a execução Transformers BF16 e a execução F64
sem arredondamentos intermediários produziram o mesmo token `496` (`" a"`).
Para a dimensão composta neste documento, ambas produziram
`terminal_logit[0] = -20.5`, portanto erro absoluto zero após o BF16 terminal.

No vetor completo de 262.144 logits, 31.015 valores foram idênticos após o BF16
terminal: taxa de divergência de 0,8816871643066406, erro absoluto médio de
0,08061687354302194 e máximo de 0,5. O argmax permaneceu `496`; seu logit mudou
de 25,625 para 25,75. O relatório executável está em
`artifacts/gemma4-terminal-logit-0.differential.json`.

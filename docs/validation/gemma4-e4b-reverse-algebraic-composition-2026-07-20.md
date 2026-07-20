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

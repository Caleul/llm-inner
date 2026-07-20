# Gemma 4 E4B: gerador de fórmula fisicamente plana

## Contrato implementado

`src/gemma4-flat-formula-generator.ts` expõe
`generateGemma4FlatFormulaObject`. Para cada output solicitado, a função:

1. substitui recursivamente toda `function-call` pelo corpo da operação;
2. substitui parâmetros formais pelos argumentos concretos;
3. desenrola somas e máximos com limites inteiros fixos;
4. solicita o valor literal de cada peso em coordenada concreta;
5. mapeia inputs para `x[...]`;
6. rejeita variáveis livres que não tenham o formato `x[...]`;
7. retorna somente `calc_final_<dimensão>: "expressão"`.

O teste unitário substituiu uma função intermediária, os pesos `-1/4` e
`-1.125`, e produziu:

```json
{
  "calc_final_0": "(((-1/4) * x[0]) + (-1.125 * x[1]))"
}
```

Não restaram ids, nomes de operação, camadas ou pesos simbólicos.

## Aplicação ao artefato completo

O comando real foi:

```bash
npm run generate:gemma4-flat-formulas -- \
  --family terminal_logit --dimension 0 \
  --batch 0 --sequence 5 \
  --max-characters 1000000000 \
  --output /tmp/gemma4-flat-0.json
```

O gerador de substituição direta encerrou antes de criar o output com
`Gemma4FlatFormulaLimitError`. A contagem sintática para uma única saída, sem o
compositor algébrico incremental posterior, foi:

```text
116062346796338308752997934553607335983728807481071013950327619967671906464343599050451016040528689516202661364868372029602719669765209175032636251894417389728416809703305487867853204236639021418631389099447899717221761037530525462058440739293505148 caracteres
```

Essa estimativa mantém cada redução como um único nó, mas conta substituições
sem simplificar entre camadas. Portanto ela mede a estratégia ingênua e não
prova o tamanho mínimo da fórmula matematicamente normalizada. O compositor
reverso incremental de `gemma4-e4b-reverse-algebraic-composition-2026-07-20.md`
substitui essa extrapolação por medições reais após cada simplificação.

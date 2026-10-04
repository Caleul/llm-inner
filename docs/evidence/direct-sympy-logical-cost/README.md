# Custos de substituição e simplificação

A comparação anterior de candidatos contava apenas o tamanho dos marcadores
temporários do compilador. Uma simplificação podia reduzir cópias de uma
dependência grande, mas ser descartada porque seu envelope curto tinha mais
caracteres. O compilador agora compara o custo da expressão totalmente
substituída e usa o tamanho residente exclusivamente para limites de alocação.

`StreamingLiterals` mantém strings matemáticas imutáveis durante a compilação.
Cada produtor continua passando pelo compilador numérico original, incluindo
`factor()` e `simplify()` e os contextos próprios de suas ramificações. A
emissão substitui todos os marcadores com parênteses; seu texto descomprimido
contém somente as variáveis fundamentais da entrada e operações admitidas.
O registro não é um artefato final nem um executor de inferência.

## Evidências locais

- `integration-tests.log`: build concluído e 12 integrações configuradas
  aprovadas, incluindo as 11 existentes; zero falhas ou skips.
- Cinco testes novos verificam a escolha por custo efetivo, a passagem do
  custo para a composição numérica, restauração de contexto e a emissão
  atômica com orçamento. A expressão pequena efetivamente salva é relida e
  compilada em C++: 30.722 comparações nativas, sem diferença de bits,
  incluindo os dois sinais de zero.
- A composição completa da dimensão 2 na posição zero foi comparada em
  60 casos com um forward CPU capturado novamente: zero divergências.
  O verificador reutiliza definições imutáveis apenas para validação.
- Com custo curto, a composição local projeta 19.807.890.757.704 caracteres.
  Com custo efetivo, projeta 15.345.975.626.240: redução de aproximadamente
  22,5%. Nenhuma dessas projeções é um arquivo final emitido.

## Limites da evidência

O arquivo completamente expandido do Llama ainda não foi emitido nem
validado. O tamanho permanece excessivo e exige continuar a investigação
de repetições e simplificações globais. Os casos com vários tokens testam
a posição zero; o adaptador ainda não compila o último token de comprimentos
variáveis. As outras dimensões não foram iniciadas.

Estados antigos mantêm a verificação estrita das identidades dos helpers
numéricos. A representação compartilhada exige um compilador novo e recusa
o callback de savepoints legado; não reutiliza seus estados silenciosamente.

## Reprodução

```sh
LLM_INNER_DIRECT_PYTHON=/private/tmp/llm-inner-pytorch/bin/python \
LLM_INNER_DIRECT_JSON_CHECKPOINT=docs/evidence/direct-sympy-test-checkpoint \
node --test dist/test/direct-sympy-strings.test.js

python helpers/direct_sympy_logical_cost_benchmark.py \
  docs/evidence/direct-sympy-test-checkpoint artifacts/logical-cost-comparison
```

O benchmark executa cada variante em um processo novo, registra os
produtores completos, tamanho residente, expansão lógica, tempo, RSS e
paridade, e compara custo curto, custo efetivo sequencial e custo efetivo
com blocos paralelos. JSONs de relatório contêm somente metadados;
as expressões são strings matemáticas.

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

## Comparação no Colab A100

`colab-terminal.json` confirma o encerramento com `COMPLETED`.
`colab-comparison.json` foi baixado pelo Access Broker sem modificar seus
bytes; `colab-derived-metrics.json` referencia seu SHA256 e calcula o RSS
agregado observado dos processos. O CAS e a referência ficaram na CPU.
`source-provenance.json` verifica os hashes do bundle congelado e a
equivalência das árvores sintáticas com o commit local: as únicas
diferenças são comentários e a remoção de um import não utilizado.

| Variante | Produtores completos | Tempo | Pico RSS do processo | Pico RSS agregado observado | Expansão lógica |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custo de marcadores | 25 | 9,76 s | 565 MB | 565 MB | 19,81 TB |
| Custo efetivo sequencial | 25 | 68,08 s | 951 MB | 951 MB | 15,35 TB |
| Custo efetivo paralelo | 25 | 70,07 s | 951 MB | 1.831 MB | 15,35 TB |

Cada variante passou em 60 casos de paridade de composição, com zero
divergências. A forma menor exige mais trabalho de simplificação do que
a escolha por marcadores. O paralelo ficou 2,91% mais lento e seu pico
agregado observado cresceu 92,53% em relação ao sequencial corrigido.
O RSS agregado é conservador: a soma pode contar páginas compartilhadas
por mais de um processo; é uma observação amostrada, não um pico contínuo.

O paralelo processou 18 blocos. As dimensões pequenas deste checkpoint
não produziram combinações em pares dentro dos folds de projeção; foram
zero merges nesse benchmark. O teste existente com blocos ímpares cobre
as combinações de continuação que preservam a ordem original. Essa
comparação não demonstra uma aceleração geral nem conclui o artefato.

`cuda-parity.json` registra a validação numérica auxiliar executada na
A100: 4.194.304 comparações, zero divergências. Esse teste cobre as
identidades modulares já existentes; não representa um forward completo
do modelo na GPU nem a paridade do artefato final.

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

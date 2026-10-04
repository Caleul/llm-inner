# Continuação compatível e paridade com zeros mistos

A compilação retomou o estado atual sem alterar nenhuma fonte numérica. A identidade de todas as fontes foi conferida, os artefatos anteriores tiveram seus hashes verificados e as provas já compatíveis foram preservadas. O lote acrescentou 64 tentativas e 31 regiões completas, terminando com 145 tentativas acumuladas e 108 regiões emitidas.

| Estado atual | Resultado |
| --- | ---: |
| Padrões abrangidos pelos artefatos | 259.300.864 |
| Padrões pendentes | 3.771.425.280 |
| Cobertura geométrica emitida | 6,43310546875% |
| Corpos distintos preservados | 25 |
| Comparações nativas dos 108 artefatos | 28.080 |
| Divergências observadas | 0 |

O estado retomável permanece em `artifacts/direct-sympy-input-partitions/frontier-final-state`. Este diretório preserva o snapshot atualizado, o mapa e os corpos efetivos como strings matemáticas. JSON é usado para estado e evidências, conforme a instrução mais recente sobre representação das expressões.

## Lacuna corrigida no verificador

O corpus anterior só acrescentava explicitamente os dois sinais de zero quando todas as coordenadas admitiam zero. Uma coordenada que admitia zero junto de outra estritamente não zero podia ser validada sem nenhum `-0` naquela posição. As regiões pendentes que cruzam esses eixos tornaram essa lacuna relevante.

O corpus agora acrescenta `+0` e `-0` em cada eixo permitido, junto das duas âncoras de fronteira das demais coordenadas. Inclui combinações de sinais quando todas podem ser zero. Para dimensões acima de dez, usa uma quantidade linear de casos de zero em vez de enumerar todas as combinações. Os cantos e casos aleatórios anteriores continuam presentes, com semente determinística.

O relatório registra a versão 2 do corpus e o SHA256 do verificador. Provas em cache só devem ser reaproveitadas se esses identificadores, a identidade numérica e o hash do artefato forem compatíveis. Nesta execução, todos os 108 artefatos foram novamente compilados em C++ e executados contra o checkpoint recarregado; nenhuma paridade anterior foi adotada como substituta desta validação.

Um teste nativo demonstra a falha original: com uma referência isolada que devolve X2, o artefato correto `X2` passa. O artefato incorreto `0.0 + X2` passa nos cantos anteriores, mas perde o sinal de `-0` nas novas entradas e é rejeitado com duas divergências esperadas. Esse controle é um teste do verificador, não uma prova do modelo Llama.

Os 108 artefatos atuais ainda não incluem os eixos de zero. Por isso foi também compilada uma região real do checkpoint: X1 em [-65.504, -320] e X2 em [-2^-14, 0]. O processo estrutural provou a saída constante nessa região antes da validação; `zero-axis.expr` preserva seu resultado de 11 caracteres. O corpus incluiu as duas novas entradas com X2 negativo zero ao lado das fronteiras não zero de X1, e passou em 1.030 comparações sem diferenças. Esse ensaio não foi somado à cobertura da árvore, pois sua região não constitui uma nova folha disjunta do estado salvo.

## Testes e conclusão parcial

Build concluído. As 24 integrações passaram sem falhas ou testes ignorados. Depois foi acrescentado o controle nativo do artefato incorreto ao conjunto do corpus; seus quatro testes passaram, e a integração atualizada foi executada isoladamente, também com sucesso. As outras 23 integrações foram filtradas nesse ensaio adicional; suas verificações de regressão já tinham passado com as mesmas fontes numéricas e o mesmo verificador.

O mapa global dos testes conserva os resultados anteriores e acrescenta esta validação. A mudança é apenas no verificador e no seu corpus, sem invalidar a identidade numérica da compilação.

A cobertura geométrica não representa execução exaustiva de todos os padrões; a paridade dos artefatos é amostral, bit a bit, sem tolerância. O escopo continua posição 0, dimensão 2, um token. A coordenada completa, o último token com comprimento variável, múltiplos tokens e o vetor inteiro permanecem pendentes. Limites de tempo e tamanho continuam solicitando subdivisão, sem descartar entradas ou declarar um resultado completo.

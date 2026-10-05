# Continuação remota do vetor do próximo token

O alvo continua sendo somente os quatro logits da última posição, com suas
contribuições causais da atenção. Matrizes de logits anteriores não são o produto.
A emissão final e sua paridade permanecem não demonstradas nesta continuação.

A primeira dimensão com seleção dos oito comprimentos incorporou 654 eventos
de estabilização de dependências, mas parou sem estabilizar as condições.
O relatório `prior-session-observation.json` transcreve a última observação ao
vivo: o runtime A100 foi encerrado antes da recuperação de seus arquivos finais.
Essa observação não é uma nova execução nem uma prova de artefato final.

Após verificar o inventário pelo Access Broker, A100 foi recusada e T4 retornou
indisponibilidade. Um novo runtime CPU do Colab foi criado. Nenhuma compilação
ou validação pesada foi transferida para a máquina local; localmente somente
edição, transporte e build TypeScript. Não há aceleração CUDA nesta nova sessão.

A seleção serial e paralela de condições agora rejeita candidatos inseguros
antes de consumir o orçamento de transformações. Compartilha a memoização da
prova de operações definidas e conserva a ordem dos candidatos e o contexto
lazy. `candidates` conta metadados examinados; `testedCandidates` conta as
transformações efetivamente tentadas. Limites esgotados continuam explícitos.
O teste inclui condições de entradas ausentes que dominam o ranking e verifica
que uma condição elegível ainda é processada, sem avaliar o ramo inativo.

O ensaio SymPy remoto compõe os operadores em pares, executando fatoração e
simplificação em cada substituição. A primeira coordenada de um token usa
22 produtores alcançáveis, 12 preparações e 11 composições em quatro níveis.
Os relatos separam paridade da expressão com primitivas, expansão elementar e
artefato final. O teto acumulado é 512 MiB; RAM de compilação é acompanhada
separadamente. Os tempos de uma única rodada pequena não comprovam speedup.

Os resultados foram relidos pelo Access Broker e preservados em `remote/`,
com SHA-256 em `remote-manifest.json`. Os sete testes de condições e os dois
smokes nativos passaram. A rodada inicial de 38 testes teve uma falha na nova
fixture: sua expressão não justificava uma promoção redutora. A fixture foi
corrigida para manter ramificações distintas, e os dois ensaios incorretos estão
preservados. Nenhum teste foi retirado ou omitido para obter aprovação.

A composição sequencial levou 1,787 s, e a de dois processos 1,697 s. Cada uma
comparou 12 valores da coordenada com a referência CPU arm64 2.12.1, sem
mismatches. A expansão elementar posterior atingiu o prazo de 120 s, sem emitir
artefato. Esse prazo não prova que a expressão seja impossível nem um tamanho
mínimo. Foi iniciado um ensaio que fecha e persiste cada produtor antes de
substituir o seguinte, para localizar o crescimento com evidência de avanço.

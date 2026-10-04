# Seleção explícita da fronteira de compilação

O compilador agora permite `--frontier-order coverage` para tentar primeiro a região pendente com maior quantidade de padrões de entrada. Os empates são determinísticos, e ambos os sinais de zero contam na cardinalidade. A seleção exclui regiões já completas ou divididas e não altera suas expressões ou provas. `--frontier-order lexical` conserva a ordem anterior, que permanece o padrão após a comparação real.

A opção altera a ordem das tentativas, não a semântica numérica. Cada região continua começando com um compilador próprio, lendo pesos conforme necessário, substituindo e simplificando antes da emissão. Uma retomada pode mudar a ordem sem invalidar artefatos compatíveis com a mesma identidade numérica. Mudanças nas fontes ainda exigem importar somente a geometria e recompilar as expressões.

## Comparação real

Os dois modos partiram da mesma geometria auditada e da mesma identidade de fontes. Ambos tiveram 24 tentativas e oito segundos por região, com os mesmos limites de caracteres e caminhos. Os processos e testes rodaram simultaneamente; os tempos são observações locais sob concorrência, não um benchmark isolado de CPU ou memória.

| Resultado | Por maior cobertura | Ordem anterior |
| --- | ---: | ---: |
| Regiões concluídas | 0 | 24 |
| Padrões abrangidos por artefatos emitidos | 0 | 180.575.744 |
| Workers encerrados pelo limite de tempo | 23 | 0 |
| Tempo do lote | 191,81 s | 39,86 s |

A outra tentativa no modo por cobertura excedeu o limite de caracteres. Visitar territórios maiores não demonstrou aceleração neste orçamento, por isso a mudança não substitui o padrão pela opção experimental. O resultado não demonstra que nenhum orçamento maior possa beneficiá-la. As regiões maiores ainda precisam de simplificações mais fortes e orçamento adequado, além de uma ordem de trabalho melhor.

## Validação e estados

Build concluído e 23 integrações passaram, sem falhas ou testes ignorados. Os testes de partição agora são seis; o novo teste verifica cardinalidade, zeros com sinais diferentes, empates determinísticos, exclusão de nós completos/divididos e ausência de mutação da árvore na seleção.

O código final foi recompilado em um estado novo, importando somente a geometria. Seu primeiro lote em ordem padrão emitiu as mesmas 24 expressões, byte a byte, e passou em 6.240 comparações nativas com o checkpoint recarregado, sem diferenças. Uma retomada com a opção por cobertura escolheu a região `1`, preservou todos os artefatos já completos e encerrou/recolheu o worker após um segundo. Não publicou uma saída parcial. A continuação posterior usa novamente a ordem padrão; seu resultado final e paridade são registrados separadamente.

Os snapshots experimentais preservam as identidades da versão medida. `current-frontier.json` e o mapa dos corpos emitidos representam o estado atual verificado. JSON neste diretório contém evidências e estado; as expressões são strings matemáticas conforme a instrução mais recente.

A continuação terminou com 81 tentativas acumuladas e 77 regiões completas: 248.476.160 padrões abrangidos pelos artefatos e 3.782.249.984 pendentes, correspondendo a 6,16455078125% de cobertura geométrica emitida. Os 15 corpos distintos estão preservados. A validação nativa dos 77 artefatos passou em 20.020 entradas, sem divergências. As 24 provas iniciais foram conservadas somente após conferir a identidade de fontes e os hashes; os outros artefatos foram executados na validação. Essa paridade é amostral, não uma execução exaustiva de todos os padrões abrangidos.

O estado retomável atual é `artifacts/direct-sympy-input-partitions/frontier-final-state`. `continuation-validation.json` conserva os números finais e a identidade numérica; o mapa global de testes mantém tanto o experimento inicial quanto a continuação.

A coordenada completa, a validação do último token com comprimento variável, múltiplos tokens e o vetor inteiro continuam pendentes. Os limites não autorizam descartar entradas nem declarar um artefato regional como modelo completo.

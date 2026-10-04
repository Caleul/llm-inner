# Descarte de contextos numericamente impossíveis

A continuação anterior terminou com 70 regiões completas e uma falha semântica: `Selected producer contradicts its path conditions`. A reprodução identificou um produtor selecionado com certificado Half e uma condição de magnitude incompatível com os limites numéricos já provados. Esse contexto é impossível; a falha abortava a compilação antes de descartá-lo.

A correção introduz uma exceção específica somente quando a interseção entre uma condição do caminho e um certificado numérico é vazia. O distribuidor descarta esse contexto e contabiliza a contradição. Outros erros continuam interrompendo o compilador. Nenhuma exceção genérica é convertida em corte de recursos ou em caminho descartado. Se todos os caminhos forem excluídos, a emissão recusa a saída e preserva o arquivo anterior.

O caso real usa X1 e X2 nos ranks Half -23.807/-15.872, posição 0, dimensão 2, um token. `reproduce.py` reproduz esse caso com a correção: uma contradição foi eliminada, três caminhos alcançáveis foram processados e a emissão parou no limite de tamanho. Isso corrige a falha original, mas não entrega a coordenada desse domínio maior. `region-run.json` registra o limite e o crescimento restante.

## Validação

Build concluído. As 23 integrações passaram, sem falhas ou testes ignorados. O conjunto de caminhos coerentes agora possui dez testes. O novo teste usa um certificado numérico real para excluir uma ramificação impossível, confere os resultados nos dois sinais de zero e nos extremos, mantém erros genéricos fatais e rejeita uma emissão sem caminhos. As validações nativas existentes desse conjunto continuam passando: 888.832 comparações de normalização e 30.722 de propagação numérica, sem diferenças.

Como o módulo de propagação mudou, somente a geometria auditada do estado anterior foi importada. Todas as expressões foram recompiladas com a identidade atual; nenhum resultado numérico foi reaproveitado. O lote terminou em 80 tentativas, 74 regiões completas e seis tentativas que exigiram subdivisão por orçamento. Uma delas excedeu 30 segundos e seu worker foi encerrado e recolhido pelo supervisor.

| Evidência atual | Resultado |
| --- | ---: |
| Padrões abrangidos pelos artefatos emitidos | 244.047.872 |
| Padrões ainda pendentes | 3.786.678.272 |
| Cobertura geométrica emitida | 6,0546875% |
| Arquivos distintos preservados | 13 |
| Comparações nativas dos 74 artefatos | 19.240 |
| Divergências observadas | 0 |

A paridade é amostral nos artefatos regionais, comparando resultados bit a bit com o checkpoint recarregado. A cobertura geométrica não significa que cada um dos 244 milhões de padrões tenha sido executado na validação.

Nas 66 regiões com a mesma geometria e artefatos anteriores, o total de caracteres caiu de 20.672.134 para 509.190; todas ficaram menores. A comparação inclui as versões de fonte anteriores e atual, sem isolar tempo ou atribuir cada redução a uma única mudança. `comparison.json` detalha cada região.

O estado retomável é `artifacts/direct-sympy-input-partitions/pruned-paths-state`. Este diretório preserva seu snapshot final, o mapa e os 13 corpos efetivos como strings matemáticas completamente substituídas. Os arquivos JSON armazenam estado e evidência, conforme a instrução mais recente de representação matemática em strings.

## Pendências

A coordenada completa sobre todo o domínio ainda não foi emitida. Continuam pendentes sua paridade completa, a saída do último token com comprimento variável, múltiplos tokens e o vetor inteiro. A ordem atual da fronteira visita regiões pequenas antes das maiores; o crescimento das regiões restantes também continua exigindo investigação. Nenhum ganho parcial é tratado como conclusão do modelo.

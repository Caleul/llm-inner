# Fatoração exata entre produtores fechados

A passagem nova reúne ocorrências repetidas de produtores cujos limites, grid numérico e tratamento de zero já estão comprovados no contexto atual. Ela representa essas folhas temporariamente por símbolos, aplica `factor()` e `simplify()`, comprova que todas as operações anteriores e propostas são exatas em F64 e restaura as expressões originais. A emissão não conserva esses símbolos ou intermediários.

Além do custo completamente expandido menor, a admissão exige menos ocorrências de produtores. Alterar apenas a escrita de coeficientes não é suficiente. Operações inexatas, divisões gerais, tipos desconhecidos e zeros com sinal sem prova continuam sendo barreiras. A estabilização obrigatória existente permanece em cada substituição.

## Falhas encontradas e corrigidas

A primeira tentativa admitia formas menores localmente que aumentavam a expressão após fechar operações numéricas posteriores. O primeiro crescimento observado começou na ativação: a composição passou de 3.248.163.585.496 para 5.387.015.906.776 caracteres lógicos. Uma forma fatorada equivalente do polinômio deixou de ser reconhecida pelo construtor especializado. O reconhecimento foi ampliado apenas para formas ordenadas equivalentes no intervalo Half certificado.

Essa correção restaurou a composição global, mas uma região ainda crescia de 15.298 para 19.008 caracteres. A regra de admissão passou então a exigir eliminação efetiva de ocorrências, conservando a forma anterior quando a proposta apenas troca coeficientes decimais por frações. As medições das tentativas rejeitadas estão preservadas; elas não representam os artefatos admitidos da versão final.

## Validação

- Build concluído e 22 integrações: 22 passaram, nenhuma falhou ou foi ignorada.
- Cinco testes novos cobrem fatoração dentro de conversões, estabilização, limites de exatidão, contexto isolado, reconhecimento da ativação e crescimento da composição completa com a passagem ligada/desligada.
- 92.166 comparações nativas sobre três expressões, incluindo ambos os zeros de entrada: nenhuma diferença.
- Antes da mudança, a continuação terminou com 63 regiões e 186.972.160 padrões cobertos; seus artefatos passaram em 16.380 comparações com a referência.
- Os resultados numéricos anteriores não foram reutilizados com os módulos novos. A importação conservou apenas a geometria auditada das divisões, e cada expressão foi recompilada.
- O primeiro lote atual tem oito regiões completas, três expressões distintas, 74.813 caracteres de corpos únicos e 176.639.488 padrões cobertos. Os oito arquivos são byte a byte iguais aos artefatos correspondentes anteriores e passaram em 2.080 comparações contra o checkpoint recarregado.

`frontier.json`, `artifact-map.json`, os arquivos `.expr`, `partition-parity.json` e `region-growth-comparison.json` registram esse lote. A execução seguinte usa o estado local `artifacts/direct-sympy-input-partitions/exact-factor-admitted-state`; este diretório de evidência conserva o lote fechado e validado, sem incorporar um log ainda em escrita.

## Limite do resultado

A mudança permite fatoração certificada de dependências repetidas, mas não reduziu a composição completa deste checkpoint: continuam 3.248.163.585.496 caracteres lógicos e 8.615.711.603 caracteres no primeiro corpo selecionado. Essas medidas não são um artefato final. O tempo registrado foi medido com outras verificações simultâneas e não comprova aceleração.

Não há coordenada de domínio completo emitida, paridade dessa coordenada, implementação do último token com comprimento variável ou vetor completo. O caso atual continua sendo posição 0, dimensão 2, um token. Expressões permanecem strings matemáticas conforme o pedido explícito mais recente; JSON registra estado e evidência.

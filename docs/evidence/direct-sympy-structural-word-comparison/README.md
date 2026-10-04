# Comparação estrutural durante a fatoração

O perfil de uma região emitida mostrou 3,66 s acumulados em `ast.dump`, dentro de 13,31 s da execução instrumentada. A fatoração imprimia subárvores inteiras repetidamente para comparar folhas. Esse perfil mede custo relativo com instrumentação; não é o tempo do benchmark.

A implementação substitui essas strings por assinaturas estruturais exatas, com uma tabela local por passagem. As constantes preservam sua representação numérica exata. Todo ancestral alterado pelo percurso recebe uma nova assinatura; não há reaproveitamento de uma chave calculada antes da alteração. As expressões finais continuam strings matemáticas completamente substituídas.

## Comparação executada

Seis compilações alternaram a comparação anterior por impressão e a comparação estrutural: três por modo, em ordem anterior/nova/nova/anterior/anterior/nova. Cada execução começou com um compilador novo no mesmo processo Python. Os caches das bibliotecas podem aquecer e outras validações estavam em execução; o resultado é uma comparação local, não uma estimativa geral.

| Mediana | Impressão de subárvores | Assinaturas estruturais |
| --- | ---: | ---: |
| Tempo | 6,520988 s | 5,641614 s |
| Produtores concluídos | 17 | 17 |
| Caminhos emitidos | 2 | 2 |
| Caracteres emitidos | 430.300 | 430.300 |

O tempo caiu 13,49% nesta região; a razão dos tempos é 1,156. Os seis arquivos são byte a byte iguais, SHA256 `2c5b7ffd0fe16faa4a070781d08692dd354c92cb34e02ee796e7260239e80e92`. O corpo foi preservado uma vez em `coordinate.expr`, após conferir os seis arquivos; `comparison.json` conserva suas medições e aponta o corpo durável. Não houve comparação isolada do pico de memória.

O caso é posição 0, dimensão 2, um token, X1 entre os ranks Half -16.181/-16.120 e X2 entre -27.775/-23.808. A função emitida passou em 1.028 comparações nativas contra o checkpoint recarregado, sem divergências. Isso comprova paridade amostral dessa região, não de todo o domínio.

## Testes e estados

Build concluído. As 22 integrações passaram sem falhas ou testes ignorados. O teste de fatoração acrescenta reescritas aninhadas com proibição de `ast.dump` e verifica a distinção entre zeros com sinais diferentes; as 278.568 comparações existentes continuam passando.

A continuação do commit de origem terminou em 72 tentativas, 66 regiões completas, 187.464.192 padrões cobertos e 3.843.261.952 pendentes. Seus arquivos passaram em 17.160 comparações sem diferenças. Os sete corpos distintos, o mapa de arquivos, a geometria e a paridade foram preservados como evidência da versão anterior.

Como o módulo mudou, esses resultados não foram importados como provas da versão nova. Somente a geometria foi reutilizada. O primeiro lote atual recompilou oito regiões, com 176.639.488 padrões cobertos, arquivos idênticos aos anteriores e 2.080 comparações sem diferenças. O estado atual é `artifacts/direct-sympy-input-partitions/structural-final-state`; `frontier.json` é o snapshot desse lote validado.

## O que falta

Não houve redução do tamanho da coordenada completa. Continuam pendentes sua emissão em todo o domínio, a validação do último token com comprimento variável, múltiplos tokens e o vetor inteiro. Não há previsão de término sustentada por esta medição.

O próximo ponto de simplificação a investigar é o limite correlacionado por componente da normalização. O adaptador atualmente usa um limite simétrico global, mesmo quando o domínio prova um sinal e uma componente dominante. Limites mais estreitos podem provar células de arredondamento constantes e eliminar dependências inteiras, sem reordenar o cálculo de referência. Essa melhoria ainda não foi implementada neste registro.

# Eliminação de dependências por limites correlacionados da normalização

A normalização mantinha apenas um limite simétrico global. Isso ignorava o sinal e a relação entre a magnitude da componente atual e as demais componentes. Uma região que prova uma componente dominante agora pode provar que sua saída arredondada é constante antes de construir a média e a inversa. O compilador incorpora essa constante e o peso e continua a substituição e simplificação. Não há lookup de respostas.

A prova usa a razão real `n*t²/(t²+B+n*epsilon)` somente para limitar a magnitude, com somas racionais exatas, endpoints arredondados para fora e uma margem conservadora para o cálculo F32 original. Não substitui o cálculo original por essa identidade algébrica. A admissão exige todos os operandos Half finitos e as precondições já existentes de largura e epsilon. Casos sem prova conservam o cálculo anterior. Os sinais dos zeros impedem eliminar uma componente quando os endpoints representam resultados distintos.

As expressões finais continuam strings matemáticas, conforme a instrução mais recente. JSON neste diretório registra medições e estado de compilação.

## Resultado regional

Seis compilações alternaram o limite anterior e o correlacionado no mesmo processo, três por modo. Cada compilação iniciou um modelo novo. O modo anterior foi reproduzido desativando apenas a nova prova, mantendo as demais otimizações. Os arquivos de cada modo reproduziram seus respectivos hashes em todas as execuções.

| Medida | Anterior | Correlacionado |
| --- | ---: | ---: |
| Mediana do tempo | 6,008009 s | 1,000858 s |
| Caracteres emitidos | 430.300 | 39.787 |
| Caminhos emitidos | 2 | 1 |
| Produtores visitados | 17 | 17 |

O tempo caiu 83,34% e o tamanho 90,75% nesta região. Não houve medição isolada do pico de memória. Isso não estima a aceleração do domínio completo.

`coordinate.expr` contém o artefato totalmente substituído dessa região, SHA256 `f185e8fc560b69916e9ee164c81ba7a1d51acb4a044aafd4e6143a955bfe43a3`. Sua paridade nativa contra o checkpoint recarregado passou em 1.028 entradas, sem divergências. O escopo é posição 0, dimensão 2, um token; X1 nos ranks Half -16.181/-16.120 e X2 em -27.775/-23.808. É paridade amostral regional.

## Validação e retomada

Build concluído e 23 integrações passaram sem falhas ou testes ignorados. O novo teste cobre 5.332.992 casos nativos de limites RMS, com zero violações, e 122.888 comparações da eliminação antecipada de média/inversa, com zero divergências. Inclui sinais, zeros, lacunas de magnitude e recusa de certificados fora das precondições.

Somente a geometria auditada do estado anterior foi importada. Os resultados numéricos foram recompilados com a nova identidade de fontes. O lote atual concluiu oito regiões, cobrindo 176.639.488 padrões de entrada e deixando 3.854.086.656 pendentes. Seus arquivos passaram em 2.080 comparações nativas, sem divergências. Os quatro corpos distintos estão preservados neste diretório, com mapa e snapshot do estado. O estado retomável está em `artifacts/direct-sympy-input-partitions/rms-components-state`.

## Limitação ainda presente

No domínio completo, a nova prova não eliminou componentes constantes. O tamanho lógico continua em 3.248.163.585.496 caracteres; o primeiro corpo selecionado continua em 8.615.711.603. O limite recusou a emissão, sem declarar um resultado completo nem descartar silenciosamente entradas.

Continuam pendentes o artefato de uma coordenada sobre todo o domínio, sua paridade completa, o último token com comprimento variável, múltiplos tokens e o vetor inteiro. O ganho regional não resolve esse crescimento global.

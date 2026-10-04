# Redução de dependências repetidas pela decomposição racional

A expressão numérica usada para fechar a raiz positiva observável em F32 passou da forma de produtos fatorados para frações parciais do mesmo interpolante racional 5/5. Os coeficientes são reproduzidos a partir de onze pontos de Lobatto, interpolação com 100 dígitos e raízes do denominador calculadas com SymPy a 70 dígitos. A acumulação segue uma ordem fixa; não se permite reassociá-la livremente.

A mudança reduz de oito para sete as ocorrências da expressão de entrada nesse cálculo. O corpo emitido caiu de 1.048 para 935 caracteres. A conversão e o ajuste numérico continuam expandidos em operações elementares, sem preservar uma chamada genérica de raiz ou arredondamento. A admissibilidade continua restrita a entradas F32 positivas finitas com resultado observado em F32; raízes F64 sem essa fronteira não recebem esta transformação.

O teste nativo compila a expressão efetivamente emitida e compara 25.167.601 casos, incluindo todas as mantissas normalizadas, ambas as paridades de expoente, todos os subnormais F32 e fronteiras dos demais expoentes. Não houve divergências nem empates nas fronteiras de arredondamento. Isso certifica esta transformação numérica, sem substituir a paridade do modelo.

## Artefatos do checkpoint

A identidade numérica mudou. O novo estado importou somente a geometria auditada das partições anteriores e recompilou todas as expressões. Nenhum resultado ou prova numérica anterior foi reutilizado. O estado retomável está em `artifacts/direct-sympy-input-partitions/partial-fractions-state`.

| Verificação | Resultado |
| --- | ---: |
| Tentativas nesta versão | 160 |
| Regiões emitidas | 142 |
| Corpos distintos preservados | 26 |
| Padrões de entrada abrangidos | 274.553.856 |
| Padrões pendentes | 3.756.172.288 |
| Cobertura geométrica | 6,8115234375% |
| Comparações nativas dos artefatos | 36.920 |
| Divergências observadas | 0 |

`artifact-map.json` relaciona os domínios aos 26 arquivos de expressões efetivas, preservados neste diretório sem intermediários de runtime. `partition-parity.json` registra a execução desses arquivos contra o forward do checkpoint recarregado, com o corpus versão 2 e o hash do verificador. A cobertura geométrica não significa validação exaustiva de cada padrão: a comparação de paridade dos artefatos é amostral, bit a bit e sem tolerância.

As 124 regiões comuns às duas versões tiveram sua soma de caracteres reduzida de 17.727.952 para 14.022.101, uma redução de 20,904%. `matched-region-sizes.json` conserva a comparação por região. Não foi feito um benchmark isolado de tempo ou memória; não se afirma aceleração de compilação a partir dessa redução.

Na medição global, o tamanho lógico caiu de 3.248.163.585.496 para 2.296.124.090.440 caracteres (29,31%). O primeiro corpo de caminho selecionado caiu de 8.615.711.603 para 5.983.548.191. O limite recusou a emissão global; esses números são diagnósticos de crescimento e **não são um artefato final**.

## Regressões e continuidade

Build concluído e 24 integrações passaram, sem falhas ou testes ignorados. O teste de persistência havia fixado a função interna em que o limite de tamanho deveria ocorrer. Com a forma menor, o mesmo limite passou a ocorrer na recomposição do trecho numérico. O teste agora verifica a expansão recusada em qualquer uma das duas fronteiras reais de admissão e mantém as verificações de dependências persistidas, retomada compatível, rejeição de dimensão diferente antes de mutação e ausência de conclusão falsa. O mapa global dos testes conserva as entradas anteriores e acrescenta esta execução.

O inventário Colab consultado pelo Access Broker não encontrou sessões ativas. Esta versão foi validada na CPU local; não se transfere para ela a evidência CUDA de uma versão anterior. `factor()` e `simplify()` continuam na CPU.

A prova atual continua na posição 0, dimensão 2, um token. A coordenada inteira, a saída para comprimento variável, múltiplos tokens, o vetor completo e a emissão Rust permanecem pendentes. A representação das expressões segue a instrução humana mais recente: strings matemáticas; JSON conserva estado e evidências. Nenhuma entrada foi omitida por limite de recursos.

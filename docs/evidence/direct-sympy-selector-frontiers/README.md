# Bifurcações de produtores concluídos — 2026-10-03

O compilador agora conserva uma visão limitada das condições de um produtor
já simplificado. Ao reutilizá-lo, reconhece a mesma decisão nos dois
operandos, em vez de tratar essas ocorrências como decisões separadas.

Por exemplo, para `P = Piecewise((A, c), (B, True))`, a composição
`P * (0.5 + P * 0.25)` pode propagar `c` e trabalhar isoladamente com
`A * (0.5 + A * 0.25)` e `B * (0.5 + B * 0.25)`. Mantém a ordem de cada
operação e passa as ramificações por `factor()` e `simplify()` até estabilizar.
Só admite a transformação quando o tamanho restaurado realmente diminui.
Decisões independentes, prioridades diferentes e chamadas sem prova de
pureza permanecem separadas.

Essas visões existem somente durante a compilação. As folhas são referências
a strings imutáveis, com um orçamento de 16 MiB para as visões e seus
metadados; não são copiadas para filas de processos. Todos os nomes privados
são restaurados antes de publicar a expressão. A retomada reconstrói as
visões de produtores validados e não expõe folhas cujas provas saíram do
cache. Identidades incompatíveis continuam sendo rejeitadas.

## Resultado efetivo e limites

A ativação emitida caiu de 9.218.105.754 para **8.230.451.759 caracteres**,
uma redução de aproximadamente 10,7%. Os outros 14 produtores do prefixo
mantiveram seus hashes anteriores. Os 15 produtores são idênticos entre
sequencial e paralelo. A verificação dos arquivos completos teve 300
comparações por execução, em 60 entradas, sem divergências.

| Mesma versão e orçamento de 9 GiB no Colab | Sequencial | Paralelo |
|---|---:|---:|
| Tempo observado | 280,081 s | 290,701 s |
| Pico de RSS agregado amostrado | 40.446.996.480 bytes | 40.464.420.864 bytes |
| Produtores completos | 15 | 15 |
| Coordenadas completas | 0 | 0 |

O paralelo foi aproximadamente 3,8% mais lento neste ensaio. RSS agregado
pode contar páginas compartilhadas novamente; não representa RAM física
exclusiva. A redução não demonstra aceleração do processo completo.

A composição da ativação com a projeção up passou a caber como expressão
aritmética intermediária, mas seu fechamento Half exigiria 38.189.296.054
caracteres, acima de 9.663.676.416. A versão anterior parava antes dessa
composição. Portanto, comparar diretamente seus tempos não isola o custo
da otimização: esta tentativa também executou uma etapa adicional que não
chegou a ser publicada como produtor completo.

A alteração seguinte, registrada em `../direct-sympy-compact-composition/`,
passa a fechar esse envelope antes de restaurar os operandos gigantescos.
Este ensaio do Colab é anterior à mudança de composição e não mede sua
economia. O fechamento ainda deve continuar até uma coordenada inteira,
antes das demais.

## Artefato preservado

Após as verificações, o Colab informou que a sessão não estava mais ativa.
A causa não foi determinada. Os relatórios foram recuperados antes disso.
A ativação foi reconstruída localmente por retomada de um estado macOS
compatível, com as mesmas 14 fontes semânticas, checkpoint e domínio.
Nenhum manifesto Linux foi reutilizado como estado macOS.

O arquivo completo está em:

`artifacts/direct-sympy-selector-frontiers-local/state/objects/06fd41cfc46c08e95f15c675426db6d994c4a147af3cbe52c7ef8e831501f63d.expr`

Seu SHA-256 foi recalculado a partir dos bytes do arquivo e coincide com os
dois arquivos verificados no Colab. A retomada parou após publicar esse
produtor, conservando um estado local com 14 dependências completas. A
amostragem local de memória cobriu parte da retomada e não deve ser
interpretada como o pico de toda a execução.

**Esse arquivo é a ativação completamente expandida, não a coordenada de
saída nem o modelo final.** A validação do prefixo se refere à posição zero;
entradas com múltiplos tokens não provam a saída do último token. As demais
coordenadas não foram iniciadas. JSON nesta pasta contém somente evidências.

Os 88 testes Python e as oito integrações SymPy passaram, incluindo 18.434
casos Half na composição com ativação e a retomada com decisões repetidas.
O build passou. O mapa de validação conserva o baseline da regressão geral
e distingue essa paridade parcial da paridade final ainda pendente.

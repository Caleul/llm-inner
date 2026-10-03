# Composição antes da materialização — 2026-10-03

A composição `R16(R32(activation * up))` agora substitui cada operando
já concluído por uma referência privada com as provas da sessão atual.
Após cada substituição, executa `factor()` e `simplify()` até estabilizar.
Fecha o envelope numérico, propaga os seletores compatíveis, mede o tamanho
restaurado e só então materializa os operandos. Todas as referências
privadas desaparecem antes de emitir a expressão; nenhuma foi adicionada
ao runtime final. A ordem aritmética e os arredondamentos são preservados.

O novo teste compara byte a byte essa composição com o fechamento feito
sobre a expressão literal completa. Verifica as substituições individuais,
a ausência de nomes privados, as provas Half e a recusa do resultado acima
do orçamento antes da restauração. Contextos vindos de outra sessão são
recusados. Templates condicionais devem ser compostos por ramo em seu
próprio contexto; o método não permite escondê-los em um contexto global.

A execução real nova, limitada a 2 GiB por expressão e 120 segundos, salvou
13 dependências em 26,252 segundos. Os 13 hashes coincidem com o prefixo
anterior. A próxima ativação, estimada em 8.230.451.759 caracteres, excede
o orçamento e não foi materializada. Esse ensaio para antes do produto:
**não mede a economia de memória nem o tempo da nova composição no Llama**.

A retomada da versão anterior foi testada e recusada por identidade
incompatível antes de carregar os objetos. O estado novo foi construído do
checkpoint, sem editar a identidade antiga. Os arquivos de ativação e os
resultados do Colab descritos em `../direct-sympy-selector-frontiers/`
pertencem à etapa anterior a esta alteração no método de composição.

Não há coordenada completa, paridade final ou avanço para outras dimensões.

A verificação dos 13 arquivos completos obteve 180 comparações em 60
entradas, com zero divergências contra o forward de referência. O escopo
continua sendo a posição zero e os intermediários do prefixo.

Validação: 89 testes Python passaram sem skips, oito integrações SymPy
passaram e o build terminou com sucesso. A regressão geral teve 599 testes:
581 passaram, três foram pulados e as mesmas 15 falhas já mapeadas
permaneceram. Não houve falhas novas; os arquivos ausentes dos exemplos
legados e do contrato de tarefas continuam pendentes no baseline.

## Medição posterior no Colab

A composição antes da materialização manteve os mesmos 15 arquivos e o mesmo
ponto de parada do ensaio anterior. Em uma nova sessão A100 com memória alta,
a execução sequencial caiu de 280,081 para 156,122 s e o RSS amostrado caiu
de 40.446.996.480 para 13.296.148.480 bytes. A execução paralela caiu de
290,701 para 165,739 s; seu RSS agregado foi 33.184.432.128 bytes. Os dois
modos tiveram 300 comparações em 60 entradas, sem divergências. A sessão,
as versões e os relatórios estão em `colab/`. São medições entre sessões,
com variabilidade do ambiente; não constituem um microbenchmark isolado.
O paralelo continua mais lento e usa mais RSS agregado neste prefixo.

A redução posterior da condição quadrática, em
`../direct-sympy-quadratic-guard/`, tem identidade e artefato próprios.

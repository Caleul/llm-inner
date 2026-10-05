# AGENTS.md

## Objetivo

Compilar Safetensors + configuração em uma função especializada que recebe
a entrada e retorna o vetor completo de valores para o próximo token.

O objetivo é eliminar o modelo enquanto estrutura de execução: incorporar
pesos, substituir intermediários e simplificar até que cada saída dependa
exclusivamente da entrada. O checkpoint não participa da execução final.

## Como compilamos

Parta de uma dimensão arbitrária da saída e percorra suas dependências:

substituir → fatorar/simplificar globalmente até estabilizar → propagar condições → próxima dependência

Incorpore os pesos como constantes e simplifique após cada substituição.
Elimine dependências comprovadamente irrelevantes antes de expandi-las.
Não expanda toda a arquitetura para somente depois tentar reduzi-la.

As variáveis das expressões são sempre Xn. Em cada etapa, Xn representa
um valor recebido da etapa anterior: substitua-o pela expressão que o
produz, incorpore seus pesos e aplique fatoração/simplificação antes de
continuar. Repita até restarem somente os Xn da entrada original. A
notação Xn não autoriza preservar valores intermediários no resultado final.

Não há formato obrigatório para representar as expressões. Escolha a
representação que permita substituir, fatorar, simplificar e propagar
condições com fidelidade: objetos simbólicos, texto matemático, JSON ou
outra estrutura adequada. Preserve entradas, constantes, operações,
tipos e bifurcações. A representação não deve ditar a arquitetura nem
substituir a entrega da expressão direta completa.

Estruturas compartilhadas podem evitar trabalho repetido durante a
compilação; não devem preservar intermediários no resultado final.

Um intermediário eliminado não deve voltar como dependência pendente.
Se seu valor aparece em vários pontos de uso, trate-os sob seus contextos
efetivos, reconheça decisões comuns e simplifique antes de materializar
cópias. Duplicação de representação não demonstra necessidade de novas
operações ou caminhos. Justifique qualquer crescimento pelo cálculo e
pelas decisões sobreviventes, não pelo tamanho da representação escolhida.

## Paralelismo e prioridade de execução

A prioridade é entregar a primeira coordenada completa e provar sua
paridade; depois avançar para o vetor inteiro. Infraestrutura, métricas
e testes de componentes não substituem esse resultado.

Paralelize dependências e blocos independentes durante a compilação.
Cada bloco mantém seu contexto de condições; combine resultados em ordem
determinística, preservando a ordem numérica original. Substitua e
simplifique até estabilizar em cada bloco e após cada combinação, sem
distribuir antecipadamente todas as combinações de bifurcações.

Componha os operadores em pares, em vez de substituir toda a arquitetura
antes de simplificar. Para uma sequência 1 → 2 → 3 → 4:

1. Incorpore os pesos e estabilize cada operador.
2. Em paralelo, substitua 1 dentro de 2 e 3 dentro de 4, formando 1|2 e 3|4.
3. Fatore, simplifique até estabilizar e propague as condições em cada bloco.
4. Substitua a saída de 1|2 na entrada de 3|4 e estabilize novamente.
5. Repita a combinação em pares até restar a expressão direta da coordenada.

As entradas provisórias de um bloco são interfaces de compilação, não
ativações de runtime. Elas desaparecem nas composições. Um bloco não pode
assumir propriedades dessas entradas sem uma prova da arquitetura; valide
as premissas e propague o contexto quando conectar seu bloco produtor.
A união em pares não autoriza reassociar somas ou mudar arredondamentos:
a ordem numérica de execução permanece 1 → 2 → 3 → 4.

Inventarie as dependências escalares distintas alcançáveis da coordenada.
Conte cada produtor uma vez nesse inventário, mesmo com vários pontos de
uso. Registre separadamente substituições, composições de blocos, passes
de fatoração/simplificação, decisões e tamanho expandido. Ocorrências
repetidas de uma fórmula não representam novos neurônios ou forwards.

Use as GPUs NVIDIA disponíveis no Colab, via Access Broker, para tarefas
numéricas compatíveis que contribuam diretamente para a compilação e
validação. Execute fatoração/simplificação na CPU quando o backend não
tiver suporte CUDA. Não apresente paralelismo de CPU ou um benchmark
isolado como aceleração da compilação na GPU.

Limite a concorrência pelo consumo efetivo de memória. Monitore RAM e
memória da GPU separadamente do teto acumulado de expressões e condições.
Toda aceleração deve manter paridade bit a bit, inclusive nas reduções.
Se um recurso externo estiver indisponível, avance nas etapas locais
independentes; não transforme manutenção de infraestrutura no objetivo.

## Simplificação e bifurcações

Neste projeto, simplificar significa aplicar regras matemáticas globalmente
à expressão alcançável após cada substituição, incluindo operações,
bifurcações e condições. Identifique fatores e subexpressões comuns em
diferentes partes da expressão, componha as expressões substituídas,
reúna termos semelhantes e reduza coeficientes. Reduções locais de
constantes, máscaras e condições não substituem essa análise global.

Use fatoração para realizar o processo contrário da expansão: reúna
produtos e somas em formas equivalentes menores, eliminando cópias
artificiais de expressões. Distribuir produtos é uma ferramenta para
possibilitar simplificações, não uma obrigação de manter a forma expandida.
Compare as formas equivalentes e conserve a que reduz a expressão,
sempre com prova de equivalência bit a bit.

Todas as expressões devem passar por esse processo após cada substituição,
até estabilizar; somente então continue para a próxima dependência.
Não monte a expressão inteira para simplificá-la depois.

Exemplo algébrico, aplicado somente onde dtype, arredondamentos e condições
do caminho garantirem equivalência bit a bit:

```text
Expressão atual: 14*X1 + 32*X2 + 54*X3
Substituição:    X1 = 132*X4 + 3*X5 + 4*X6
Após substituir:
14*(132*X4 + 3*X5 + 4*X6) + 32*X2 + 54*X3
Após distribuir e simplificar:
1848*X4 + 42*X5 + 56*X6 + 32*X2 + 54*X3
```

Substitua então X2, simplifique novamente e reúna os coeficientes dos
mesmos Xn; repita para X3 e para cada próxima dependência. A fatoração
também deve reconhecer, sob as mesmas precondições numéricas:

```text
3*(2*X1 + 5*X2) + 7*(2*X1 + 5*X2)
→ 10*(2*X1 + 5*X2)
```

Continue o percurso e introduza bifurcações nos casos em que arredondamentos,
conversões de tipo ou funções de ativação efetivamente exigirem decisões.
Não crie escolhas independentes para cada ocorrência da mesma decisão.
As bifurcações e suas condições também participam da simplificação, incluindo as que
explicitam arredondamentos. Não descarte fronteiras numéricas sem prova.
Distribuir produtos para simplificar algebricamente é diferente de
distribuir combinações de bifurcações: estas só são expandidas depois
que as simplificações estabilizarem.

Aplique propagação de constantes, fatoração, cancelamentos, denominadores
comuns, identidades matemáticas e simplificação de condições quando suas
precondições garantirem equivalência bit a bit.

Preserve dtype, ordem das operações e reduções, arredondamentos e zeros
com sinal. Igualdade sobre números reais não basta.

Simplifique até estabilizar antes de distribuir combinações e achatar
condições. Elimine condições constantes, redundantes ou impossíveis e
resultados equivalentes. Achate branches sempre que correto.

Operações aritméticas não introduzem decisões próprias apenas porque
seus operandos são condicionais. Propague as condições e reconheça
dependências comuns, sem assumir escolhas independentes.

Expanda conversões, arredondamentos, ativações, raiz e exponencial em
operações elementares aritméticas/bitwise e decisões explícitas.
Minimize as bifurcações sem perder fidelidade; não deixe essas operações
como abstrações pendentes no artefato final.

Inventarie decisões antes dos cortes e registre decisões introduzidas,
eliminadas, sobreviventes e duplicadas. Diferencie decisões de caminhos.
Investigue crescimento excessivo como possível falha de substituição,
propagação ou simplificação antes de apenas adicionar recursos.

Descubra a quantidade de pesos escalares, operações e decisões a partir
do checkpoint e da arquitetura. Registre essas contagens separadamente:
quantidade de parâmetros não é automaticamente a quantidade de operações
ou bifurcações. Não adote um orçamento baseado nesses números sem derivar
e conferir suas premissas no percurso da coordenada compilada.

## Invariantes

- Paridade exata, bit a bit, sem tolerância, com o forward de referência.
- Dimensões, layers, vocabulário e demais propriedades descobertos do
  checkpoint/config; Llama é validação, não definição do compilador.
- Compilação incremental/streaming, com leitura dos pesos necessários
  e persistência verificável que permita retomada.
- Limites de recursos não autorizam omitir caminhos ou reduzir
  silenciosamente o domínio de entrada.

O resultado final não deve conter executor, IR/DAG de runtime, execução
genérica por layers, cache de ativações, leitura do checkpoint, lookup
de respostas ou especialização para um prompt/comprimento específico.

Avaliadores e representações intermediárias são ferramentas de
compilação e validação, não a arquitetura de execução final.

## Direção atual e conclusão

Primeiro compile uma coordenada arbitrária até uma expressão direta
completamente substituída e simplificada. Valide com um e múltiplos tokens,
diferentes entradas e casos numéricos de fronteira.

Depois aplique o processo a todas as coordenadas. A emissão e validação
Rust serão derivadas das expressões compiladas, sem exigir um formato
intermediário específico.

Mantenha o mapa dos testes existentes e verifique regressões.

Não conclua apenas porque um arquivo foi emitido ou uma primitiva virou
código escalar. Conclua quando as expressões dependerem exclusivamente
da entrada, os pesos estiverem incorporados, as expansões estiverem
completas e a paridade exigida tiver sido demonstrada.

# Substituição por strings e SymPy

A instrução atual substitui a representação JSON de expressões por strings
matemáticas compatíveis com SymPy. `StringCompiler` recebe e devolve somente
strings; não converte essas expressões para os arrays do compilador anterior.
JSON continua permitido para configuração do checkpoint e metadados de testes,
nunca como expressão deste percurso. Os arquivos antigos são referências para
semântica e regressão. `compile:direct-json` foi substituído por
`reference:direct-json`; a rota nova é `compile:direct-string`.

```sh
npm run compile:direct-string -- PYTHON CHECKPOINT OUTPUT --dimension 2 --max-characters 1048576 --max-seconds 60
```

O Python selecionado precisa de `sympy==1.14.0`, Safetensors e PyTorch.
O ambiente de validação já contém essas dependências. O arquivo
`helpers/requirements-direct-sympy.txt` fixa a dependência CAS; nenhum pacote
global foi instalado nesta alteração.

O ciclo é: envolver a expressão em parênteses, substituir um identificador Xn
pela expressão entre parênteses, chamar `factor()`, chamar `simplify()`,
restaurar a forma fatorada, certificar a proposta e repetir até estabilizar. Só então o próximo Xn pode
ser substituído. A comparação de estabilidade usa a sintaxe analisada para
evitar ciclos provocados apenas por espaços. Não se confunde reutilização de
um produtor com um ciclo de dependências.

`Piecewise((expressão, condição), (expressão, True))` é a sintaxe de bifurcação.
Cada ramo passa pelo ciclo sob seu próprio contexto. O ramo seguinte recebe
a negação dos anteriores; intervalos são ajustados à grade dyádica, inclusive
em desigualdades estritas. Os contextos não são compartilhados com irmãos.
Bifurcações dentro de outra operação também são visitadas. Resultados de
caminhos diferentes não são eliminados apenas por terem o mesmo produtor.

SymPy não fornece prova de equivalência IEEE. Uma proposta é aceita somente
quando um certificado independente prova operações originais e propostas
exatas, com limites racionais, quantum representável, ausência de overflow e
preservação do sinal do zero. Valores desconhecidos, divisões não certificadas
e fronteiras numéricas continuam barreiras. `factor()` e `simplify()` são
chamados mesmo quando a proposta é rejeitada. Chamadas protegidas são átomos
temporários do CAS, restaurados para a string original antes de certificação
ou escrita; nomes de átomos não entram no arquivo. O parser usa a AST Python
com gramática restrita, sem `sympify`/`parse_expr` com avaliação de texto.

O exemplo `3*(2*X1+5*X2)+7*(2*X1+5*X2)` é reduzido para
`10*(2*X1+5*X2)` sob domínio inteiro pequeno certificado. Depois de `factor()`
e `simplify()`, outra chamada `factor()` recupera fatores numéricos que
`simplify()` pode redistribuir. Não se promete uma forma canônica
única. A documentação do SymPy esclarece que `simplify()` é heurístico:
https://docs.sympy.org/latest/tutorial/simplification.html

## Integração com o checkpoint e limites atuais

O adaptador Llama novo lê escalares necessários por `safe_open/get_slice`,
descobre geometria/configuração e monta expressões de trabalho em strings.
Cada substituição de operando F32 executa o ciclo, e cada produtor estabiliza
antes de entrar no consumidor. O checkpoint não é carregado inteiro.

A coordenada inicial é posição 0/dimensão solicitada. A probabilidade de sua
única chave causal é exatamente 1 em qualquer comprimento válido; portanto,
tokens posteriores não influenciam essa coordenada. Não se admite que essa
prova cubra a última posição com múltiplos tokens nem o vetor completo.

O modo `--reference-boundaries` preserva `R32`, `R16`, `sqrt` e `Silu16` para
registrar as fronteiras de referência. O percurso padrão já substitui as
conversões certificadas por operações elementares; fronteiras desconhecidas,
raiz e ativação ainda impedem sua conclusão. Isso impede sua
admissão como resultado final. Quando termina uma expressão de trabalho, o CLI grava somente
`OUTPUT.work.expr` e o log de dependências `OUTPUT.growth.tsv`, e termina com código 2 para marcar a
pendência. Não publica `OUTPUT` como função concluída. Os limites de caracteres
rejeitam a expressão inteira antes de alocar uma substituição acima do limite;
não omitem dependências ou caminhos.

Na validação histórica de referência, o arquivo de trabalho foi relido e sua coordenada comparada a 60 casos de uma
captura PyTorch nova, bit a bit, em comprimentos 1/2/3/4/8. A execução do teste
usa funções numéricas de referência, inclusive SiLU PyTorch; esse avaliador
é validação do trabalho intermediário, não runtime do artefato final. A string
tem 209.807 caracteres e houve 227 passagens CAS na execução registrada. Esses
números não podem ser comparados ao tamanho do JSON totalmente expandido:
as primitivas numéricas ainda não foram expandidas nesta nova rota.

O mapa em `direct-string-validation.json` preserva os testes anteriores,
registra os novos e separa a paridade intermediária da paridade final. Ainda
faltam migrar os kernels elementares e provas de faixa para strings, produzir
a primeira coordenada totalmente fechada e validá-la sem primitivas, e somente
então avançar para as demais posições e dimensões.

Validação desta rodada: build aprovado; 593 testes de regressão, 575 passes,
as mesmas 15 falhas conhecidas e 3 skips. Nenhum passe anterior perdido.
Os dois testes integrados novos passaram, incluindo 11 casos do motor Python
e a recaptura/releitura dos 60 resultados da coordenada. A rota antiga foi
preservada como referência, com o mapa anterior intacto.

## Conversões elementares e crescimento por produtor

O percurso padrão agora fecha conversões F64 → F32/F16 sobre fontes finitas
certificadas. As strings contêm reinterpretações `Bits64`/`Float64`, operações
inteiras `U64And`, `U64Or`, `U64Shr`, `U64Add` e decisões `Piecewise`.
Não contêm um nó de modo de arredondamento. Subnormais, empate, overflow e
ambos os sinais do zero têm tratamento explícito; braços desaparecem somente
quando o intervalo ou a grade da fonte demonstra que são inalcançáveis.
As constantes destas máscaras são inteiros exatos.

Na faixa normal certificada, a máscara e o incremento atuam diretamente na
palavra com sinal. Isso elimina uma terceira cópia do produtor dentro do
mesmo cálculo. Cópias em caminhos distintos permanecem. Produtos de dois
F16 finitos são exatos em F32; operações com zero ou ±1 preservam o tipo
quando comprovado. A operação com zero permanece se necessária ao sinal.
A conversão F32 intermediária de uma soma/subtração de dois F16 antes de
armazenar F16 também é dispensável, mas a regra não se estende a somas de
produtos. Lanes constantes de uma redução são calculadas com a semântica
IEEE original. Cada substituição executa novamente o ciclo SymPy.

O certificado de RMS usa a correlação entre o numerador e a soma positiva
dos quadrados para limitar a normalização, em vez de combinar intervalos
independentes que inventam overflow. Aplica-se somente a operandos F16
finitos, largura e epsilon dentro da faixa provada. Residual que possa
transbordar invalida essa prova. Ausência de quantum não autoriza assumir
uma grade F32 na multiplicação anterior à conversão.

O CAS protege as fronteiras funcionais depois de visitar seus argumentos
individualmente. Um cache exclusivo da compilação reaproveita uma prova
sob expressão e domínio idênticos; cada ocorrência ainda chama `factor()`
e `simplify()`. A chave inclui faixa, quantum e possibilidade de zero
negativo. Seu orçamento de caracteres é quatro vezes o limite de uma
expressão; isso não constitui limite absoluto de RAM. Nenhum alias ou cache
é emitido na string de runtime.

O CLI tem orçamento explícito de caracteres e tempo (60 segundos por padrão).
Ao excedê-lo, grava o crescimento e o último produtor completo em
`OUTPUT.prefix.work.expr`, termina com código 1 e não admite uma coordenada.
Esse prefixo é evidência para inspeção, não um estado reutilizável ou um
resultado parcial de inferência. Não há truncamento nem seleção de prompts.

Validação numérica: 201.743.408 comparações de strings efetivamente compiladas
em C++ com conversões nativas, mais 507.904 pares para as regras de tipos,
sem divergências. A cadeia salva do inverso RMS tem 2.137 caracteres e
passou 527.904 comparações nativas sem divergências; a raiz ainda é avaliada
nativamente nessa prova. O prefixo equivalente da primeira tentativa tinha
140.056 caracteres. Isso demonstra redução de um produtor, não velocidade
do modelo nem paridade da coordenada completa.

A execução limitada da coordenada e a regressão final estão registradas no
mapa e nos arquivos de evidência. Ainda falta fechar a primeira coordenada,
eliminar as primitivas restantes, provar sua paridade completa e então
avançar às demais. A validação de referência permanece separada.

Resultado desta ampliação: build aprovado; 594 testes, 576 passes, as mesmas
15 falhas anteriores, três skips e nenhum passe perdido na comparação por
nome. Os três testes integrados SymPy passaram. A projeção V salva também
passou 527.904 casos nativos, com raiz de referência, sem divergências.
Seu crescimento de 44.781 para 448.238 caracteres permanece registrado:
conversões elementares ainda repetem a expressão por necessidade de cálculo;
a correção de lanes vazias reduziu esse produtor de 896.738 caracteres.
Essa execução não produziu nem admitiu o artefato da coordenada completa.

## Identidades exatas de palavras

Depois de substituir uma conversão, o compilador elimina os pares tipados
Bits64/Float64 inversos, combina máscaras e simplifica deslocamentos, zeros,
idempotência e constantes unsigned de 64 bits. Cada alteração passa novamente
por factor/simplify; somente o ponto fixo entra no produtor seguinte.
Identificadores ou chamadas sem tipo comprovado não permitem cancelamento.
Isso não distribui caminhos nem reassocia aritmética de ponto flutuante.

A ampliação passou os três testes integrados, seis testes Python de conversão,
as comparações numéricas anteriores e 16.048 comparações das identidades sobre
palavras, incluindo padrões NaN. A projeção V ficou com 442.929 caracteres e
preservou os 527.904 resultados nativos. O inverso RMS permanece com 2.137
caracteres. A suíte ampla anterior permanece registrada como baseline; nesta
ampliação foram executados o build e os testes direcionados.

As execuções com 1 MiB/60 segundos e 4 MiB/120 segundos terminaram pelo tempo,
com quatro produtores completos. Logo, aumentar o orçamento de tamanho não
resolveu esta etapa. A medição de custo da compilação orientará o próximo
ajuste. A coordenada e o vetor final ainda não foram emitidos nem validados.

O perfil instrumentado identificou 36,2 segundos cumulativos em ast.dump,
dentro de 60,3 segundos observados. O novo passe comparava subárvores inteiras
até quando reduce_call devolvia o mesmo objeto. Essa comparação passou a ser
por identidade do nó; comparação estrutural de operandos só acontece quando
seus tipos e funções podem coincidir. As simplificações aceitas e chamadas
obrigatórias do CAS permanecem iguais. Esses números são de profiling,
não um benchmark de velocidade final do modelo.

Após corrigir a comparação de nós, a execução de 4 MiB/60 segundos manteve
quatro produtores completos. O prefixo da projeção V é byte a byte igual ao
que passou a prova nativa. A suíte numérica foi repetida após a correção.
Não se conclui ganho de avanço ou velocidade da coordenada inteira a partir
desta medição; o passe evita o trabalho estrutural redundante identificado.

## Assinaturas de prova sem impressão repetida

O cache de faixas e tipos das conversões usa assinaturas estruturais exatas,
em vez de serializar toda a subárvore por ast.dump em cada consulta. A chave
é internada por tuplas; colisões de hash não são usadas como prova de igualdade.
Floats são codificados com todos os bits IEEE, inclusive -0.0, e tipos de
constantes permanecem distintos. Nós mutados são invalidados antes/depois da
substituição dos filhos para não transportar a prova anterior à nova árvore.
Tudo pertence exclusivamente à compilação. A saída continua uma string direta
sem nomes das assinaturas, referências intermediárias ou executor.

A execução de 4 MiB/60 segundos concluiu cinco dependências, avançando até
context:0, onde a execução anterior havia parado em quatro. O prefixo de
442.951 caracteres passou 527.904 comparações nativas. O teste distingue a
projeção V da redução causal de uma chave: adicionar +0 ao valor Half -0 pode
alterar o sinal, portanto o forward da projeção não é usado como referência
de contexto sem reproduzir essa redução. Raiz permanece explícita nesta prova.

Sete testes Python das conversões passaram, incluindo invalidação após mutação,
tipos, nomes X1/X10 e ambos os sinais do zero. A suíte numérica anterior foi
repetida. O avanço de um produtor não estabelece o tempo da coordenada inteira;
o orçamento de tempo ainda terminou sem um artefato final admitido.

Regressão ampla após as assinaturas: 594 testes, 576 passes, as mesmas 15 falhas
conhecidas, três skips e nenhum passe anterior perdido, conferidos por nome.
O prefixo salvo possui apenas X1/X2 como entradas, sem conversões R16/R32
residuais; a raiz ainda impede sua admissão como compilação final.

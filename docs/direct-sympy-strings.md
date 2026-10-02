# Substituição por strings e SymPy

A instrução atual substitui a representação JSON de expressões por strings
matemáticas compatíveis com SymPy. `StringCompiler` recebe e devolve somente
strings; não converte essas expressões para os arrays do compilador anterior.
JSON continua permitido para configuração do checkpoint e metadados de testes,
nunca como expressão deste percurso. Os arquivos antigos são referências para
semântica e regressão. `compile:direct-json` foi substituído por
`reference:direct-json`; a rota nova é `compile:direct-string`.

```sh
npm run compile:direct-string -- PYTHON CHECKPOINT OUTPUT --dimension 2 --max-characters 1048576
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

A expressão de trabalho preserva `R32`, `R16`, `sqrt` e `Silu16` para registrar
as fronteiras ainda não migradas para operações elementares. Isso impede sua
admissão como resultado final. O CLI grava somente `OUTPUT.work.expr` e o log
de dependências `OUTPUT.growth.tsv`, e termina com código 2 para marcar a
pendência. Não publica `OUTPUT` como função concluída. Os limites de caracteres
rejeitam a expressão inteira antes de alocar uma substituição acima do limite;
não omitem dependências ou caminhos.

O arquivo de trabalho foi relido e sua coordenada comparada a 60 casos de uma
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

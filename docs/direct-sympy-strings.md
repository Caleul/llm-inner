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

## Reutilização da análise de regiões já convertidas

O passe de conversões registra a assinatura de um produtor estabilizado
somente quando sua string não contém mais chamadas R16/R32. Ocorrências
idênticas podem preservar literalmente a subárvore sem repetir a visita de
conversões. O domínio de entrada da sessão é uma cópia imutável. Fronteiras
pendentes não admitem esse reaproveitamento. O produtor inteiro ainda passa
pelo ciclo SymPy após a substituição; nenhuma expressão vira alias ou execução
por cache no arquivo emitido.

O teste compara o resultado com e sem o reaproveitamento e exige menos nós
visitados, com strings iguais; também verifica que conversões desconhecidas
permanecem pendentes. Os três testes integrados e oito testes Python passaram,
com as mesmas comparações numéricas nativas sem divergência. A regressão ampla
da rodada anterior permanece como baseline; esta mudança recebeu validação
direcionada e build.

Em 60 segundos foram estabilizados cinco produtores, sem avanço adicional
nesse orçamento. O prefixo é byte a byte igual ao contexto já validado em
527.904 casos nativos. O grafo de referência possui 19 produtores para esta
coordenada, mas essa contagem não inclui a conclusão dos kernels elementares.
A coordenada completa permanece pendente.

A execução de 4 MiB/120 segundos estabilizou sete produtores e chegou a
context:1. O passe reaproveitou oito regiões e visitou 211 nós novos. O prefixo
de 442.921 caracteres passou 527.904 comparações nativas sem divergência, com
os pesos da segunda linha da projeção V e a redução causal correspondente.
Isso não estabelece ganho isolado da otimização: o orçamento de tempo é
diferente do ensaio de 60 segundos. Ainda falta o residual, MLP, normalização
final e cabeça de saída, além da expansão das primitivas remanescentes.

## Composição certificada das duas conversões

O compilador reconhece F64 → F32 → F16 antes de expandir as duas conversões
separadamente. A composição exige fonte finita, sem overflow F32, e uma grade
que exclua subnormais F32 não nulos. Fontes fora dessa prova usam a rota anterior.
Isso não recorta o domínio de entrada nem aproxima os valores.

Na faixa normal F16, uma única operação sobre a palavra de origem conserva os
empates das duas conversões. Num midpoint F16, o significando F32 é par; a
célula fechada do primeiro arredondamento precisa ser incluída antes de escolher
o endpoint F16 par. O incremento depende do bit retido e é expresso com
U64Mul, U64Add, máscaras e deslocamentos. Nos subnormais F16, a grade fixa recebe
o valor F32 efetivamente arredondado. As condições usam preimagens exatas dos
limites F16 através do arredondamento F32, inclusive o empate no overflow.
Cada substituição continua passando por factor/simplify até estabilizar.

A nova construção passou 559.130 comparações nativas, incluindo vizinhanças dos
dois lados da célula F32 em cada midpoint normal F16, subnormais, overflow e
ambos os sinais do zero. A composição genérica também foi comparada em 559.086
casos. As demais provas numéricas permanecem no mapa. O novo contexto salvo
passou 527.904 casos nativos sem divergências.

A normalização inicial passou de 22.070 para 13.291 caracteres por dimensão.
context:1 passou de 442.921 para 267.341 caracteres. O ensaio de 4 MiB/120
segundos ainda completou sete produtores: redução de tamanho foi demonstrada,
mas avanço adicional e tempo da coordenada inteira não foram demonstrados.
Build aprovado; regressão ampla com 594 testes, 576 passes, as mesmas 15 falhas
conhecidas, três skips, nenhum passe perdido e nenhuma falha nova por nome.

O produtor salvo continua com raiz de referência. Não é a coordenada final.
A projeção também contém reduções com zeros e conversões aninhadas cuja
eliminação exige prova do sinal do zero; sua composição não deve ser feita
por identidades sobre números reais. O residual e as etapas seguintes continuam
pendentes, assim como a expansão das primitivas restantes e o vetor completo.

## Identidades aritméticas antes da expansão das conversões

Antes de fechar R32/R16, a sessão elimina conversões já comprovadamente
redundantes e multiplicação/divisão finitas por um positivo. Isso expõe pares
de conversões à composição certificada sem mudar a ordem das reduções.
Cada substituição aceita passa novamente por factor/simplify até estabilizar.

A soma com zero positivo só desaparece quando há prova de que o outro
operando não pode ser zero negativo. Uma soma finita F64 só produz zero
negativo se ambos os operandos forem zero negativo; um resultado não nulo
não pode desaparecer por underflow nessa soma, pois os operandos já pertencem
à grade F64. R32/R16 só propagam essa prova quando a grade da fonte exclui
underflow não nulo para zero. Por isso o zero inicial da redução causal pode
continuar necessário, mesmo após fechar a projeção V.

No mesmo orçamento de 4 MiB/120 segundos, context:1 passou de 267.341 para
160.219 caracteres (40,07% menos). Foram concluídos sete produtores; o ganho
de avanço e o tempo total da coordenada ainda não foram demonstrados. O
prefixo salvo passou 527.904 comparações nativas sem divergências. Ele mantém
sqrt e ainda não representa a coordenada completa.

Build aprovado, dez testes Python de conversões/identidades aprovados e
regressão ampla com 594 testes: 576 passes, as mesmas 15 falhas conhecidas e
três skips. A comparação por nome não encontrou passes perdidos ou falhas
novas. O mapa mantém os registros anteriores e acrescenta
arithmeticIdentityValidation com hashes das fontes, artefatos e resultados.
Residual, MLP e saída continuam pendentes; as demais coordenadas não avançaram.

## Preparação dos envelopes e admissão da gramática

O perfil cProfile do commit dbfcc40 registrou 656.309.128 chamadas em
120,365 segundos instrumentados. syntax acumulou 81,753 segundos, incluindo
33,176 segundos internos em compile/parse. Esses tempos incluem a sobrecarga
do instrumento e não estabelecem aceleração da compilação.

A admissão da gramática agora possui um cache limitado a 8 MiB de strings.
Cada chamada continua reparsando uma árvore nova; somente a validação dos
nós admitidos é reutilizada. Nenhuma árvore mutável ou prova numérica/de
contexto fica compartilhada por esse mecanismo. Os testes verificam mutação
isolada, zeros com sinal, rejeição de sintaxe inválida e validação após expulsão.

Chamadas protegidas idênticas recebem o mesmo átomo temporário para o SymPy,
em vez de variáveis independentes. O envelope Piecewise reabre somente sua
própria raiz após as provas dos ramos, sem reabrir as árvores descendentes.
As expressões emitidas preservam as cópias e não possuem átomos CASBoundary.
Essa alteração não autoriza fatoração IEEE sem a prova numérica independente.

Os 16 testes do motor passaram. Build aprovado e regressão ampla: 594 testes,
576 passes, as mesmas 15 falhas conhecidas e três skips; nenhum passe perdido
ou falha nova por nome. O prefixo context:1 continua byte a byte igual ao
anterior e passou novamente 527.904 comparações nativas sem divergências.

O orçamento de 4 MiB/120 segundos continua encerrando em sete produtores.
O ensaio de 8 MiB/300 segundos encontrou uma substituição acima do limite
antes de alocá-la, também com sete produtores concluídos. Ainda falta medir
o tamanho exato dessa tentativa e distinguir suas cópias por operação/caminho
no residual; aumentar recursos não foi tomado como solução. Não foi demonstrado
avanço adicional ou ganho no tempo total da coordenada. O prefixo mantém sqrt;
a coordenada completa e o vetor final continuam pendentes.

Um ensaio de amostragem de stacks com faulthandler terminou com exit 139;
a causa nativa permanece não demonstrada, e nenhum resultado desse ensaio
foi admitido. O perfil válido foi obtido depois com cProfile sem esse sampler.
Os registros estão em grammarAndEnvelopeValidation, preservando o histórico.

## Crescimento por substituição e grade subnormal exata

A tentativa que ultrapassava 8 MiB foi medida antes de alocá-la: um molde
de 427 caracteres substituiria cinco ocorrências por uma fonte de 1.923.552
caracteres, produzindo 9.618.149 caracteres. A conversão anterior já havia
substituído seis ocorrências de uma fonte de 320.519 caracteres. O diagnóstico
salva os dois operandos como strings matemáticas de trabalho e registra cada
tentativa em TSV, incluindo resultados admitidos e operações pendentes.
Esses arquivos não são estados de retomada verificados nem artefatos finais.

A soma de dois valores Half pertence à grade 2^-24. Abaixo do limite normal
Half, um valor nessa grade tem no máximo dez bits significativos, e portanto
o kernel normal, que retém onze, não altera seus bits. Isso permite eliminar
o kernel subnormal separado, inclusive preservando ambos os sinais do zero.
O mesmo argumento vale para a grade F32 2^-149, com até 23 bits abaixo do
limite normal e 24 bits retidos. Fontes entre pontos da grade continuam usando
o tratamento subnormal. A composição F64 → F32 → Half admite a mesma prova
na grade 2^-24 e conserva suas correções de arredondamento duplo.

Os testes nativos cobrem todos os 16.777.216 words subnormais F32 com sinal
(incluindo zeros), os 2.048 words Half correspondentes e todos os 67.108.866
valores com sinal da grade Half em [-2, 2] na composição. Nenhuma divergência.
As comparações anteriores de células, empates e 507.904 pares Half continuam
aprovadas. Onze testes Python de conversões e 17 do motor passaram; build
aprovado. A regressão ampla mantém 576 passes, 15 falhas conhecidas e três
skips entre 594 testes, sem passe perdido ou falha nova por nome.

O ensaio de 8 MiB/300 segundos concluiu oito produtores e salvou residual:0
com 3.847.215 caracteres. A comparação com os 9.618.149 caracteres é contra
a expansão antiga estimada, que havia sido rejeitada antes da alocação.
O residual emitido passou 527.904 comparações nativas contra uma redução
independente da projeção V, contexto, projeção O e soma residual, sem diferença
de bits. A referência mantém os quatro acumuladores e o zero inicial causal.

Ainda não há coordenada final: o segundo residual estava em andamento no
timeout, e faltam MLP, normalização final, saída e primitivas remanescentes.
O residual salvo contém sqrt. Os resultados estão em
subnormalGridAndExpansionValidation; o histórico permanece intacto.

## Retomada em produtores concluídos

O CLI aceita --savepoint-directory e --resume. Cada produtor concluído salva
sua expressão diretamente sobre Xn em um arquivo .expr identificado pelo
SHA-256. O manifesto contém apenas identidade, fronteira e provas numéricas;
não contém AST, DAG, ativações ou expressões JSON. O runtime final não lê esses
arquivos. Produtores interrompidos no meio de uma operação não são retomados.

A identidade exige o mesmo conteúdo de configuração/Safetensors, dimensão,
posição, domínios, política numérica, fontes do compilador, Python, SymPy e
plataforma. A leitura dos hashes do checkpoint usa blocos de 1 MiB. Objetos e
manifesto são sincronizados e publicados por rename atômico, seguido de fsync
do diretório. Um lock permite um único escritor; escritores fechados rejeitam
novas operações. A restauração valida todos os arquivos e a ordem da fronteira
antes de alterar o estado do compilador. Provas de dtype, faixa, grade e zero
com sinal são vinculadas novamente às expressões completas, no mesmo domínio.

Três testes verificam geração contínua/retomada byte a byte, corrupção,
checkpoint alterado, dimensão diferente, dois escritores, escritor fechado
e falha na publicação do manifesto. A expressão retomada passou 30.722 casos
Half nativos sem divergências. Build aprovado, quatro testes integrados
aprovados e regressão ampla com 595 testes: 577 passes, as mesmas 15 falhas
conhecidas e três skips, sem passe perdido ou falha nova por nome.

O ensaio real de 8 MiB/300 segundos salvou oito produtores. A retomada carregou
essa fronteira e concluiu residual:1, chegando a nove produtores. O residual
salvo tem 3.847.239 caracteres e passou 527.904 comparações nativas sem diferença
de bits. Os contadores de conversões/nós visitados da retomada correspondem
somente ao novo trabalho; os eventos de produtores preservam o histórico.

A proteção contra uso do escritor fechado foi acrescentada após o ensaio.
Foi realizada uma migração explícita e localizada do manifesto: o diff exato
contém apenas dois guards de lock fechado, todas as outras identidades foram
mantidas e os nove hashes de expressões permaneceram iguais. Isso não estabelece
compatibilidade automática com mudanças posteriores. Os fontes anteriores e
o registro da migração acompanham a evidência.

A fronteira completa de nove produtores foi arquivada em
docs/evidence/direct-sympy-frontier-nine, separada do artefato final. Ela só é
compatível com as fontes identificadas no manifesto. O registro está em
verifiedSavepointValidation. A normalização após a atenção, o MLP e a saída
continuam pendentes; o prefixo contém sqrt e não prova a coordenada completa.

## Quadrado de um produtor Half finito

O construtor de multiplicação reconhece operandos textualmente iguais apenas
quando há prova de que o operando é Half finito. Em vez de substituir a mesma
expressão duas vezes em R32(A*A), substitui uma vez em R32(A**2), passa por
factor/simplify até estabilizar e elimina R32 pela prova de produto exato.
O produto tem no máximo 22 bits significativos, cabe em F32 e seu menor valor
não nulo é 2^-48. O quadrado de qualquer zero tem sinal positivo.
A emissão Rust posterior deve baixar esse expoente inteiro dois para
multiplicação direta; esta sintaxe não autoriza um operador genérico de potência.

Não há aplicação desta regra a um valor F64 arbitrário, operando desconhecido,
outro expoente ou produto de operandos distintos. Os intervalos reutilizam o
certificado da multiplicação, incluindo a correlação do quadrado. Todas as
63.488 entradas Half finitas foram comparadas por bits em Python e C++ contra
o produto e seu armazenamento F32, sem divergências. Um teste do construtor
verifica uma substituição no caso certificado e duas no caso não certificado.

Esta mudança altera a identidade do compilador: os estados anteriores são
rejeitados. As expressões antigas permanecem arquivadas; a execução nova
reconstrói seus próprios produtores e salva somente fronteiras concluídas.

O ensaio novo de 8 MiB/180 segundos concluiu oito dependências e validou o
residual:0 em 527.904 comparações nativas sem divergência. Uma retomada de
180 segundos admitiu a substituição compacta do quadrado com 3.847.225
caracteres, contra 7.694.438 na substituição anterior com duas ocorrências,
mas parou durante a construção de residual:1. O quadrado não foi arquivado
como produtor concluído. O tempo de análise da string permanece um gargalo;
a redução medida não demonstra ganho de tempo para a coordenada completa.

Build, quatro testes integrados e as provas numéricas passaram. A regressão
ampla manteve 595 testes, 577 passes, 15 falhas conhecidas e três skips,
sem falha nova ou passe perdido por nome. O mapa anterior foi preservado e
finiteHalfSquareValidation registra fontes, evidências, estado e limitações.
A fronteira nova está em docs/evidence/direct-sympy-square-frontier.
Ainda faltam normalização posterior, MLP, saída e primitivas remanescentes;
não há artefato final nem paridade da coordenada completa nesta etapa.

## Evitar reprocessar um arredondamento comprovadamente redundante

O perfil interrompido de 45 segundos registrou 113 chamadas de syntax,
com 26,523 segundos acumulados nelas. O quadrado Half já tinha uma prova
de produto F32 exato antes da substituição, mas o construtor ainda criava
R32 e percorria toda a expressão outra vez para removê-lo. Agora substitui
diretamente no envelope de quadrado, preservando factor/simplify até o
ponto fixo. O restante da aritmética e dos certificados não foi alterado.

A fronteira de oito produtores foi migrada explicitamente para uma cópia:
o diff completo só altera a construção do cast redundante, todos os outros
campos de identidade foram comparados, e os oito objetos mantiveram seus
hashes. O registro e o fonte anterior acompanham a evidência. A restauração
normal permanece estrita; não existe compatibilidade automática entre fontes.

A execução de 8 MiB/300 segundos concluiu residual:1 e salvou nove produtores.
O residual passou 527.904 comparações nativas sem divergências. A execução
parou na simplificação obrigatória da soma dos quadrados, com estimativa
de 7.694.476 caracteres. Esta fronteira não contém a normalização posterior,
MLP, norma final ou saída; sqrt permanece nos produtores.

Build e quatro testes integrados passaram. A regressão preservou os 577
passes, as mesmas 15 falhas e três skips, sem regressão nova por nome. O mapa
redundantSquareCastValidation conserva fontes, evidências e limitações.
Ainda não foi medido ganho de tempo para uma coordenada completa. A fronteira
está em docs/evidence/direct-sympy-square-fast-frontier.

## Envelope de produtores estabilizados e certificados de empate F32

Um produtor concluído cujo resultado é uma chamada inteira pode ser
protegido durante o processamento de um novo envelope. Essas chamadas já
eram opacas à álgebra em cas_view. A proteção agora ocorre antes do parsing
completo, sob o mesmo domínio de entrada. Os símbolos são privados do CAS;
o texto integral é restaurado e os limites de tamanho continuam incidindo
sobre a expressão completa. Não são emitidos aliases de runtime.

Uma condição externa que refine qualquer domínio de entrada força a análise
completa do produtor sob cada ramo. Condições word/bitwise que mantêm todos
os domínios idênticos permitem proteger os produtores; cada braço ainda passa
por factor/simplify. Os testes cobrem mudança de domínio, branches externos,
zero com sinal, limites, nomes de funções e restauração das cópias literais.

Foram acrescentados três certificados para eliminar a leitura de paridade
redundante no kernel normal F32. Eles não se aplicam ao kernel F16:

- Soma de dois quadrados Half finitos: ao alinhar quadrados com expoentes
  distintos, a soma é 1 módulo 4. Um empate F32 exato só pode descartar um
  bit, cujo bit retido é par. Expoentes iguais produzem até 23 bits. Se a
  adição F64 arredonda, a menor parcela está longe de qualquer midpoint F32.
- Raiz de F32 não negativo: o quadrado de um midpoint F32 normal tem 49 ou
  50 bits significativos e não pode ser a entrada de 24 bits. A separação
  também impede o arredondamento F64 de atingir esse midpoint.
- Recíproco de F32 não nulo: o significando ímpar de um midpoint não pode
  dividir uma potência de dois. A precisão de 24 bits do denominador fornece
  separação suficiente do intervalo de arredondamento F64.

Os kernels subnormal/overflow mantêm seu tratamento anterior. A aplicação
depende da prova de dtype e finitude; somas Half gerais conservam a correção
de paridade. O controle negativo 1 + 3*2^-24 demonstra que remover a paridade
de uma soma geral produziria um resultado incorreto.

O próprio código gerado passou 503.856.640 pares de magnitudes Half,
quotientados pela simetria exata de sinal/quadrado e ordem da soma: zero
divergências, zero empates ímpares e 13.918.556 empates pares. O kernel de
raiz passou todos os 2.139.095.040 valores F32 não negativos e preservou -0.
O recíproco passou 4.219.469.826 casos com ambos os sinais, para denominadores
F32 de 2^-127 a 2^125. Nenhum caso divergiu ou atingiu um midpoint F32.

Silu16 só fornece um certificado Half quando recebe Half finito; assim seu
produto por outro Half é exato F32. O limite da ativação não é atribuído a
uma entrada F64 não certificada. A primitiva continua pendente de expansão
e essa prova de dtype não constitui sua implementação final.

A execução nova de 8 MiB concluiu dez produtores, incluindo post:inverse.
Os residuais passaram de cerca de 3,85 milhões para cerca de 972 mil
caracteres. post:inverse foi emitido com 3.887.805 caracteres e passou
527.904 comparações nativas contra as projeções, residuais e RMS posteriores
calculados independentemente. sqrt permanece: esta é uma prova de produtor,
não a paridade final da coordenada. A etapa seguinte estimou 29.158.260
caracteres para seis ocorrências do produto normalizado e foi rejeitada antes
da alocação. A retomada de 32 MiB preserva o mesmo domínio e semântica.

A retomada com 32 MiB validou a identidade dos dez produtores salvos e admitiu
uma expressão de 29.158.252 caracteres. Parou pelo orçamento de 300 segundos
no processamento numérico, sem concluir outro produtor. A fronteira continua
em dez; não existe expressão final da coordenada. O próximo trabalho é medir
o processamento da conversão Half admitida e reduzir apenas cópias cuja
redundância numérica seja demonstrada, mantendo os caminhos de cada decisão.

O mapa `completedEnvelopeAndF32TieValidation` preserva os resultados anteriores,
arquiva a fronteira e os hashes e compara os nomes dos testes: 595 testes,
577 aprovados, as mesmas 15 falhas e três ignorados; nenhuma nova falha ou
aprovação perdida. Os testes específicos passaram (19 de expressão, 14 de
conversões, três de estados e quatro de integração). O build passou.
A paridade da inversa é intermediária: sqrt/Silu ainda não foram eliminados,
a coordenada e o vetor final não estão concluídos, e a saída do último token
para múltiplos tokens ainda não foi validada pelo adaptador desta experiência.

## Envelope numérico no passe bitwise

A medição da tentativa de retomada identificou processamento repetido de
strings concluídas: 66,16 segundos acumulados em parsing e 44,06 em assinaturas
estruturais, contra menos de um segundo no estabilizador SymPy daquela amostra.
Os tempos acumulados se sobrepõem; não representam etapas somáveis.

O passe bitwise agora protege produtores com fechamento numérico certificado
no mesmo domínio. Expõe a chamada externa Float64 e conserva a prova de largura
do argumento inteiro para cancelar uma Bits64 adjacente sem reler seu conteúdo.
As substituições continuam passando por factor/simplify e as strings completas
são restauradas antes da saída. Novos guards que refinam os Xn reabrem o conteúdo;
regiões sem certificado, nomes reservados e limites de tamanho são verificados.

No kernel real de 29.158.252 caracteres, o passe anterior levou 104,70 segundos
e o novo 0,624 segundo. Ambos produziram exatamente os mesmos 29.158.223
caracteres, com hash SHA-256
`6154e6e3c042d3c8797801d531c90881505f066e925cf17f59e468c08c5733d3`.
É um ganho do passe bitwise isolado, não de uma coordenada ou modelo completos.

A retomada atual concluiu o produtor `model.layers.0.post:0`, com 29.158.223
caracteres, e salvou onze dependências com hashes compatíveis com o compilador.
A expressão efetivamente salva passou em 527.904 comparações nativas contra
a receita independente de operações F32/Half: zero divergências. O orçamento
de 300 segundos venceu durante a preparação da próxima dependência. Trata-se
da primeira componente de um intermediário pós-normalização, não de uma nova
coordenada do vetor de saída. A coordenada final continua em compilação.

O registro `wordEnvelopeValidation` inclui a medição reproduzível sobre a
expressão real, a fronteira de onze produtores e o certificado nativo. Os
mesmos nomes de 577 testes aprovados e 15 falhas anteriores foram preservados;
três testes continuam ignorados. Os quatro testes específicos foram executados
com Python e checkpoint explícitos e passaram; 19 testes de strings, 15 de
conversões e três de estados também passaram. O build passou.

## Persistência sem reanálise dos produtores já publicados

A publicação de uma nova fronteira relia e construía assinaturas para cada
string anterior, mesmo com conteúdo e contexto numérico inalterados. A
persistência agora reutiliza somente os registros publicados com sucesso,
associados à mesma instância de string imutável e à mesma ConversionSession.
Uma expressão substituída ou uma nova sessão exige recertificação. A retomada
continua validando identidade, manifesto e cada objeto antes de admitir os
registros. Uma falha de publicação não promove registros novos ao cache.
Esse cache existe exclusivamente durante a compilação.

Na fronteira real com onze produtores, a republicação anterior consumiu
55,154 segundos e a nova 0,0254 segundo. Os manifestos são idênticos byte a
byte. A migração explícita de uma cópia da fronteira alterou somente o hash
do helper de persistência: expressões, pesos, domínios, dtypes e helpers de
semântica numérica permaneceram iguais. A compatibilidade automática continua
estrita; nenhum estado antigo é admitido silenciosamente.

A retomada concluiu `model.layers.0.post:1`: agora são doze dependências.
A string efetivamente salva possui 29.158.367 caracteres e passou em 527.904
comparações nativas, sem divergência. A próxima substituição da composição
MLP exige 58.316.682 caracteres, acima dos 32 MiB permitidos; parou antes da
alocação. O próximo trabalho é inspecionar essa composição e seus metadados
antes de ampliar a expansão. A coordenada final, a remoção de sqrt/Silu e o
vetor completo continuam pendentes.

O mapa `publishedRecordCacheValidation` arquiva o benchmark reproduzível,
a migração, os doze produtores e o checkpoint minúsculo original de teste.
O checkpoint arquivado serve à reprodução das provas, não à execução gerada.
Build e quatro testes específicos passaram; os quatro testes de persistência
incluem troca de expressão/sessão, corrupção, falha de publicação e 30.722
casos nativos de retomada. A regressão manteve os mesmos nomes de 577 testes
aprovados, 15 falhas conhecidas e três ignorados.

## Simplificação antes da admissão de uma expansão grande

Uma substituição cuja estimativa excede o orçamento passa agora pelo SymPy
sobre um envelope de chamadas já certificadas, sem alocar todas as cópias.
Somente contextos de entrada idênticos permitem essa proteção; um guard novo
que restringe Xn exige reabrir a expressão. Ramos mortos podem desaparecer e
permitir a substituição. Se a expressão estabilizada ainda exceder o limite,
a compilação continua recusando a alocação, sem omitir caminhos.

Na soma real da projeção MLP, 136 caracteres virtuais estabilizaram em 132;
a expansão correspondente continuou com 58.316.678 caracteres. Na composição
posterior gate/up, 334 estabilizaram em 330, correspondendo a 116.633.422
caracteres. Portanto essas chamadas ao SymPy não demonstraram redução
algébrica suficiente. A retomada com 64 MiB levou 145,49 segundos e parou
antes de alocar a composição; a fronteira permanece em doze produtores.

A admissão gramatical de um composto também evita reler os descendentes já
validados. A proteção recusa sufixos que fundem identificadores ou transformam
um resultado escalar em função, preservando a rejeição da sintaxe inválida.

O teste nativo de fatoração verificou 126.972 casos: preservar os stores Half
manteve zero divergências; puxar a inversa comum através desses stores alterou
15.421 resultados. Isso exige prova adicional para esse tipo de fatoração.

O emissor C++ de validação tinha uma lacuna: frações matemáticas como `3/4`
podiam executar divisão inteira. Operações matemáticas simples agora convertem
os dois operandos para double; as operações unsigned permanecem em U64*.
Sete casos de literais, os certificados numéricos existentes e o produtor
post:1 (527.904 casos) passaram com o emissor corrigido. O mapa
`preallocationCASValidation` preserva os testes e registra essas distinções.

O próximo trabalho é sincronizar decisões repetidas por contexto de caminho
e concluir a simplificação numérica de cada projeção escalar antes de compor
ativação e up. A coordenada final e a remoção de sqrt/Silu continuam pendentes.
# Fronteiras numéricas das projeções gate/up

Cada projeção escalar de gate e up passa agora por `producer`: substituição,
`factor()`/`simplify()` até estabilizar e fechamento das conversões. Somente
depois disso a projeção é incorporada à ativação ou ao produto. Uma falha ao
fechar gate impede a construção de up; os dois resultados permanecem strings
completas, sem aliases no artefato. Os registros de produtores são ferramentas
de compilação e persistência, não caches de runtime.

O teste de ordem cobre também a interrupção antes da próxima dependência.
O teste nativo usa os pesos gate/up do checkpoint arquivado, todos os 63.488
valores Half finitos na primeira entrada e oito fronteiras na segunda:
1.015.808 comparações das duas projeções, sem divergências. As strings são
salvas e relidas antes da emissão C++. A redução de referência mantém suas
quatro lanes F32 e o armazenamento Half. Os arquivos
`docs/evidence/direct-sympy-projection-{gate,up}.work.expr` dependem dos valores
Half recebidos da normalização; não representam a coordenada final nem
substituem o fechamento das dependências anteriores. O arquivo `gated` mantém
Silu16 e a conversão final como trabalho pendente.

A paridade de referência da coordenada na posição zero continua passando em
60 casos, com primitivas ainda presentes. A mudança de identidade do adaptador
rejeita o estado antigo antes de carregar expressões ou modificar o modelo.
Nenhum estado antigo foi retomado ou migrado nesta validação.
# Reutilização das provas no fechamento numérico

`ConversionSession` mantém provas de intervalo, dtype e zero com sinal para
strings imutáveis cujo fechamento de conversões terminou. Quando a próxima
substituição contém uma dessas chamadas completas no mesmo domínio, o
fechamento numérico trabalha sobre um envelope temporário de chamadas opacas,
preservando essas provas. `factor()`/`simplify()` continuam sendo obrigatórios.
As expressões literais são restauradas antes de admitir o produtor; nenhum
identificador `CASNumericRegion` pode aparecer na string devolvida.

Essa reutilização existe apenas na compilação. Ela não acrescenta variáveis,
cache de valores nem executor ao resultado. Um compilador/sessão diferente
não herda as provas. Uma condição nova que restringe os Xn fundamentais exige
reabrir a expressão. A retomada continua verificando identidade e integridade
de todos os arquivos antes de inicializar a reutilização das provas fechadas.

O tamanho da string restaurada é calculado antes de alocar suas cópias. Se
ele exceder o orçamento, a compilação para sem publicar um produtor parcial.
O TSV `numeric-envelopes` registra tamanho inicial, tamanho virtual, tamanho
virtual fechado, tamanho restaurado estimado e quantidade de regiões. Ele
mede o fechamento numérico; não constitui um artefato final nem paridade.

Os testes verificam fechamento sobre envelopes pequenos, ausência de aliases,
recusa de identificadores reservados, isolamento entre sessões, reabertura
em condições novas e interrupção antes da alocação. A execução contínua e a
retomada produzem strings idênticas, com 30.722 comparações nativas sem
divergência na fronteira de arredondamento composta.
A execução real sobre as duas expressões completas de pós-normalização
recebeu 58.316.705 caracteres e fechou um envelope de 153 caracteres em
1.055 caracteres, visitando 19 nós novos e reutilizando duas regiões. A
restauração exigiria 349.900.367 caracteres: seis cópias do cálculo na
conversão tandem F32 → Half, contando o teste de faixa, o caminho subnormal
e o caminho normal. O orçamento de 64 MiB interrompeu antes da alocação.
O processo terminou em 130,50 segundos, incluindo a retomada estrita. Esse
tempo mede a tentativa limitada, não a compilação da coordenada nem um
benchmark comparável às tentativas anteriores que paravam em outra etapa.

A fronteira continua em 12 produtores, terminando em `post:1`. A migração
explícita alterou apenas a identidade dos três arquivos de compilador que
mudaram; todos os arquivos de expressão, checkpoint, domínio e provas
numéricas anteriores permanecem idênticos e foram verificados pela retomada
estrita. Os objetos estão no arquivo de evidências anterior, referenciado
pelo mapa de validação; o manifest novo registra seus mesmos digests.
# Decisões conhecidas no caminho

O compilador agora propaga fatos de verdade sobre condições matemáticas
repetidas, além dos intervalos dos Xn. As chaves estruturais preservam tipos e
bits dos literais, inclusive zero com sinal; não usam aproximação nem hashes
como prova de igualdade. Cada ramo recebe suas próprias hipóteses e as
negações dos guardas anteriores. A cache de simplificação inclui esses fatos:
uma prova no ramo verdadeiro não é reutilizada no ramo falso.

Se um `Piecewise` interno repete uma decisão já tomada, o compilador elimina
a decisão e conserva apenas o corpo alcançável. Também propaga `Not`, os
operandos de `And` verdadeiro e os operandos de `Or` falso. Isso não converte
`not(x < 0)` em `x >= 0`: comparações calculadas podem conter NaN. O ciclo
obrigatório de `factor()`/`simplify()` permanece em cada ramo e no envelope.
Regiões completas não são ocultadas sob um novo guarda externo, incluindo
guardas bitwise, pois a hipótese nova pode simplificar suas decisões internas.

O emissor C++ de validação passou a distinguir os seis operadores de
comparação e a emitir negação e operadores booleanos corretamente. A paridade
nativa foi verificada em 196.608 comparações (três expressões sobre todos os
65.536 padrões Half), incluindo NaNs, infinitos e zeros com sinal. O número
de `Piecewise` nas três expressões passou de 3/2/3 para 1/2/1. Os testes também
cobrem cache entre irmãos, reabertura de regiões e conjunções.

Nas expressões reais arquivadas, `pre:0` permaneceu byte a byte com 3.305
caracteres e um `Piecewise`; `v:0` permaneceu byte a byte com 40.427 caracteres
e 13 ocorrências. Não foram encontradas decisões já conhecidas no caminho
nesses dois casos. A mudança não demonstra redução da coordenada completa:
as ocorrências repetidas em operações irmãs precisam de sincronização após
estabilizar as simplificações, antes de distribuir suas combinações.

O registro `conditions.tsv` informa ações por guarda visitado; são eventos
de compilação, não contagem de caminhos do modelo. O helper de condições
participa da identidade dos estados salvos. As expressões arquivadas foram
verificadas pelos digests e usadas somente como entrada da análise; nenhum
estado antigo foi retomado ou migrado nesta etapa.


## Sincronização de seletores após fechamento numérico — 2026-10-03

Após o ciclo obrigatório de `factor()`/`simplify()` estabilizar e as conversões
numéricas serem fechadas, operações puras com seletores idênticos podem ser
reunidas em um único `Piecewise`. As condições e sua ordem precisam coincidir
estruturalmente, com ramo final verdadeiro. Cada corpo conserva a ordem de
suas operações. Seletores independentes não são distribuídos por esta etapa.
O candidato passa novamente pelo SymPy e só substitui o original quando fica
menor. Não existem aliases ou compartilhamento de intermediários no runtime.

A simplificação bitwise reúne duas máscaras constantes sobre a mesma palavra
unsigned. A prova do tipo, faixa e fechamento é transferida exclusivamente
para a raiz completa equivalente; os corpos selecionados não recebem uma
prova global que poderia depender de sua condição.

Os testes nativos compararam 131.072 casos, incluindo todos os padrões Half,
NaNs, infinitos e zeros com sinal, sem divergências. O build e os quatro testes
Node dirigidos passaram. A regressão manteve os mesmos resultados por nome:
595 testes, 577 aprovados, 15 falhas anteriores e três ignorados.

A análise dos objetos reais, verificados por digest, conservou exatamente
3.305 caracteres em `pre:0`, 40.427 em `v:0` e 971.823 em `residual:0`.
Os dois primeiros não tinham operações irmãs sincronizáveis; o residual
atingiu o limite de 256 levantamentos temporários e foi mantido integralmente.
A busca limita-se a um MiB e 262.144 nós temporários. Esses limites interrompem
uma tentativa de otimização, sem omitir caminhos ou restringir entradas.
Nenhum estado anterior foi retomado ou migrado; a identidade dos estados
inclui agora o novo helper de sincronização.

A fronteira completa permanece em 12 produtores e 29.158.367 caracteres em
`post:1`; esta etapa não concluiu a coordenada nem o vetor. A rota de referência
atual recebe `inputs_embeds` Half arbitrários e compara a posição zero,
coordenada dois. A incorporação dos embeddings a partir de IDs de tokens e a
saída do último token ainda precisam ser alinhadas com o contrato de entrada.
Não houve redução silenciosa do domínio enquanto essa definição está pendente.
As evidências estão em `docs/evidence/direct-sympy-synchronization-*` e o mapa
foi atualizado em `docs/direct-string-validation.json`.


## Seletores sob demanda e coordenada de trabalho — 2026-10-03

A sincronização inspeciona agora contextos puros sem distribuir cada operação
unária antecipadamente. Materializa corpos somente quando encontra operações
irmãs com as mesmas condições ordenadas. Seletores independentes não gastam o
orçamento de levantamentos. O residual real caiu de 971.823 para 890.918
caracteres (314 para 287 ocorrências Piecewise), com oito levantamentos e uma
sincronização. Em 253.952 comparações nativas do trecho original e reduzido,
não houve divergências. Os quatro testes dirigidos passaram novamente.

Uma compilação nova, sem reutilizar estados incompatíveis, chegou a 21
produtores simbólicos e emitiu a coordenada zero/dois com 3.814.474 caracteres.
A expressão arquivada passou nos 60 casos do corpus contra a referência.
Esses casos usam embeddings Half e incluem comprimentos variados, mas comparam
a posição zero; não demonstram a saída do último token.

Esse avanço não equivale ao fechamento numérico: o arquivo contém 2.235 R16,
7.074 R32, 6.762 sqrt e dez Silu16. A reescrita altera formas reconhecidas pelas
provas numéricas e conversões seguintes permanecem abertas. A próxima etapa
precisa transportar provas de dtype/faixa com seus contextos de ramo antes de
admitir esses produtores como completamente baixados. O processo terminou
com código dois, destinado a expressões de trabalho, e não admitiu artefato
final. Os objetos dos estados e a expressão emitida estão arquivados em
`docs/evidence/direct-sympy-lazy-synchronization-*`.

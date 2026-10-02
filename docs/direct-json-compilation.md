# Compilação direta em JSON

## Meta e estado

Meta autorizada em 2026-10-01: compilar Safetensors/config em expressões JSON
diretas, com pesos concretos, somente entradas fundamentais e decisões explícitas;
produzir o vetor completo do Llama diagnóstico e demonstrar paridade bit a bit.
Rust será uma etapa posterior. A meta continua aberta. Os geradores Rust antigos
foram interrompidos e não serão reiniciados para esta investigação.

O checkpoint usado em `/private/tmp/llm-inner-rust-validation` tem 44 parâmetros
F16 (88 bytes de payload), uma camada, largura 2, MLP de largura 1, vocabulário 4
e contexto 8. Essa geometria é evidência deste checkpoint, não contrato universal.
A entrada é uma matriz de embeddings F16 finitos, não apenas IDs do vocabulário.

## Formato implementado

Cada nó é um array `[operador, tipo_do_resultado, ...operandos]`:

```json
["if", "f64",
  ["lt", "bool", ["input", "f64", "X1"], ["constant", "f64", "0x0000000000000000"]],
  ["constant", "f64", "0x0000000000000000"],
  ["input", "f64", "X1"]
]
```

Este exemplo representa uma decisão de ReLU, não a ativação do Llama, que é SiLU.
As constantes armazenam bits exatos; o tipo define a interpretação desses bits.
`input` identifica uma variável fundamental; o header liga seu nome à posição,
coordenada e dtype da entrada, ou ao comprimento da matriz fornecida. Esse
comprimento é necessário para condições numéricas dependentes do comprimento
real, sem especializar a função para um prompt. Arrays aninhados definem precedência.

`reinterpret` apenas reinterpreta bits entre um inteiro e um float da mesma
largura. `convert` só muda largura inteira. Não existem nós `round`, `nearest-even`,
`sqrt`, `exp`, layers, pesos externos ou referências a ativações neste esquema.
O auditor também rejeita operações F32 com arredondamento implícito. Operações
temporárias ainda não baixadas não podem ser publicadas como expressão concluída.

Os módulos AST e o avaliador são ferramentas de compilação e teste. Não definem
um executor de modelo para o produto. A serialização final contém expressões
substituídas, e não referências ao grafo usado temporariamente pelo compilador.

## Conversões implementadas

F32→F16 e F64→F32 são expressões de máscaras, shifts, aritmética inteira,
comparações, decisões e reinterpretations. O empate para par usa:

```text
(mantissa + (meio - 1) + (quociente & 1)) >> deslocamento
```

Esse cálculo não introduz uma decisão de empate. As magnitudes da mantissa e
do viés provam ausência de overflow no inteiro escolhido. Os shifts dependentes
de expoente aparecem somente nos braços em que seu intervalo foi estabelecido.
Há decisões explícitas para normal/subnormal, underflow, overflow e não finitos;
um único `if` por conversão não é uma garantia. Provas de domínio do checkpoint
poderão remover classificações impossíveis.

Os testes cobrem os 63.488 padrões F16 finitos, sinais de zero, empates, transições,
overflow e amostras F32/F64. NaNs têm uma política explícita no lowerer; não foi
demonstrada paridade de payload NaN com PyTorch. O modelo exige prova de domínio
finito, e a validação dessas primitivas não prova o logit final.

## Simplificação e matemática

Implementado: constant folding tipado, fatos condicionais, limites inteiros,
contradições/implicações, eliminação de braços iguais quando a condição é total,
identidades bitwise e fatoração em aritmética inteira modular. A passagem repete
até ponto fixo estrutural, com limite explícito; não declara convergência silenciosa.
Divisão por zero e shifts indefinidos não podem desaparecer por multiplicação por
zero ou eliminação indevida de condições.

Próximos grupos: normalização racional com MDC/MMC, fatoração e cancelamento com
provas de denominador não nulo, normalização de comparações e identidades
trigonométricas quando o domínio/tipo comprovar equivalência. O repositório já
possui racionais exatos e normalização de preimages que devem ser reutilizados.
Nenhuma regra poderá atravessar uma fronteira de arredondamento sem prova.
Por exemplo, `x*a+x*b` pode divergir de `x*(a+b)` em F32, e valores de seno/cosseno
arredondados não satisfazem automaticamente a identidade real exata.

Todas as regras habilitadas devem convergir antes da expansão dos produtos de
condições e do achatamento. "Todas as simplificações possíveis" significa a
aplicação até ponto fixo do conjunto de regras comprovadas; não é uma promessa de
forma mínima global para qualquer expressão matemática.

## Streaming implementado

JSONL: um header, um registro escalar por posição/dimensão, e um terminador.
O writer recebe um iterador assíncrono, mantém uma unidade residente e usa buffer
de 64 KiB. Não renderiza a expressão inteira em uma string. A ordem canônica
comprova unicidade/cobertura sem guardar uma lista de todas as coordenadas.
Há limites de bytes, profundidade e ocorrências expandidas, exclusividade do
target e publicação por rename somente após emitir todas as coordenadas.

`finalParity:false` permanece explícito após a emissão. Um arquivo incompleto
continua sendo draft e não substitui um arquivo completo anterior. Retomada por
unidade e identidade/checksum do checkpoint ainda precisam ser conectadas ao
adaptador; o writer isolado não afirma fornecer retomada.

O auditor conta ocorrências expandidas, nós distintos por identidade de objeto,
decisões e referências à entrada. Nós distintos não equivalem ainda à contagem
de decisões originais do modelo: o adaptador deverá fornecer proveniência.

## Adaptador e fechamento numérico implementados

`direct-json-model.ts` descobre a geometria e os pesos da fonte e constrói uma
coordenada de saída de trás para frente. A leitura usa páginas limitadas; a
memoização conserva expressões do compilador, nunca ativações de runtime. O
adaptador preserva a ordem de redução CPU arm64, RoPE, máximos de attention,
fronteiras F16/F32, residual e MLP. Comprimentos 1, 2, 3, 4 e 8 são exercitados.
O denominador de três termos possui decisão explícita sobre o comprimento total,
pois a ordem de soma de PyTorch muda ao entrar no kernel vetorial.

Os nós temporários `pending-sqrt`, `pending-exp`, `pending-silu` e `widen`
existem apenas na construção. O lowerer produz exclusivamente operações
aritméticas F64, bitwise inteiras e decisões. A avaliação fechada rejeita esses
nós temporários e qualquer aritmética F32 implícita.

- Widening F16/F32 é expandido em campos de bits e normalização de subnormais.
- A raiz positiva normal F32 usa semente bitwise e quatro passos Newton F64,
  seguidos de máscara de arredondamento. Não há chamada de raiz nem decisão
  no caminho empregado pelo modelo. O kernel foi comparado nos 16.777.216
  padrões de mantissa em [1,4); testes do próprio JSON cobrem expoentes e bordas.
- A exponencial de attention usa o polinômio e a ordem numérica do backend
  fixado, com prova de intervalo [-0,34; 0]. A redução de faixa se torna constante.
- SiLU usa polinômio fatorado no intervalo comprovado pelo checkpoint, limitado
  a magnitude 0,1. A admissão verifica todos os F16 desse intervalo contra o
  perfil de referência: 23.758 pontos na faixa máxima. Escolhe grau 2 ou 4
  somente depois dessa certificação integral; não remove termos por tolerância. O perfil é uma fonte de
  certificação durante compilação; nenhum lookup permanece na expressão.
- F32 arredondado e mantido exatamente alargado para F64 usa uma única máscara
  de 29 bits. F16 composto com widening conserva as decisões de subnormal e
  overflow e os sinais de zero. Produtos de dois F16 dispensam F32 round porque
  têm no máximo 22 bits significativos. Identidades não atravessam arredondamentos.

A admissão atual é deliberadamente restrita: pesos F16 finitos, RMS epsilon
positivo normal, prova de finitude do checkpoint, recursos limitados e as faixas
certificadas dessas primitivas. Checkpoints fora dessas faixas são recusados;
não recebem aproximação silenciosa. A generalização das provas segue pendente.

`direct-json-precision.ts` cancela máscaras/arredondamentos repetidos somente
quando os bits descartados já são comprovadamente zero. As provas são objetos
do compilador e não metadados de execução. A regra foi validada em todos os
F16 finitos, mas não encontrou cancelamento adicional neste checkpoint.

## Simplificação por faixa e células de duplo arredondamento

`direct-json-range.ts` propaga intervalos conservadores nas expressões de
construção. Cada operação F32 recebe um passo de margem para fora, cobrindo
arredondamento do cálculo do endpoint; um divisor que atravessa zero não recebe
limite. Produtos quadráticos têm mínimo não negativo. RMS recebe o limite
correlacionado provado, evitando a estimativa incorreta por divisão independente.
SiLU recebe a magnitude máxima verificada em todos os pontos de seu certificado.

Esses fatos eliminam braços de overflow, subnormal e sinal apenas quando sua
inacessibilidade está demonstrada. A restauração geral de sinal F16 usa OR do
bit original, sem decisão; conserva inclusive underflow para zero negativo.

A composição F64→F32→F16 normal possui uma regra exata adicional. Não substituir
por F64→F16 direto: o primeiro arredondamento alarga as células de empate. Na
representação inteira F64, o viés é `2^41 - 2^28 - 1 + paridade*(2^29 + 1)`,
seguido da máscara de 42 bits. Essa regra foi verificada em 368.628 casos:
ambas as bordas de cada célula de empate normal F16, seus vizinhos F64 e ambos
os sinais. O lowerer a usa somente com prova de resultado F16 normal e finito.

A árvore prevista reduziu cerca de 33.677 vezes para posição 0/dimensão 2 e
9.079.351 vezes para posição 7/dimensão 2 em relação ao checkpoint anterior.
A medida continua enorme: nenhuma emissão foi iniciada por causa dessa redução.

## Redução certificada de SiLU

O polinômio do lowerer F64 agora elimina termos após comparar todos os F16
admitidos pelo limite do checkpoint com o perfil numérico. Grau 2 é
`x/2+x²/4`; grau 4 acrescenta `-x⁴/48`. Na faixa máxima de magnitude 0,1,
o grau 2 tem 44 contraexemplos fora da célula de underflow e é recusado. O
primeiro em magnitude é `-0,031005859375`; a regressão cobre essa fronteira.
O grau 4 coincide em toda a faixa certificada.

A célula `|x|<=2^-24` retorna zero com o sinal original: no backend, o produto
F32 nesse domínio é `x/2` e o arredondamento F16 resulta em zero com empate
para par. Arredondar diretamente o polinômio real daria um resultado diferente
em `+2^-24`, por isso essa célula é explícita. Não há lookup de logits/prompts.

O limite de gate foi estreitado por Cauchy com o orçamento L2 já admitido pela
prova de finitude: `||gate_row||2 * (1,25*sqrt(width)*max_gamma +
sqrt(width)*2^-23)`, acrescido das margens de redução e half. Isto substitui a
soma de máximos independentes por coordenada. No Llama diagnóstico, a faixa
é 0,02928494551503347: 20.224 valores F16 verificados, todos equivalentes ao
grau 2. Essa faixa é descoberta de pesos/geometria, não constante do adaptador.

O perfil foi revalidado contra `torch.nn.functional.silu` no PyTorch 2.12.1
CPU instalado, com um thread: 63.488 padrões F16 finitos, zero divergências.
O SHA-256 e os contadores da faixa máxima/checkpoint estão no relatório.

A fronteira artificial F32 do polinômio escolhido foi eliminada após essa prova
de resultado completo. A conversão direta F64→F16 foi validada independentemente
em todos os midpoints F16 e vizinhos F64, incluindo subnormais e ambos os sinais.
Ela não substitui a composição F64→F32→F16 em outras operações do modelo.

Essa redução diminuiu a expressão prevista cerca de 6,17 vezes adicionais nas
duas coordenadas medidas. O JSON final continua pendente.

## Combinação de consultas à mesma condição

`direct-json-cofactor.ts` começa com o ponto fixo das regras básicas. Para cada
condição existente repetida, constrói dois resultados temporários sob fatos
`condição=true` e `condição=false`, simplifica ambos e mede exatamente o JSON que
resultaria de uma única consulta externa. Mantém somente o candidato de menor
tamanho, quando menor que o original. Não altera ordem de soma/produto em um
caminho nem distribui antecipadamente todas as combinações de condições.

A promoção é recusada quando o cálculo da condição pode conter uma operação
inteira indefinida, uma primitiva pendente ou uma entrada que não era obrigatória
no percurso original. O conjunto de entradas obrigatórias usa interseção entre
braços de um `if`, preservando a leitura lazy. Limites de candidatos, nós e
visitas interrompem trabalhos além do orçamento sem publicar resultado parcial.

A passagem padrão testa até 256 condições e retém no máximo uma por chamada.
`reducingCandidates` conta alternativas menores; `accepted` é zero ou um para a
transformação final retida. Isso não prova saturação global nem forma mínima.
Os testes cobrem signed zero, NaN, condicionais booleanas fundamentais, braços
com entradas ausentes e divisões inteiras indefinidas.

Na primeira posição/dimensão 2, 27 candidatos foram considerados e nenhum trouxe
ganho. Na última, 168 candidatos foram considerados, 6 alternativas eram menores
e uma foi retida: aproximadamente 6,81e41 bytes previstos, 1,98 vezes menos.
Os nós físicos de compilação aumentaram para 15.927 e as decisões físicas para
453; a contagem expandida de decisões caiu. A busca da coordenada terminal
levou aproximadamente 15 segundos, e a validação dos 32 pares passou de cerca
de 15 segundos para 189 segundos. A redução de árvore não é evidência de
compilação mais rápida nesta fase. Esses contadores medem coisas
diferentes e não podem ser apresentados como bifurcações originais do modelo.

## Diagnóstico de duplicação e limite de conclusão

`direct-json-measure.ts` calcula exatamente ocorrências e bytes da árvore que
seria serializada, sem renderizá-la. Valores BigInt evitam overflow do contador.
Na medição anterior à fusão normal/subnormal, posição 0/dimensão 2 possuía 1.249 nós físicos,
28 decisões físicas, profundidade 444 e aproximadamente 1,12e28 bytes expandidos.
Posição 7/dimensão 2 possui 15.927 nós, 453 decisões, profundidade 661 e
aproximadamente 6,81e41 bytes após combinação de uma condição. Essas medidas incluem duplicação de operandos
nas expansões numéricas; não são tamanho de arquivo gerado nem número de
bifurcações originais do modelo. Nenhuma emissão dessas expressões foi iniciada; a tentativa real do pipeline foi recusada na admissão.

As expressões fechadas dos 32 pares posição/dimensão passaram em 864 resultados
comparados bit a bit com PyTorch. Isso comprova o corpus diagnóstico após o
fechamento; não comprova todas as entradas possíveis nem um artefato final
serializado. O avaliador memoizado e a árvore compartilhada são ferramentas de
teste/compilação, não a arquitetura final do produto. `finalParity` continua falso.

## Marcos restantes

1. Ampliar simplificação com faixas, sinal, denominadores e precisão: eliminar
   classificações inalcançáveis e repetições numéricas antes de distribuir condições.
2. Inventário com proveniência das decisões originais, separado das ocorrências
   expandidas de cada coordenada. Não usar parâmetros ou nós físicos como caminhos.
3. Generalizar o fechamento numérico fora das faixas atualmente certificadas e
   verificar formas/geometrias e certificados adicionais de modelos suportados.
4. Obter uma expressão serializável de dimensão arbitrária e validá-la após
   leitura do JSON, sem checkpoint ou primitivas pendentes.
5. Todas as dimensões/posições, savepoints e artefato final completo independente
   do checkpoint; paridade para múltiplas entradas e comprimentos.
6. Manter mapa de regressões e commits. Push depende de remote: nenhum está
   configurado. Rust fica para a etapa posterior solicitada pelo usuário.

O orçamento anterior ~1.072 decisões para oito tokens depende da hipótese de uma
decisão por fronteira numérica. Não é uma contagem verificada nem limite do JSON
achatado e deve ser substituído pelo inventário real do novo adaptador.

## Mapa de testes

Baseline preservado: `docs/direct-parallel-test-inventory.json` e
`docs/direct-parallel-validation.json`. Nenhum teste antigo foi removido.

Novos testes em `test/direct-json.test.ts`:

- Bits e signed zero na serialização; todos os F16 finitos.
- Fatos condicionais e convergência; limites e contradições inteiras.
- Fatoração modular, com contraexemplo à fatoração irrestrita em F32.
- Preservação de zero negativo e de erros inteiros definidos pelo contrato.
- Rejeição de tipos e conversões ocultas.
- F32→F16: cobertura exaustiva F16 e oráculo diádico independente.
- F64→F32: comparação com IEEE nas fronteiras e amostras.
- Auditoria de duplicação, inputs e fronteiras F32 residuais.
- JSONL completo, parsing fiel e preservação do target após falha/incompletude.

Validação histórica inicial: 44/44 testes aprovados, incluindo os 11 novos e
os testes existentes `direct-round-preimage` e `direct-flat-substitution`.

Validação atual: `test/direct-json.test.ts` tem 29 casos; também cobre widening,
raiz, exponencial, SiLU certificado, conversões compostas, medição de duplicação
e cancelamento de round com prova de precisão. `test/direct-json-model.test.ts`
cobre os 32 pares posição/dimensão e as 864 comparações antes e depois do
fechamento/simplificação. Este teste é pulado sem as variáveis
`LLM_INNER_DIRECT_PYTHON` e `LLM_INNER_DIRECT_JSON_CHECKPOINT`; a validação registrada
forneceu ambas. Os 26 testes focados anteriores passaram sem skips após considerar as condições repetidas
do vetor completo. A admissão agora tem 26 testes de núcleo e o pipeline possui um
teste integrado adicional de recusa antecipada, ambos aprovados nas execuções focadas. Limites de visitas acrescentados depois também passaram na suíte focada e
nas 864 comparações do vetor completo; a coordenada terminal manteve a redução. O corpus inclui máximos F16 com
ambos os sinais, magnitudes misturadas e fronteiras normal/subnormal.

Suíte completa atual: 531 testes, 513 aprovados, 15 falhas, 3 skips.
Todos os 479 testes aprovados no baseline continuam aprovados; as mesmas 15
falhas anteriores estão mapeadas em `docs/direct-json-validation.json`. A suíte
completa final e os contadores de simplificação são reconciliados nesse arquivo.
Nenhum teste antigo foi apagado ou relaxado. Arquivo final do modelo ainda pendente.

## Pipeline executável e admissão antes da emissão

`npm run compile:direct-json -- CHECKPOINT PYTHON OUTPUT.jsonl` conecta descoberta,
leitura paginada, substituição, fechamento numérico, simplificação de precisão,
ponto fixo das regras básicas, combinação de uma condição e emissão JSONL literal.
Os limites configuráveis incluem memória de pesos, dependências, candidatos,
nós físicos, ocorrências expandidas e bytes de saída. Uma coordenada preparada
não conta como coordenada emitida. O arquivo completo só é publicado após cobertura
canônica de todas as posições/dimensões; `finalParity` permanece falso até validação
do artefato publicado.

A admissão mede bytes, decisões, ocorrências e referências à entrada com BigInt,
percorrendo os nós distintos de compilação. Isso permite rejeitar duplicações
exponenciais sem percorrer a árvore expandida. A auditoria também verifica a
profundidade pelo caminho mais longo, mesmo quando uma subexpressão compartilhada
foi visitada inicialmente por um caminho curto. O status de falha registra a
coordenada rejeitada e suas contagens previstas; `predictedBytes` é o tamanho da
expressão, sem o envelope JSONL. O target completo anterior é preservado.

No checkpoint diagnóstico, a execução real da CLI preparou posição 0/dimensão 0
com 11.177.958.458.709.996.050.992.512.067 bytes previstos e
496.211.326.597.868.164.195.004.927 ocorrências. O writer recusou esse volume antes
da emissão da expressão: zero coordenadas emitidas, nenhum JSON final publicado.
O teste integrado reproduz essa recusa e verifica o status e a ausência do target.
Isso prova a ligação do pipeline e seu comportamento de recusa; não prova um
arquivo compilado do Llama.

Um experimento separado repetiu a combinação de condições na posição 7/dimensão 2,
com 1.024 candidatos e 50 mil nós permitidos. A segunda rodada reduziu a previsão
de 6,81e41 para 3,58e41 bytes, com 29.296 nós físicos e 910 decisões físicas;
levou aproximadamente 89 segundos e reportou 403 MB de RSS. A terceira levou
aproximadamente 363 segundos, sem redução: os 897 candidatos excederam o limite de
nós. Isso é esgotamento do orçamento, não prova de ponto fixo ou forma mínima.
Essas rodadas adicionais não foram ativadas no pipeline padrão nem validadas
novamente no corpus completo. A CLI mantém uma rodada de combinação.

O formato permanece literal, sem referências serializadas entre subexpressões.
A alternativa de referências exclusivamente no JSON de compilação depende da
resposta do usuário ao alinhamento de formato; não foi assumida como autorizada.

## Simplificação de máscaras e deslocamentos

As regras inteiras agora combinam máscaras AND/OR/XOR consecutivas, removem
uma máscara completa, unem campos mascarados da mesma fonte e compõem dois
deslocamentos constantes válidos na mesma direção. Uma máscara antes de SHR
só desaparece quando todos os bits observados pela máscara posterior já estavam
preservados. As regras mantêm o operando quando sua avaliação pode falhar;
não fundem um deslocamento inválido nem usam essas identidades para números reais.

O teste independente compara oito formas antes/depois para todos os 65.536
valores u16, mais 4.101 valores/fronteiras em cada largura u32/u64: 589.904
comparações. Inclui máscara necessária, shift inválido, entrada ausente e divisão
por zero. Não há alteração na ordem das operações de ponto flutuante.

No checkpoint, a primeira coordenada medida (posição 0/dimensão 2) passou de
1.249 para 1.207 nós e de profundidade 444 para 421. A terminal (posição 7/dimensão
2) passou de 15.927 para 15.556 nós e de profundidade 661 para 631. As 28 e 453
decisões físicas, respectivamente, permaneceram iguais. A previsão literal caiu
apenas cerca de 0,007%, para 1,1177e28 e 6,8084e41 bytes. Essa limpeza reduz a
estrutura de bits, mas não resolve a duplicação do artefato final.

A suíte completa após essas regras manteve os 479 passes do baseline e as
mesmas 15 falhas anteriores. As 864 comparações de vetores antes/depois do
fechamento passaram bit a bit. O mapa registra os nomes e o hash do log.

## Fusão F32→F16 com faixa atravessando zero

`lowerJsonFiniteF32ThenF16AsF64` estende a composição existente aos resultados
F16 subnormais. O chamador precisa provar F32 normal-ou-zero e uma faixa do
resultado F32 estritamente abaixo de 65.520 em magnitude. A fórmula inteira
fundida preserva as células de duplo arredondamento no braço normal. Somente o
braço subnormal mantém o arredondamento F32 antes da quantização fixa F16;
a restauração de sinal usa a fonte original, pois F32 preserva seu sinal.

A decisão compara a fonte bruta com 2^-14. Se uma fonte ligeiramente abaixo
for promovida a 2^-14 pelo primeiro arredondamento, o braço de quantum fixo
também produz esse mesmo valor. Não há intervalo perdido na fronteira. Zero
negativo e underflow assinado continuam explícitos. Fora da prova finita,
o lowerer conserva a conversão anterior, com suas classificações.

O oráculo independente passou em 380.932 entradas: bordas e vizinhos F64 das
células F32 em todos os midpoints F16, incluindo subnormais, ambos os sinais,
zeros e fronteira normal/subnormal. A suíte completa manteve as mesmas falhas
preexistentes e as 864 comparações do vetor passaram antes/depois do fechamento.

Com a combinação padrão de uma condição, posição 0/dimensão 2 passou de
1,1177e28 para 2,1077e26 bytes previstos, aproximadamente 53 vezes menos;
posição 7/dimensão 2 passou de 6,8084e41 para 3,0876e39, aproximadamente 220
vezes menos. Nós físicos: 2.586 e 16.562; decisões físicas: 71 e 453;
profundidades: 407 e 611. A primeira coordenada agora tem uma combinação
vantajosa, que duplica alguns nós físicos enquanto reduz a árvore expandida.
Esses números não são contagem das bifurcações originais do modelo.

A execução real da CLI confirmou 2,1077e26 bytes para posição 0/dimensão 0 e
recusou a expressão antes da emissão; continuam zero coordenadas publicadas e
`finalParity:false`. O arquivo final do Llama ainda não foi produzido.

## Certificado de exponencial para diferenças de scores F16

O adaptador registra a faixa de scores junto de cada exponencial, além da
faixa da diferença. Scores e máximo são F16 finitos, portanto múltiplos inteiros
de 2^-24. Com |score|<=0,17, a diferença é exata F32: seu índice inteiro tem
menos de 24 bits significativos. O certificado percorre a malha inteira entre
as extremidades F16 admitidas. Cada divergência de um polinômio menor é testada
contra TODOS os scores admitidos para verificar se algum par pode produzi-la.
Um único par alcançável rejeita o candidato. A busca nunca usa amostragem para
admitir um grau; o grau 7 permanece como fallback.

Para o checkpoint atual: 20.603 scores distintos, faixa
±0,03313671320996587, 1.111.041 pontos de diferenças na malha envolvente.
O grau 6 diverge do polinômio de referência em dois pontos da malha,
-0,034568071365356445 e -0,05368697643280029; nenhum par admitido os produz.
O grau 5 tem contraexemplo alcançável e não é admitido. Certificados imutáveis
são reconhecidos por identidade; um objeto forjado não habilita a redução.
Somente os coeficientes e operações menores entram na expressão emitida,
sem tabela de resultados ou referências a scores de runtime.

A expressão JSON do candidato foi comparada em 8.192 pares amostrados, como
validação do lowering; a admissão permanece exaustiva pela malha e pelos pares
possíveis. Um teste opcional com as variáveis do checkpoint captura `torch.exp`
F32 CPU e compara todo o polinômio de referência com o backend na malha inteira,
exigindo que toda diferença do candidato esteja na lista provada inalcançável.
Essa comparação é sobre `torch.exp`, sem presumir o dispatch interno do softmax;
a paridade do vetor completo continua uma verificação separada.

Na exploração anterior, a malha ligeiramente mais larga tinha 1.123.004 pontos:
o polinômio de referência coincidiu bit a bit com `torch.exp` em todos eles.
O teste integrado usa as extremidades F16 efetivamente admitidas e 1.111.041
pontos. Os perfis são somente diagnóstico, nunca lookup no artefato.

A coordenada terminal passou de 3,0876e39 para 1,6184e39 bytes previstos
(1,91 vezes menos), com 16.378 nós, 453 decisões físicas e profundidade 602.
A primeira mantém 2,1077e26 bytes: seu argumento exponencial é constante zero.
O JSON final permanece pendente; essa redução não torna sua emissão viável.

Após o certificado, a suíte completa passou em 511 testes, manteve as mesmas
15 falhas e 3 skips. Nenhum dos 479 passes anteriores foi perdido. O teste
PyTorch percorreu os 1.111.041 pontos e as 864 comparações de vetores também
passaram bit a bit. O JSON final continua não emitido.


## Limite correlacionado de score por cabeça

`direct-json-score-bound.ts` reutiliza o orçamento L2 da normalização finita e
lê cada coeficiente Q/K em páginas. A norma de Frobenius da projeção limita
cada cabeça por Cauchy. Para RoPE, cada par usa a matriz [[c,-s],[s,c]], cuja
norma é sqrt(c²+s²); os coeficientes F16 concretos de todas as posições admitidas
são verificados. Não se presume rotação real perfeita. O fator 1,001 cobre a
avaliação das normas; 1,125 cobre a redução F32 da projeção/dot e arredondamento
relativo F16; 1,01 e sqrt(headDim)*2^-23 cobrem os arredondamentos RoPE. Os
limites de finitude são checados em cada fronteira antes de aceitar a prova.

A geometria vem do checkpoint: heads, KV heads, headDim, largura, contexto,
escala, epsilon e pesos. GQA usa a cabeça KV correspondente. A prova retorna
indisponível para geometria não suportada, pesos não finitos ou orçamento de
coeficientes insuficiente; nesses casos o adaptador conserva o limite anterior.
O resultado é apenas metadado de compilação, sem valores de ativações.

No checkpoint diagnóstico, o limite caiu de 0,03313671320996587 para
0,0023031365207302873. Os 60 forwards PyTorch tiveram maior score absoluto
0,0007162094116210938, dentro da prova. O hook diagnóstico captura scores antes
da máscara e depois chama a implementação eager original sem alterá-la;
esses valores nunca são consumidos pelo compilador.

O certificado da exponencial agora admite grau 3: 12.655 scores F16 distintos,
77.249 pontos na malha, sete discrepâncias do candidato provadas inalcançáveis.
O teste live PyTorch percorreu toda essa malha, e as 864 comparações do vetor
completo passaram bit a bit. A suíte completa manteve os 479 passes do baseline,
as mesmas 15 falhas e 3 skips; os dois testes novos cobrem leitura incremental,
GQA, coeficientes RoPE, orçamento, geometria inválida e finitude.

A coordenada terminal passou de 1,6184e39 para 3,3286e38 bytes previstos
(aproximadamente 4,86 vezes menos), com 15.826 nós físicos, 453 decisões físicas
e profundidade 575. A primeira permanece em 2,1077e26 bytes. Esses volumes ainda
impedem a emissão literal; o JSON final do Llama continua pendente.

Depois da suíte completa, a admissão ganhou uma checagem explícita de finitude
na saída escalada do score. O caso de escala que faria overflow F16 é recusado;
os dois testes de limites e o teste live PyTorch passaram novamente (3/3).
O limite do checkpoint diagnóstico permaneceu igual.


## Raiz simplificada pela composição RMS F16

`direct-json-rms-certificate.ts` certifica o consumidor inteiro para largura
2 descoberta do checkpoint, não a raiz isolada. Três passos de Newton diferem
em oito resultados F32 na varredura dos 16.777.216 significandos normalizados
[1,4). A escala binária exata transporta esses casos para todos os expoentes
normais. Para o epsilon F32 9,999999974752427e-7, 208 variâncias estão no domínio.
Preimagens por pontos médios diádicos invertem exatamente o arredondamento da
variância; 224 células da soma contêm 121 pares de magnitudes F16 alcançáveis.
Os dois componentes da primeira saída F16 coincidem em todos esses pares.
Permutação dos componentes e simetria de sinais cobrem as demais entradas;
a multiplicação pelo gamma aprendido acontece depois dessa fronteira idêntica.

O certificado é autenticado por identidade e o compilador só o associa às
raízes desse consumidor RMS. Largura não suportada, orçamento insuficiente ou
contraexemplo mantêm quatro passos. A raiz independente continua usando quatro
passos, preservando seu resultado F32 exato. Os pares são evidência diagnóstica,
não uma tabela de execução. O lowering continua emitindo apenas operações
aritméticas e bitwise, sem raiz, arredondamento implícito ou certificado em runtime.

Outra identidade exata elimina o arredondamento da média RMS quando a largura
é potência de dois: a soma positiva F32 dos quadrados F16, normal ou zero,
dividida por essa largura permanece exatamente na malha F32. O limite de
largura já admitido evita underflow; a ordem de soma permanece a do backend.

O teste do JSON fechado compara todos os 968 componentes críticos com sinais.
A implementação real `LlamaRMSNorm` CPU foi verificada separadamente em 968
casos com todos os sinais e formatos de um/oito tokens, sem divergências.

A primeira coordenada medida (posição 0, dimensão 2) passa de
210765582749392127773866486 para 3013399545832912041575670 bytes previstos,
com 2.491 nós, 71 decisões físicas e profundidade 377. A terminal medida
(posição 7, dimensão 2) passa de 332863937830794166975190178224479757953 para
4758883707766393775804908405907217025 bytes, com 15.496 nós, 453 decisões físicas
e profundidade 545. A redução é aproximadamente 70 vezes em ambas.
Esses números são tamanhos da árvore literal após simplificação, não de um
arquivo emitido. O JSON final do Llama permanece pendente.


Após essa mudança, a suíte completa registrou 535 testes: 517 passes,
as mesmas 15 falhas anteriores e três skips. Todos os 479 passes do baseline
foram preservados. As 864 comparações dos vetores de referência e JSON fechado
passaram bit a bit. Um teste adicional da composição JSON nos pares críticos
foi acrescentado durante a execução; a suíte focada final passou em 5/5 testes.
A CLI preparou 1/32 coordenadas e escreveu zero: o orçamento recusou a árvore
literal antes da emissão da expressão, registrando `pending` e `finalParity=false`.


## Soma de operandos F16 sem arredondamento F32 intermediário

Para dois operandos finitos F16 a e b, R16(R32(a ± b)) = R16(a ± b),
incluindo zeros com sinal e overflow F16. A soma exata cabe em F64.
Se a diferença dos expoentes binários efetivos não supera 12, a soma tem
no máximo 24 bits significativos, incluindo carry, e já é exata em F32.
Se a diferença é pelo menos 13, o menor operando é no máximo
2^(e-12) - 2^(e-23), onde e é o expoente do maior. A menor distância do
maior a uma fronteira de sua célula F16 é 2^(e-12), na borda de uma binade.
A margem que sobra é maior que o erro máximo do arredondamento F32.
Nesse caso ambos os caminhos terminam no maior operando, com o sinal correto.
Operandos subnormais têm menos bits significativos e não invalidam a margem.

O lowering reconhece a composição apenas se os dois operandos são F16
exatamente alargados ou constantes exatamente representáveis em F16.
Substitui a conversão composta por arredondamento F16 explícito sobre a soma
F64, mantendo os operadores bitwise, decisões de magnitude e sinal necessários.
A regra não altera somas de produtos das projeções: esses produtos podem ter
22 bits significativos. Um teste demonstra um contraexemplo em uma fronteira
F16 ímpar e exige que o arredondamento F32 seja conservado nesse caso.

`helpers/certify_half_addition.rs` é diagnóstico: quantizadores independentes
F64 e F32→F16 compararam 1.007.713.280 somas/diferenças, sem divergências.
Isso percorre todos os pares não ordenados dos 31.744 valores de magnitude
F16 finitos; simetrias de sinais e permutação cobrem todos os pares assinados.
Os oito casos de operação entre zeros com sinal são verificados separadamente.
O teste do JSON fechado compara 507.904 casos sobre todos os operandos F16
finitos, incluindo gaps 12/13, cancelamento, signed zero e overflow.
Nenhum programa diagnóstico é usado na função gerada.

Após simplificação, a coordenada medida na posição 0/dimensão 2 tem
2092638568808685997560870 bytes previstos (31% menos), 2.391 nós físicos,
71 decisões físicas e profundidade 373. A posição 7/dimensão 2 tem
2753983529206566619594294091032880833 bytes (42% menos), 14.890 nós físicos,
453 decisões físicas e profundidade 539. Os volumes continuam inviabilizando
a emissão literal; o JSON final permanece pendente.


A suíte completa passou em 522 testes de um total de 540, manteve as mesmas
15 falhas e três skips, sem perder nenhum dos 479 passes do baseline.
As 864 comparações dos vetores completos de referência e JSON fechado passaram
bit a bit. A CLI confirmou a nova estimativa na primeira coordenada, preparou
1/32 e escreveu zero unidades, com status pendente e paridade final não verificada.

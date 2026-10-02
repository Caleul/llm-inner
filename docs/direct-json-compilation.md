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
- A raiz positiva normal F32 usa semente racional normalizada e dois passos Newton F64,
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


## Identidades de zero preservando o sinal e a origem do arredondamento

As reduções CPU terminam com somas de lanes vazias iguais a +0. Elas não podem
ser removidas de um operando que possa ser -0: -0 + +0 resulta em +0.
O lowering agora demonstra o sinal possível do zero nos produtores F32 antes
de remover esse tipo de soma. Uma soma F32 de operandos F32 finitos só pode
produzir -0 se ambos forem -0; a soma exata é um múltiplo de 2^-149, portanto
não há underflow de um resultado negativo não nulo para zero. Subtração F32
exclui -0 quando o primeiro operando já o exclui. Ifs precisam da prova nos dois
braços. Constantes, alargamento exato e raízes/exponenciais com domínios positivos
certificados também propagam a prova. Conversões F16 permanecem desconhecidas:
um número negativo não nulo pode arredondar para -0 nessa fronteira.

As identidades incondicionais x + -0 = x e x - +0 = x também são aplicadas
no domínio finito. As demais identidades com zero exigem a prova acima.
O produtor já arredondado é devolvido diretamente, preservando sua origem
numérica de compilação. Isso permite que a conversão F16 seguinte encontre
o cálculo anterior ao arredondamento F32 e aplique a fusão já certificada.
Nenhum metadado de prova ou referência intermediária é emitido em runtime.

Os testes percorrem todos os 63.488 valores F16 finitos nas quatro identidades
de soma/subtração com +0/-0, além de reduções com 18.764 pares finitos amostrados,
branches com sinais diferentes e underflow F16 negativo. Eles exigem tanto
paridade numérica quanto preservação da origem que permite a fusão.

A primeira coordenada medida (posição 0, dimensão 2) passa de
2092638568808685997560870 para 72659232914655964882628 bytes previstos:
aproximadamente 29 vezes menos, com 2.094 nós físicos, 71 decisões físicas
e profundidade 331. A terminal (posição 7, dimensão 2) passa de
2753983529206566619594294091032880833 para 57373289093730727904233766012325529
bytes, aproximadamente 48 vezes menos, com 13.657 nós físicos, 453 decisões
físicas e profundidade 493. Os tamanhos literais continuam inviáveis;
a emissão do JSON final do Llama permanece pendente.


A suíte completa registrou 543 testes: 525 passes, as mesmas 15 falhas
anteriores e três skips. Todos os 479 passes originais e os 522 passes da
suíte imediatamente anterior foram preservados. As 864 comparações dos
vetores de referência e JSON fechado passaram bit a bit. A CLI preparou
1/32 coordenadas e escreveu zero: recusou a árvore literal antes da emissão,
registrando status pendente e paridade final não verificada.


## Malha exata dos scores F16 e exponencial sem despacho minúsculo

Todos os valores F16 finitos são múltiplos inteiros de 2^-24. Quando a soma
dos limites absolutos de dois operandos F16 não ultrapassa um, sua soma ou
diferença tem no máximo 24 bits significativos nessa malha, com os extremos
±1 também exatos. O lowering elimina o arredondamento F32 nesse caso, após
verificar o tipo dos dois operandos e os limites provados. Um limite pequeno
sem a malha F16 não basta: produtos F32 de operandos F16 podem ter uma malha
mais fina e continuam exigindo o arredondamento.

O builder agora registra o limite correlacionado já certificado em cada score
F16. A união dos braços do máximo propaga o mesmo limite. Isso permite aplicar
a identidade às diferenças usadas pela exponencial, sem introduzir uma regra
específica para uma camada ou quantidade de cabeças. A prova é geométrica e
numérica; pesos, largura, cabeças e limites continuam descobertos do checkpoint.

O certificado autenticado da exponencial admite apenas diferenças de scores
F16 não positivas. Elas são zero ou têm magnitude pelo menos 2^-24. Seus
quadrados e tails são F32 normais ou zero, e o polinômio em ±0 produz exatamente
um. Portanto o despacho minúsculo da exponencial pode desaparecer nesse domínio.
A versão genérica conserva o if para argumentos F32 subnormais. Não há tabela
de resultados, novo operador de arredondamento ou primitiva exp no JSON fechado.

Os testes percorrem os 33.554.433 pontos da malha em [-1,1], verificam signed
zero separadamente e comparam 458.784 somas/diferenças JSON sobre todos os
operandos F16 admitidos em [-0,5;0,5]. Domínios maiores e operandos fora da malha
mantêm o arredondamento, com contraexemplos explícitos. Outro teste exige zero
decisões no polinômio certificado, uma decisão na versão genérica e paridade
com seu kernel numérico em toda a malha do certificado de teste.

A coordenada terminal medida (posição 7, dimensão 2) passa de
57373289093730727904233766012325529 para 27697449910356226706169067089727129
bytes previstos: redução de aproximadamente 52%. Possui 13.465 nós físicos,
437 decisões físicas (antes 453), profundidade 485 e
44396514754408464210009611003 decisões expandidas. A primeira coordenada mantém
72659232914655964882628 bytes, pois seu argumento exponencial é constante.
Esses números continuam sendo previsões da árvore literal, não arquivos finais.
A emissão do JSON completo do Llama permanece pendente.


Após as duas mudanças, a suíte completa registrou 547 testes: 529 passes,
as mesmas 15 falhas anteriores e três skips. Todos os 479 passes originais
e os 525 passes da suíte imediatamente anterior foram preservados. As 864
comparações dos vetores de referência e JSON fechado passaram bit a bit.
A comparação live da malha de exponencial com PyTorch também passou.
A CLI preparou 1/32 coordenadas e escreveu zero unidades, recusando a primeira
árvore por tamanho antes da emissão e registrando paridade final não verificada.


## Promoções de condições em rodadas com término explícito

A combinação de condições não termina mais automaticamente após a primeira
promoção. Cada rodada reaplica as regras básicas até seu ponto fixo, procura
uma promoção estritamente menor em bytes literais e continua a partir dela.
A ordem das operações em cada caminho permanece intacta. Não há produto
cartesiano antecipado de condições nem geração de novos valores de referência.

O limite padrão é de oito rodadas, com 256 candidatos por rodada e o limite
já existente de 100.000 nós físicos. `--max-condition-rounds` permite controlar
o primeiro limite; uma rodada reproduz a política anterior. Contadores são
cumulativos. `converged` só fica verdadeiro se a última busca completa não
achar promoção permitida que reduza bytes. `stopReason` distingue `fixed-point`,
`round-budget`, `candidate-budget` e `resource-budget`. Esse ponto fixo vale
somente para esta transformação, não demonstra que toda simplificação
matemática possível foi realizada. Uma falha de orçamento em rodada posterior
preserva a última expressão já admitida e informa que a busca ficou incompleta.

O teste com dois grupos de decisões demonstra que a segunda promoção reduz
mais que a primeira e mantém os quatro resultados possíveis. Outro teste
exige que limites de rodadas e candidatos nunca sejam reportados como convergência.
Os testes anteriores de lazy inputs e operações inteiras indefinidas continuam
aplicáveis em cada rodada.

Na medição inicial, a primeira coordenada (posição 0/dimensão 2) aceitou uma
promoção e atingiu o ponto fixo desta regra na segunda rodada, em cerca de
1,17 segundos. Manteve 72659232914655964882628 bytes previstos. A terminal
(posição 7/dimensão 2) aceitou duas promoções em três rodadas e caiu de
27697449910356226706169067089727129 para 14559843742299065345314960831221058
bytes, cerca de 47% menos. A terceira rodada parou no limite de candidatos:
672 candidatos foram examinados no total e a convergência não foi demonstrada.
A busca terminal levou aproximadamente 140,6 segundos, com 24.894 nós físicos,
878 decisões físicas e 23336047748076654126818621733 decisões expandidas.
O crescimento dos nós/decisões físicos acompanha uma redução das ocorrências
na árvore literal. O custo adicional de compilação é real e a opção de uma
rodada preserva o caminho anterior. O JSON final permanece pendente.

A suíte serial terminou e foi reconciliada: 549 testes, 531 passando,
15 falhas anteriores e três ignorados. Nenhum dos passes anteriores foi perdido.
As 864 comparações dos vetores de referência e expressões fechadas passaram.
O terceiro teste de orçamento tardio passou separadamente porque foi adicionado
após o início dessa execução.

## Busca paralela durante a compilação

Candidatos independentes de promoção são avaliados por até quatro workers,
limitados pelos núcleos disponíveis e pela quantidade de candidatos seguros.
Cada worker recebe uma cópia do grafo interno de uma coordenada; o leitor de
pesos permanece incremental no processo principal. Operações numéricas de cada
caminho mantêm sua ordem. A escolha final usa o menor tamanho e, no empate,
a ordem serial dos candidatos. Nenhum worker ou referência é serializado no JSON.

`--condition-workers 1` executa a busca serial. Mais workers aumentam o uso de
CPU e memória temporária de compilação. Os limites de candidatos, nós e rodadas
continuam explícitos; atingir um limite não declara convergência.

Na coordenada terminal medida, quatro workers levaram 43,5 segundos contra
140,6 segundos na busca serial, aproximadamente 3,2 vezes mais rápido.
Os tamanhos e contadores coincidiram. Os três testes focados verificam a escolha
determinística, entradas condicionais, operações indefinidas, zeros com sinal e
orçamentos. A suíte integrada paralela terminou: 553 testes, 535 passando, as mesmas 15
falhas anteriores e três ignorados. Todos os 479 passes originais e os 531
passes da suíte serial imediatamente anterior foram preservados. As árvores
completas e os contadores serial/paralela coincidiram nas posições inicial e
terminal (dimensão 2). Os 864 logits passaram nas duas representações, totalizando
1.728 verificações bit a bit. O teste completo do modelo levou 790,0 segundos
contra 2.131,9 segundos na execução serial, aproximadamente 2,7 vezes mais rápido,
mesmo incluindo a comparação serial adicional. O artefato literal final continua
pendente.

O perfil da primeira coordenada fechada diferencia a origem da duplicação:
71 decisões físicas viram 117119344839022567 ocorrências de `if:f64`, enquanto
duas entradas fundamentais viram 642642545222578663320 ocorrências.
112 multiplicações F64 viram 340441137248730895154 ocorrências; 244 adições de
palavras u64 viram 266952571788309193812. Esses números medem a sintaxe literal,
não caminhos distintos de execução. A duplicação aritmética e das conversões
continua sendo um gargalo mesmo quando decisões são combinadas. O perfil está
registrado no mapa de validação para orientar a próxima redução exata.

## Arredondamento F16 com sinal incorporado — validado

Quando o intervalo provado exclui overflow F16 (magnitude menor que 65520),
o braço normal soma o viés de arredondamento diretamente ao word com sinal.
O bit retido de paridade independe do sinal, e a adição não pode carregar para
o bit 63 nesse domínio. A máscara final preserva o sinal. O braço subnormal
mantém a quantização e restauração explícita de sinal, inclusive underflow para
-0. O caminho genérico com overflow continua usando a transformação anterior.

Os três testes focados passaram. O oráculo dyádico independente que verifica
cada midpoint F16, seus vizinhos F64 imediatos e os dois sinais agora verifica
também o novo caminho certificado. Os testes de intervalos e de dupla conversão
continuam passando. A suíte completa desta versão terminou e foi reconciliada no log
`/private/tmp/llm-inner-json-signed-round-full-suite.log`: 553 testes, 535
passando, as mesmas 15 falhas anteriores e três ignorados. Todos os passes
da suíte anterior foram preservados. As árvores e os contadores serial/paralela
coincidiram, e os 864 logits passaram nas duas representações (1.728 verificações
bit a bit). O teste do modelo levou 987,3 segundos, mas a busca ampliada rodou
concorrentemente; esse tempo não deve ser usado como comparação isolada de velocidade.

Na posição 0/dimensão 2, a nova forma permitiu uma segunda promoção e caiu de
72659232914655964882628 para 58127363857017047908129 bytes previstos, cerca de
20%. A busca completou três rodadas, 240 candidatos e seu próprio ponto fixo.
Na posição 7/dimensão 2, caiu de 14559843742299065345314960831221058 para
14559833163458060370270377838148162 bytes, somente 0,000073%. A profundidade
caiu de 484 para 472; as referências à entrada e decisões expandidas permanecem
iguais nessa coordenada. O ganho inicial não representa o vetor inteiro.
A busca terminal continua incompleta pelo limite de candidatos. Não houve
emissão do JSON final. A validação completa desta alteração foi concluída.

## Compartilhamento estrutural interno — validado

Após a primeira promoção terminal, 437 objetos de condição representavam
somente 183 estruturas distintas. As duas simplificações de ramo construíam
cópias de prefixos idênticos. Isso desperdiçava busca e consumia o orçamento de
candidatos com condições equivalentes, mesmo sem crescimento da sintaxe final.

O compilador agora interna os nós por operador, dtype, payload exato de folhas
e IDs estruturais dos filhos. A travessia é iterativa, admite orçamento de nós
e rejeita ciclos. Ela não calcula valores, modifica operações ou reordena
reduções. A árvore literal, seus bytes, ocorrências e ordem de avaliação
permanecem iguais; apenas aliases em memória do compilador se unem. Nada desse
índice é emitido no JSON final. Assim, a contagem de candidatos agrega consultas
estruturalmente idênticas em vez de consumir o orçamento com suas cópias.

Os 39 testes focados passaram, incluindo quatro novos testes de estrutura,
IEEE/payloads/zeros, falhas condicionais e uma redução que antes ficava oculta
porque a mesma condição existia como dois objetos diferentes.

Na coordenada inicial, os nós físicos caíram de 4320 para 1310 e os candidatos
cumulativos de 240 para 104, mantendo exatamente os mesmos bytes literais.
Na busca terminal ampliada (2048 candidatos, 16 rodadas), cinco rodadas
examinaram 1010 candidatos, aceitaram quatro promoções e atingiram o ponto fixo
da transformação. O tamanho caiu de 14559833163458060370270377838148162 para
8051335912809326043023129588358268 bytes, aproximadamente 45% menos. A expressão
possui 8549 nós físicos, 255 decisões físicas e profundidade 470. Doze casos
terminais comparados com PyTorch passaram bit a bit. Essa medição levou 42,0
segundos; a busca anterior sem compartilhamento foi interrompida sem resultado
após a duplicação estrutural ser demonstrada, não se deve atribuir a ela um
speedup completo. A medição com limites padrão terminou em 41,5 segundos e produziu exatamente
os mesmos contadores e medidas, com os mesmos 12 passes bit a bit. Portanto,
a busca terminal atinge seu ponto fixo também com a configuração padrão, sem
parar no limite de candidatos. Isso vale para essa transformação, não para
todas as regras matemáticas possíveis. A suíte completa terminou: 557 testes,
539 passando, as mesmas 15 falhas anteriores e três ignorados. Todos os passes
anteriores foram preservados; as árvores serial/paralela coincidiram e os 864
logits passaram em ambas as representações (1728 verificações bit a bit).
O teste completo do modelo levou 519,4 segundos (8,7 minutos), contra 790,0
segundos (13,2 minutos) na execução paralela anterior. A execução intermediária
de 987,3 segundos tinha uma busca ampliada concorrente e não é uma comparação
isolada de velocidade.
O CLI preparou 1/32 coordenadas e recusou a emissão por tamanho antes de escrever
a primeira expressão. O tamanho literal continua inviável e o JSON final
permanece pendente.

## Provas de precisão e intervalos de palavras — paridade validada; custo corrigido em validação

Duas regressões focadas reproduziram a perda de prova: uma identidade ou
alargamento retornava o mesmo produtor F16, mas substituía sua anotação de
42 bits inferiores zerados pela anotação F32 de 29 bits. O compilador e o
simplificador de bits agora preservam a maior prova já demonstrada. Os três
testes específicos passaram após a correção; a equivalência numérica cobre
todos os 63.488 valores F16 finitos. Essa correção isolada não reduziu a
expressão inicial do Llama.

As comparações de magnitude que escolhem os braços F16 agora operam sobre
palavras u64 com o sinal removido. A ordem unsigned coincide com a ordem dos
valores positivos nos limiares finitos utilizados; NaNs continuam acima deles.
Isso permite propagar fatos pelos operadores inteiros sem introduzir álgebra
real sobre as operações de ponto flutuante. O teste cobre todos os words F16,
limiares e vizinhos F64, payloads NaN e words aleatórios.

A análise conservadora de intervalos percorre somas, produtos, divisões
definidas, máscaras, shifts e conversões unsigned. Ela conserva intervalos
modulares apenas quando ambos os extremos pertencem à mesma célula de wrap;
caso contrário retorna o domínio completo. Fatos de um operando permanecem
relevantes nas condições derivadas de seus descendentes inteiros. Condições
que poderiam esconder uma operação indefinida continuam protegidas pelo
critério de totalidade. Os testes exaustivos cobrem os 65.536 valores u16.

Os 45 testes focados e as 12 comparações terminais com PyTorch passaram. Na
coordenada inicial a previsão caiu para 58125302555898598982655 bytes; na
terminal, para 8051052324794883384027948204697032 bytes. O ganho terminal é
somente cerca de 0,0035%; decisões e referências à entrada não diminuíram.
A medição terminal levou 96,4 segundos com testes concorrentes, portanto não
demonstra aceleração. A suíte completa está em andamento; o mapa de validação
registra seu log e hashes dos arquivos testados. O JSON final permanece
pendente e nenhuma coordenada literal completa foi emitida.

Uma investigação isolada de semente quadrática com duas iterações de Newton
testou exaustivamente 16.777.216 entradas F32 normalizadas, incluindo as duas
paridades de expoente. Falhou em seis pontos; por exemplo, 0x3f800001 produziu
0x3f800001 em vez de 0x3f800000. Uma busca de 129 vieses e outra de 129 sementes
com endpoints fixos também não obtiveram paridade. O algoritmo foi rejeitado
e não alterou a raiz utilizada pelo compilador. Os logs estão no mapa.
O CLI desta versão preparou 1/32 coordenadas e recusou o tamanho antes de
emitir a primeira expressão; continuam zero coordenadas finais emitidas.

A semente racional ajustada passou nos mesmos 16.777.216 pontos normalizados
e em 53.775 verificações da expressão JSON, cobrindo todos os valores F16
positivos finitos, extremos de todos os expoentes F32 normais e amostras F32.
O protótipo terminal preservou as 12 comparações com PyTorch e reduziu o tamanho
previsto para 5151393979909648332159646255562184 bytes (36,0% menos), com 34,9%
menos referências à entrada e 24,9% menos decisões expandidas. Ainda não foi
adotado: faltam o certificado durável e a comparação do vetor completo.

Um segundo protótipo reutiliza provas de totalidade e consulta a memoização
por escopo antes de recalcular intervalos. Manteve medidas, contadores e os
12 resultados exatos da versão de intervalos, levando 55,7 segundos em vez
de 96,4 segundos na medição inicial. As execuções tiveram outros testes
concorrentes; a comparação indica trabalho redundante evitado, mas não é um
benchmark isolado. A implementação de produção permanece estável para a
suíte atual. Os dois protótipos estão fora do repositório e não contam como
validação final nem coordenadas emitidas.

A suíte anterior ao cache terminou com 563 testes, 545 passes, as mesmas 15
falhas e três ignorados. Os 539 passes da rodada anterior e os 479 originais
foram preservados. Os 864 logits passaram nas duas representações (1.728
comparações bit a bit), e árvores/contadores serial e paralelo coincidiram.
Porém o teste do modelo levou 1240,8 segundos, contra 519,4 na rodada anterior;
isso é uma regressão de tempo nesta execução, não um ganho de desempenho.
A correção do cache de provas foi aplicada após a suíte terminar. O build
passou; testes focados e a nova suíte completa foram iniciados, registrados
separadamente para não atribuir o resultado anterior ao código alterado.

Os 45 testes focados da correção de cache passaram. Sua suíte completa ainda
está em execução; os hashes separam essa versão da suíte concluída anterior.
O código reproduzível da investigação da raiz foi salvo em
`docs/experiments/direct-json-rational-sqrt-proof.cpp`. A compilação com
`-O3 -ffp-contract=off -std=c++17` e a execução desse arquivo repetiram os
16.777.216 pontos sem divergência. Esse arquivo é um experimento de prova,
não uma nova primitiva do compilador nem o artefato final do Llama.

## Raiz racional normalizada — implementação em validação

A primitiva nova recebe um F32 positivo, finito e normal, exatamente alargado
para F64. O compilador mantém a exigência da prova desse domínio antes de
baixar `pending-sqrt`. A semente racional usa coeficientes binários fixos,
duas iterações Newton F64 e conversão F32 por bits. Não usa raiz nativa,
lookup de respostas, referência a peso ou intermediário serializado. As
implementações anteriores continuam como referências independentes nos testes.

Para x = m × 2^e, m = 1 + fraction × 2^-23, decompomos e = 2k + p, p em {0,1}.
A semente e as duas iterações operam somente sobre m. O fator de escala é 2^k
ou o valor binário64 de sqrt(2) multiplicado exatamente por 2^k. O expoente
dessa escala é reconstruído por bits, sem decisão. No domínio F32 normal,
todas as etapas F64 e a saída F32 são normais e finitas. Multiplicar por 2^k
comuta exatamente com os arredondamentos dessas operações. Portanto cada
resultado é o ponto normalizado da mesma paridade, escalado exatamente.

A grade de 16.777.216 pontos ([1,2) e [2,4)) cobre ambas as paridades e, pela
covariância de potências de dois, os 2.130.706.432 words F32 positivos normais.
O novo teste gera C++ da própria árvore JSON literal, com contração FMA
desabilitada, exige avaliação IEEE sem precisão estendida e compara todos os
pontos normalizados com a raiz F32 nativa de referência. Outro teste cobre
todos os positivos F16 finitos, limites de todos os expoentes normais e
amostras F32. A expressão nova contém 26 ocorrências da entrada e nenhuma
decisão, contra 62 na raiz exata anterior de quatro iterações.

O fonte da nova primitiva foi preparado enquanto a suíte da versão com cache
executava o dist anterior. O typecheck passou. O build e os testes novos
aguardam a conclusão dessa suíte para não misturar versões de workers. O
protótipo terminal já mostrou redução literal de 36% e 12 comparações exatas;
isso ainda não demonstra a paridade do vetor completo nesta implementação.

A suíte completa do commit 93e39a5, com cache de provas, terminou: 563 testes,
545 passes, as mesmas 15 falhas e três ignorados. Todos os passes anteriores
e originais foram preservados; árvores/contadores serial e paralelo coincidiram
e as 1.728 comparações de logits passaram. O teste do modelo levou 699.8
segundos, contra 1240,8 antes do cache e 519,4 antes das regras de intervalos.
Os resultados não são de benchmarks isolados: o custo das regras novas ainda
precisa ser acompanhado apesar da recuperação obtida com o cache.

A medição isolada sequencial levou 31,6 segundos com 14 workers e 55,2
segundos com quatro (1,75× mais rápido). Medidas e contadores coincidiram,
e cada execução preservou as 12 comparações exatas. O padrão agora usa
`availableParallelism()`, limitado pelas CPUs disponíveis e pelo número de
candidatos seguros; `--condition-workers` continua permitindo um limite
explícito. Nenhuma operação numérica do modelo é reordenada. O build passou.
Os testes focados completos e a suíte integral dessa versão foram iniciados;
o mapa registra os hashes e logs separados. A emissão literal continua
pendente: a redução de tamanho não tornou o arquivo final viável.

Os 52 testes focados passaram sem skips, incluindo os casos críticos RMS com
PyTorch e a prova exaustiva gerada da expressão JSON. O CLI preparou 1/32
coordenadas (37186567571186275285119 bytes previstos na primeira), mas recusou
a emissão pelo orçamento antes de escrever a expressão. Zero coordenadas
finais foram emitidas; a suíte completa da nova versão continua em execução.

### Próxima investigação: preservar os operadores escalares F32

O contrato requer tipos, ordem e conversões exatas. A representação atual
promove cada operação F32 a F64 e reexpande seu arredondamento por bits. Isso
introduz duas cópias do cálculo em cada fronteira, mesmo quando um operador
escalar F32 direto expressaria a operação original com seu dtype. Não existe
round/nearest-even no JSON, mas essa expansão artificial pode dominar o
tamanho. Investigar um backend de expressões literais com operadores F32
nativos e casts F16/F64 ainda baixados por bits, sem preservar intermediários,
referências, primitivas exp/sqrt/SiLU ou estrutura do modelo. A auditoria atual
veda F32 aritmético; essa política precisa ser avaliada contra a semântica,
não simplesmente contornada ou desabilitada para obter um passe.

Um protótipo pode mapear os produtores F16 para valores F32 exatamente
alargados, usar uma conversão F32→F16→F32 por words u32 (braço normal assinado
e subnormal quantizado por (abs(x)+0.5)-0.5, com sinal restaurado), e preservar
a ordem dos operadores binários F32. A raiz pode normalizar diretamente o
word F32 para o racional F64 e retornar F32 por bits. Os kernels exp/SiLU
certificados podem continuar em F64 com fronteiras explícitas; não substituir
produto+adição fundidos do perfil exp por dois arredondamentos F32 separados.
São necessárias provas exaustivas das conversões, paridade integral contra
PyTorch e medição literal antes de adotar qualquer alteração da representação.
Essa investigação permanece proposta; o compilador atual ainda usa F64.

A suíte completa da raiz racional com o novo padrão de workers terminou:
565 testes, 547 passes, as mesmas 15 falhas e três ignorados. Todos os 545
passes anteriores e os 479 originais foram preservados. As árvores completas
e os contadores serial/paralelo coincidiram nas coordenadas inicial e terminal.
Os 864 logits passaram nas duas representações (1.728 comparações bit a bit).
O teste do modelo levou 568,6 segundos, contra 699,8 com cache e raiz anterior
e 1240,8 antes do cache. Ainda ficou acima dos 519,4 da versão anterior às
regras de intervalos; o benchmark isolado de workers demonstra o ganho de
paralelismo, mas não elimina essa diferença entre versões completas.
Os hashes do fonte testado foram reconferidos. O JSON final segue pendente,
com zero coordenadas emitidas, apesar da paridade da expressão compilada.


## Substituição incremental por dependência (2026-10-02)

A CLI agora seleciona uma coordenada com `--position` e `--dimension` (defaults
0 e 0). O registro de crescimento é `OUTPUT.growth.PID.jsonl`; cada processo
possui seu arquivo, sem sobrescrever o diagnóstico de outro compilador. Há
medidas literais antes/depois de cada substituição e após cada dependência.

Uma sessão do lowerer vive somente durante a compilação dessa coordenada. Ao
concluir uma dependência da fonte, baixa seus produtores com filhos já
estabilizados; repete regras estruturais e de precisão até um ponto fixo
conjunto, propagando os fatos dos braços condicionais. Somente então libera o
consumidor. Não distribui combinações de condições nessa fase. A fase posterior
de cofactoring recebe as expressões estabilizadas e mantém sua própria prova de
ponto fixo. A AST de referência da fonte permanece exclusivamente no compilador.

As regras que não alteram uma árvore agora preservam sua identidade. Assim,
a reescrita não duplica artificialmente produtores equivalentes nem perde sua
proveniência por reconstrução inútil. A subtração de produtores estruturalmente
iguais reconhece o zero no domínio finito admitido. Projeções leem o peso antes
de solicitar sua dependência: pesos zero dispensam o produtor e pesos um
preservam diretamente o operando. Na redução finita de produtos F16, iniciada
em +0, produtos zero de qualquer sinal têm contribuição equivalente. Uma única
posição causal torna `exp(score-score)` exatamente um; elimina Q/K e a
exponencial antes da expansão, preservando V e o restante do forward.

O gravador aceita a coordenada selecionada como um artefato escalar explícito
(`coordinate-header` / `coordinate-end`, `completeVector:false`). O protocolo
anterior do vetor completo permanece separado e exige cobertura integral. A
emissão escalar sintética foi testada por gravação e releitura; isso não é o
artefato do Llama.

A coordenada posição 0/dimensão 2 preservou os bits dos 60 casos do corpus vivo.
Entradas possuem vários comprimentos, mas tokens posteriores não influenciam
a posição 0 por causalidade. Essa prova não demonstra a saída do último token
para todos os comprimentos nem paridade de um arquivo final serializado.
A tentativa real de emissão ainda foi recusada, antes da expressão, com
37.186.567.571.186.275.285.119 bytes previstos e zero unidades emitidas. O tamanho
final não diminuiu nessa coordenada: a mudança evita expansão e reconstrução
prematuras, mas não resolve a duplicação literal das conversões e dos kernels.

O mapa de validação mantém os resultados históricos das 32 coordenadas e
acrescenta a rodada restrita: 566 testes, 548 passes, as mesmas 15 falhas e
3 skips. Os três testes de `direct-json-model.test.ts` ficaram explicitamente
adiados nesta rodada para não avançar o teste integral de todas as coordenadas.
Os quatro testes novos cobrem identidade estável, ordem de substituição,
gravador de coordenada e paridade viva da coordenada completa na memória do
compilador. A meta do JSON efetivo e sua paridade continua pendente.


## Proveniência do sinal durante a substituição (2026-10-02)

O sinal requerido pela conversão F16 passa a ser deduzido do produtor original
quando isso foi provado no domínio finito. Produtos e divisões usam o XOR dos
sinais dos operandos; raiz positiva normal e exponencial certificada têm sinal
positivo. Widening e conversão F16 preservam o sinal, inclusive underflow para
-0. A SiLU certificada preserva o sinal em seu domínio completo de entradas
F16. Somas permitem propagação somente quando as provas dos dois sinais são
iguais. Intervalos contendo zero nunca classificam seu sinal.

Isso permite à conversão usar o sinal de X, por exemplo, em vez de repetir
X multiplicado pelo inverso positivo da raiz. Sem prova, mantém a extração do
sinal da expressão completa. Essas provas e memoizações vivem somente no
compilador; o JSON continua contendo cálculos literais e máscaras de bits.
Nenhuma ordem aritmética ou fronteira de arredondamento foi alterada.

Na coordenada posição 0/dimensão 2, o tamanho literal previsto caiu de
37.186.567.571.186.275.285.119 para 31.363.758.398.253.704.659.919 bytes
(15,7%). Os 60 casos vivos continuam iguais bit a bit. A CLI tentou gravar a
coordenada e recusou a expressão por orçamento: zero unidades, finalParity=false.
Ainda não existe o JSON efetivo do Llama nem prova de paridade do arquivo final.

A regressão restrita anterior foi preservada: 569 testes, 551 passes, as mesmas
15 falhas conhecidas e 3 skips; os testes do vetor completo permanecem adiados.
Os testes focados adicionais incluem 1.015.808 produtos/divisões F32 sobre todos
os F16 finitos, sinais de coeficientes, zeros assinados e fronteiras de conversão
direta e dupla. O mapa registra separadamente a regressão e os testes acrescentados
ou repetidos depois dela, com SHA dos fontes e logs.


A conversão interna da SiLU usa a mesma proveniência. Fora da célula tiny,
`.5*x` domina a correção não negativa, limitada por `.025*|x|` em `|x|<=.1`.
O resultado mantém distância de pelo menos `.475*|x|` de zero; as operações
F64 não podem mudar seu sinal nesse domínio. A célula tiny preserva seu sinal
separadamente. A conversão extrai o sinal de x em vez de repetir o polinômio.
O certificado completo foi repetido após essa mudança (23.758 pontos na faixa
máxima), e a mesma coordenada manteve as 60 comparações bit a bit.
O tamanho previsto passou a 28.192.143.437.206.182.792.739 bytes, redução total
de 24,2% contra o commit anterior. A emissão continua recusada: zero unidades.
A validação específica dessa mudança posterior à regressão consta separadamente
no mapa; não afirma uma nova execução da suíte inteira.


## Experimento de identidade nas células residuais (2026-10-02)

`direct-json-residual-cell.ts` prova quando `R16(R32(base+correction))` conserva
exatamente um base F16 não zero. Primeiro ajusta o limite de correção para o
maior código F16 admissível, sem aproximar valores pequenos para zero. Reserva
1/4096 do raio conservador da célula para o erro da soma F32. Exige desigualdade
estrita; zeros de base continuam no caminho original para preservar sua
canonicalização. O caminho complementar mantém toda a expressão residual.

Os testes verificam todos os 63.488 códigos F16 finitos, para sete limites,
com o guard JSON e os dois extremos admissíveis de correção. Um oráculo dyádico
independente confirma os bits finais; monotonicidade de R32 seguida de R16
cobre as correções intermediárias. Incluem -0, +0 e limites inválidos. A rodada
focada teve dez passes, incluindo as 60 comparações vivas da coordenada usada.
A prova discreta final foi repetida após o ajuste dos limites (dois passes).

A integração é exclusivamente experimental: `residualCellGuards` no builder é
false por padrão e a CLI não a habilita. As expressões são exatas, mas seu
resultado literal aumentou de 28.192.143.437.206.182.792.739 para
28.197.564.004.341.063.533.925 bytes. O orçamento de 2.048 candidatos/16 rodadas
terminou no mesmo ponto fixo; o problema não era o limite de candidatos.
Essa estratégia não foi adotada como otimização do compilador principal.
O modo padrão repetiu a paridade e o tamanho anteriores.

A lacuna a investigar é a passagem das provas numéricas por escopos de
condições. O lowerer calcula faixas globais, fecha operações numéricas em
bitwise e somente depois o cofactoring promove condições. A simplificação
estrutural propaga fatos inteiros, mas não refina as faixas dos produtores
originais de acordo com esses escopos. Apenas inserir guardas antes dessa
correção aumenta a expressão complementar. A evidência fica preservada no mapa;
não equivale à emissão do JSON do Llama, que permanece pendente.


## Propagação da pré-imagem F16 e provas de magnitude (2026-10-02)

O lowerer recupera agora o limite do produtor F32 a partir do limite já
certificado de seu resultado F16. A pré-imagem usa o ponto médio exato
entre códigos F16; empates pertencem ao código par. No máximo finito,
exclui o empate que produz infinito. Essa informação correlacionada
elimina braços de overflow e permite fundir conversões F32/F16, sem
reordenar operações nem introduzir guardas de runtime.

O teste independente cobre todas as 31.744 células de magnitude F16 finita,
ambos os sinais e os dois lados da fronteira F32: 126.976 comparações de bits.
A infraestrutura opcional de magnitude conserva domínios com uma lacuna
em torno de zero e prova identidades residuais/conversões normais dentro
do escopo. Não presume escopos na compilação padrão. Quem dividir um domínio
deve manter seu complemento; metadados e caches ficam somente no compilador.

A coordenada posição 0/dimensão 2 caiu de 28.192.143.437.206.182.792.739
para 3.525.760.683.880.951.562.623 bytes literais previstos (87,5%). O teste
integrado recapturou o forward PyTorch e verificou 60 resultados bit a bit.
Os comprimentos variam entre 1 e 8, mas tokens posteriores não influenciam
a posição zero: não é prova da última posição para todos os comprimentos.

A divisão experimental por |X1| >= 16 conserva os 60 resultados, mas sua
união completa prevê 3.746.307.046.986.377.784.855 bytes, maior que o padrão
atual. Não foi adotada como otimização automática. O mapa conserva esse
resultado negativo para evitar promover uma melhora só do ramo favorável.

Regressão: 577 testes, 559 passes, as mesmas 15 falhas conhecidas e 3 skips;
nenhum teste anteriormente aprovado foi perdido. Três testes históricos
que compilam várias coordenadas continuam adiados. Testes focados: 11 passes.
Os logs, corpus de referência e crescimento por substituição/dependência
estão em docs/evidence e vinculados ao mapa por SHA-256.

A CLI tentou emitir a coordenada com orçamento de 64 MiB e recusou antes
da expressão: zero unidades, finalParity=false. A queda de tamanho não é
o artefato. O JSON efetivo e sua paridade permanecem pendentes; outras
coordenadas não foram compiladas nesta rodada.


## Célula F32 constante antes de expandir produtores (2026-10-02)

`direct-json-rounding-cell.ts` delimita a operação real de soma, subtração
ou produto usando as faixas já provadas dos operandos. Cada extremo
recebe um passo F64 para fora; só dobra o resultado se ambos arredondarem
para o mesmo F32 finito não zero. Assim o teste não depende de a expressão
inteira ter sido fechada. Empates não demonstrados, overflow, zeros, faixas
inválidas e divisões conservam o caminho original. A análise lê a sintaxe
dos produtores e seus certificados, sem gerar sua representação bitwise.

A raiz passa a testar seu operando depois da substituição, corrigindo uma
lacuna: antes, apenas constantes presentes no fonte disparavam o folding.
Uma constante descoberta pela simplificação agora elimina também a raiz.
Os testes incluem um produtor variável eliminado antes do fechamento,
comparações com oráculo dyádico e preservação de zeros assinados.

A coordenada completa posição 0/dimensão 2 conservou as 60 comparações
exatas, mas seu tamanho previsto não mudou: 3.525.760.683.880.951.562.623
bytes. A união de escopos também conservou o tamanho anterior. Essa regra
corrige uma lacuna do simplificador; ainda não demonstrou redução adicional
nesse checkpoint. Não é a emissão da coordenada, que continua pendente.


## Decidir condições antes de fechar seus ramos (2026-10-02)

O lowerer anteriormente visitava ambos os ramos de um `if` antes de
estabilizar sua condição. Agora fecha e simplifica a condição primeiro.
Uma constante seleciona somente o ramo alcançável, evitando fechar
produtores que seriam descartados. Comparações eq/lt/le também usam
faixas finitas certificadas antes de expandir seus operandos. Faixas
sobrepostas, desconhecidas, inválidas ou não finitas conservam a decisão.
Igualdade numérica trata +0 e -0 conforme a semântica IEEE da comparação.

Os testes verificam estritamente ambos os extremos, igualdade de zeros,
comparações desconhecidas e a seleção após substituição. Um ramo morto
contém um produtor numérico sem certificado: o teste só passa se esse
produtor não for visitado. O resultado do ramo sobrevivente foi comparado
sobre todos os 63.488 códigos F16 finitos.

A coordenada padrão manteve 60 comparações exatas e o mesmo tamanho literal
previsto. Essa mudança corrige a ordem de fechamento das condições, mas
não reduz adicionalmente o checkpoint atual. O JSON efetivo continua
pendente; os detalhes da rodada estão em constantConditionValidation no mapa.

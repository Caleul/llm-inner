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
  perfil de referência: 23.758 pontos na faixa máxima. O perfil é uma fonte de
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

## Diagnóstico de duplicação e limite de conclusão

`direct-json-measure.ts` calcula exatamente ocorrências e bytes da árvore que
seria serializada, sem renderizá-la. Valores BigInt evitam overflow do contador.
Após fechamento e simplificação, posição 0/dimensão 2 possui 1.266 nós físicos,
29 decisões físicas, profundidade 456 e aproximadamente 6,89e28 bytes expandidos.
Posição 7/dimensão 2 possui 7.913 nós, 170 decisões, profundidade 674 e
aproximadamente 8,33e42 bytes. Essas medidas incluem duplicação de operandos
nas expansões numéricas; não são tamanho de arquivo gerado nem número de
bifurcações originais do modelo. Nenhuma emissão desse volume foi iniciada.

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

Validação atual: `test/direct-json.test.ts` tem 22 casos; também cobre widening,
raiz, exponencial, SiLU certificado, conversões compostas, medição de duplicação
e cancelamento de round com prova de precisão. `test/direct-json-model.test.ts`
cobre os 32 pares posição/dimensão e as 864 comparações antes e depois do
fechamento/simplificação. Este teste é pulado sem as variáveis
`LLM_INNER_DIRECT_PYTHON` e `LLM_INNER_DIRECT_JSON_CHECKPOINT`; a validação registrada
forneceu ambas. Os 23 testes focados passaram sem skips. O corpus inclui máximos F16 com
ambos os sinais, magnitudes misturadas e fronteiras normal/subnormal.

Suíte completa atual: 520 testes, 502 aprovados, 15 falhas, 3 skips.
Todos os 479 testes aprovados no baseline continuam aprovados; as mesmas 15
falhas anteriores estão mapeadas em `docs/direct-json-validation.json`. A suíte
completa final e os contadores de simplificação são reconciliados nesse arquivo.
Nenhum teste antigo foi apagado ou relaxado. Arquivo final do modelo ainda pendente.

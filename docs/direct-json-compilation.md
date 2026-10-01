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

## Marcos restantes

1. Adaptador genérico: construir/substituir expressões diretamente de config e
   pesos paginados, sem passar pelo texto Rust existente.
2. Provas e regras matemáticas ampliadas; widening exato e operações numéricas
   completas para SiLU, raiz e exponencial, sem tabelas de respostas em runtime.
3. Inventário das decisões originais por operação, redução e duplicação.
4. Uma dimensão arbitrária completamente compilada e comparada com PyTorch.
5. Todas as dimensões/posições, savepoints e artefato final completo independente
   do checkpoint; avaliação diagnóstica bit a bit para múltiplas entradas.
6. Comparação da suíte completa com baseline equivalente, commit e push quando
   houver remote configurado. Atualmente nenhum remote está configurado.

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

Primeira validação: 44/44 testes aprovados, incluindo os 11 novos e os testes
existentes `direct-round-preimage` e `direct-flat-substitution`. Log em
`/private/tmp/llm-inner-direct-json-tests.log`. O vetor final do Llama e sua paridade
continuam pendentes. Não foram inferidos a partir desse resultado.

# Preimagem exata da condição da ativação — 2026-10-03

Na aproximação já certificada `p = x * (0.5 + x * 0.25)`, a condição
que separa os caminhos de conversão normal e subnormal repetia `x` duas
vezes. Para `x` Half dentro de `[-3/128, 3/128]`, o compilador agora usa:

```text
(x + 2**-25)**2 < 2**-26
```

Essa condição usa `x` uma vez. Não é uma aproximação da decisão: a preimagem
sobre a grade Half contém `-2**-13` e exclui `+2**-13`. A soma deslocada e
seu quadrado são exatos em F64 nesse domínio. O certificado foi verificado
nos 19.458 padrões Half admitidos, incluindo ambos os zeros, comparando a
condição com o limiar F32 original e verificando a exatidão com racionais.
A avaliação nativa do kernel emitido também coincide bit a bit com SiLU
Torch em todos esses valores. Os caminhos e os cálculos dos resultados
continuam preservados; apenas a expressão da condição foi substituída.

A regra exige o formato quadrático exato, o mesmo operando nas duas posições,
a prova de dtype Half e o intervalo admitido. Expressões diferentes, fontes
sem prova e o intervalo maior `[-1/32,1/32]` mantêm o caminho genérico.
Parênteses, substituição, `factor()` e `simplify()` continuam obrigatórios.
A redução não depende dos pesos ou da arquitetura Llama.

No kernel pequeno, as ocorrências de entrada caíram de sete para seis e
os caracteres de 410 para 364. No checkpoint, o fechamento estabilizado
da ativação caiu de 8.230.451.759 para **6.913.579.521 caracteres**: menos
1.316.872.238 caracteres, aproximadamente 16%. A execução local limitada
a 2 GiB conservou os mesmos 13 produtores anteriores e recusou a ativação
antes da alocação. Esse ensaio não emitiu o arquivo de ativação reduzido.

Os 90 testes Python passaram, as oito integrações passaram e o build passou.
A regressão geral manteve 599 testes: 581 passaram, três foram pulados e
as mesmas 15 falhas anteriores permaneceram, sem falhas novas.

A validação Colab está sendo registrada separadamente: a execução da
composição anterior e a execução desta condição possuem raízes distintas
e manifests próprios. Não reutilizamos estados com identidade incompatível.
A primeira coordenada e a paridade final do modelo continuam pendentes.

O Colab emitiu a ativação reduzida e verificou os 15 produtores completos:
300 comparações em 60 entradas, zero divergências. A ativação possui SHA-256
`0c23d581d09ba4a9b33c073801d10fc1dc636e4cf59050a3a904e624336a351f`.
A compilação sequencial durou 146,299 s e parou antes de materializar o
produto seguinte, estabilizado em 32.921.807.102 caracteres, acima de 9 GiB.
A exportação gzip possui 147.062.682 bytes. O arquivo completo foi recuperado
em `artifacts/direct-sympy-quadratic-guard/activation.expr`, descomprimido em
blocos de 1 MiB e conferido pelo SHA-256 e tamanho. Ele é um produtor parcial,
não a saída final do modelo. Nenhum estado Linux foi importado como estado
macOS.

O teste CUDA também passou: 888.832 produtos Half, 19.458 valores SiLU e
19.458 decisões da condição quadrática. Não houve diferenças na decisão,
nem nos bits do quadrado em F64 entre CPU e A100. Os testes numéricos usam
CUDA; fatoração e simplificação permanecem na CPU.

## Próximo ponto de simplificação

O crescimento restante concentra-se nos operandos normalizados repetidos
nas ramificações de conversão. As visões atuais preservam as condições,
mas as provas numéricas usadas no fechamento ainda são majoritariamente
limites do produtor inteiro. O próximo trabalho deve carregar os limites
provados de cada braço em seu próprio contexto, sem transferir a prova de
um braço para o produtor inteiro. Isso permitirá eliminar caminhos de
conversão impossíveis antes de compor a próxima dependência. Essa mudança
não está implementada neste commit e a coordenada continua incompleta.

## Comparação da versão com a nova condição

A execução sequencial levou 146,299 s, a paralela 151,115 s. Ambos os modos
emitiram exatamente os mesmos 15 produtores, incluindo o SHA-256 da
ativação reduzida. Houve 300 comparações por modo, sem divergências. O
paralelo ficou aproximadamente 3,3% mais lento. Seu pico agregado de RSS
amostrado foi 29.201.965.056 bytes; esta execução sequencial não coletou RSS,
portanto não fazemos uma comparação de memória entre os dois modos aqui.

O primeiro lançamento paralelo falhou no preparo do diretório pelo cliente
Colab. A sessão foi consultada, um snapshot informou que o lançamento não
havia ocorrido e o arquivo do PID não existia. Só então foi repetido o
lançamento, na mesma sessão. A nova execução terminou e seus arquivos
foram verificados antes de registrar esta comparação.

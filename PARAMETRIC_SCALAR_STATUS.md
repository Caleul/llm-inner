# Compilador escalar paramétrico F16: estado verificado

## Contrato

O alvo é uma função por dimensão final, parametrizada pela posição e pelo número de tokens. A função recebe apenas a entrada original, contém pesos literais extraídos do Safetensors e preserva os pontos de arredondamento. Paridade de um estágio intermediário não satisfaz esse contrato.

## Evidência atual

- `src/fixed-f16-parametric-formulas.ts` compila projeções, RMSNorm, MLP e atenção causal em fórmulas escalares. A atenção substitui Q/K/V, RoPE, softmax, AV e O na fórmula de cada dimensão; o limite causal é `j <= t`, sem comprimento fixado na compilação.
- A RMSNorm de entrada é substituída na atenção. O fator de normalização escalar pode ser calculado para `t` ou `j` dentro da fórmula. O residual da atenção também é gerado.
- No checkpoint `artifacts/tiny-random-llama`, PyTorch 2.12.1 em CPU com `eager` e F16: as 16 dimensões de saída da atenção das camadas 0 e 1 coincidiram bit a bit para comprimentos 1 a 8. Entradas da atenção com escala 10, 100 e 1000 também coincidiram. A comparação para normalização + atenção + residual usou os valores de entrada bruta e da normalização seguinte, para comprimentos 1 a 4, nas duas camadas.
- Nas entradas de escala 10/100/1000, as diferenças de escores válidos após subtrair o máximo incluíram o intervalo de −128 a −1 e valores menores que −128. A paridade observada não certifica todas as entradas possíveis nem todas as diferenças de `exp` entre JavaScript e PyTorch.
- RMSNorm e MLP isolados, e RMSNorm substituída no MLP, coincidiram por dimensão para comprimentos 1 a 4.

## Composição final do fixture

A substituição textual do MLP no residual havia sido estimada em aproximadamente 25,3 bilhões de caracteres. `composeCachedScalarSource` e `composeCachedScalarSources` evitam essa cópia: a função gerada calcula uma coordenada escalar por posição/dimensão e a reutiliza em um cache local ao cálculo. O cache não recebe ativações de camada, não lê o checkpoint e não é um executor de operações.

O teste `duas camadas completas preservam valores finais por dimensão com cache escalar` compara o resultado da camada 0 e da camada 1 com o forward PyTorch para comprimentos 1 a 4. Também compara **todos os 32.000 logits, em todas as posições**, para comprimentos 1 e 8 quando a entrada é `inputs_embeds`, e para comprimentos 1 e 4 quando a entrada é uma sequência de IDs que atravessa o embedding literal. As comparações passaram bit a bit no ambiente descrito acima.

## Descoberta e limites do produto

`compileFixedF16ScalarModelFromDirectory` usa o catálogo e o construtor de forward já existentes no projeto. Ele lê configuração e tensores, verifica a ordem e as dependências de cada operação e compila somente o padrão F16 de RMSNorm direta, Q/K/V, RoPE completo, atenção causal, projeção O, dois residuais, MLP SiLU, normalização final e projeção de logits. O compilador escalar recebe nomes de pesos descobertos no forward, sem conter nomes de tensores Llama. Um teste sem nomes fornecidos manualmente reproduziu todos os logits para sequências de IDs de 1 e 4 tokens. Padrões adicionais são rejeitados até receberem semântica numérica e validação própria.

A projeção gate da MLP agora é calculada uma vez por neurônio escalar antes da SiLU e da multiplicação com up. Essa fatoração reduziu a fórmula antes da composição com a RMSNorm e preservou a paridade bit a bit dos testes da MLP e dos logits finais.

Safetensors sozinho contém tensores e metadados; a ordem do forward vem da configuração e dos adaptadores de arquitetura existentes. `writeFixedF16ScalarArtifact` salva as fórmulas e constantes em um JSON que pode ser lido e avaliado sem reabrir o checkpoint; o teste de round-trip confirmou os valores finais. Esse JSON ainda usa a biblioteca numérica genérica do projeto para executar F16, portanto não é um módulo standalone. A validação numérica demonstrada continua limitada ao checkpoint, backend e entradas descritos acima. Faltam um módulo executável standalone, validação com outras entradas/arquiteturas e uma contagem dinâmica de operações e medição de desempenho robusta.

## Medição exploratória do fixture

Ambiente: macOS arm64, Node 24.13.0, PyTorch 2.12.1, Transformers 5.5.0, CPU eager F16; checkpoint `artifacts/tiny-random-llama`. A fonte gerada tem 87.655.527 caracteres e o JSON serializado 87.751.902 bytes. Para os IDs `[1,17,109,31999]`, uma execução mediu 0,80 s para compilar o checkpoint em fonte, 1,19 s para preparar JavaScript e 10,06 s para o segundo forward de quatro tokens; a mediana de dez forwards PyTorch após aquecimento foi 1,108 ms. As medições são de processos distintos e não são um benchmark estatístico estabilizado.

Em geração gulosa sem KV cache a partir de `[1]`, ambos produziram `[5971,10641,15242,14848]`. Com Node `--no-sparkplug`, os quatro forwards escalares levaram aproximadamente 2,26 s, 3,55 s, 6,26 s e 10,05 s. Sem essa flag, o quarto forward levou 243 s apesar de produzir o mesmo valor; um processo novo avaliou o mesmo prefixo de quatro tokens em 11,46 s. Isso aponta para compilação dinâmica do V8 como causa provável da pausa, ainda sem correção no emissor. Uma contagem dinâmica completa de operações também permanece pendente.

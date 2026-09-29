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

## Pendências do produto

O teste ainda fornece explicitamente os nomes de tensores, a ordem das operações e os parâmetros de atenção. O motor de fórmulas não assume nomes de tensores Llama, mas ainda falta um descobridor que obtenha a semântica e a ordem do forward dos artefatos do modelo. Safetensors sozinho contém tensores e metadados; não especifica o forward. Também faltam a emissão de um artefato executável independente do compilador, validação com outras entradas/arquiteturas e a comparação de tamanho, operações, tempo e tokens com o forward original. A paridade do fixture não encerra essas pendências.

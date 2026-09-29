# Compilador escalar paramétrico F16: estado verificado

## Contrato

O alvo é uma função por dimensão final, parametrizada pela posição e pelo número de tokens. A função recebe apenas a entrada original, contém pesos literais extraídos do Safetensors e preserva os pontos de arredondamento. Paridade de um estágio intermediário não satisfaz esse contrato.

## Evidência atual

- `src/fixed-f16-parametric-formulas.ts` compila projeções, RMSNorm, MLP e atenção causal em fórmulas escalares. A atenção substitui Q/K/V, RoPE, softmax, AV e O na fórmula de cada dimensão; o limite causal é `j <= t`, sem comprimento fixado na compilação.
- A RMSNorm de entrada é substituída na atenção. O fator de normalização escalar pode ser calculado para `t` ou `j` dentro da fórmula. O residual da atenção também é gerado.
- No checkpoint `artifacts/tiny-random-llama`, PyTorch 2.12.1 em CPU com `eager` e F16: as 16 dimensões de saída da atenção das camadas 0 e 1 coincidiram bit a bit para comprimentos 1 a 8. Entradas da atenção com escala 10, 100 e 1000 também coincidiram. A comparação para normalização + atenção + residual usou os valores de entrada bruta e da normalização seguinte, para comprimentos 1 a 4, nas duas camadas.
- Nas entradas de escala 10/100/1000, as diferenças de escores válidos após subtrair o máximo incluíram o intervalo de −128 a −1 e valores menores que −128. A paridade observada não certifica todas as entradas possíveis nem todas as diferenças de `exp` entre JavaScript e PyTorch.
- RMSNorm e MLP isolados, e RMSNorm substituída no MLP, coincidiram por dimensão para comprimentos 1 a 4.

## Limite material ainda aberto

O residual após a atenção normalizada contém aproximadamente 8,2 milhões de caracteres de fonte para 16 dimensões. O MLP com sua RMSNorm pós-atenção contém aproximadamente 5,6 milhões de caracteres. A substituição literal deste MLP no residual foi estimada em **aproximadamente 25,3 bilhões de caracteres** antes de gerar a primeira camada inteira. A estimativa é feita antes da alocação; essa expansão não foi executada. A composição camada 0 → camada 1, embedding, RMSNorm final e logits ainda não foi produzida, portanto os valores finais do modelo não foram validados e ainda não há benchmark comparável ao forward completo.

O próximo passo requer uma fatoração escalar equivalente que reduza esta expansão preservando a ordem de cálculo e os arredondamentos F16/F32. A fatoração deve permanecer dentro da função final por dimensão, sem aceitar ativações intermediárias como entrada nem delegar atenção/MLP a um executor de estágios. Depois é necessário comparar cada logit com o forward para múltiplas entradas e comprimentos, investigar a primeira divergência e só então medir tamanho, operações, tempo e tokens produzidos.

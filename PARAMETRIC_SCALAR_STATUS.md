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

O teste `duas camadas completas preservam valores finais por dimensão com cache escalar` compara o resultado da camada 0 e da camada 1 com o forward PyTorch para comprimentos 1 a 4. Também compara **todos os 32.000 logits, em todas as posições**, para comprimentos 1 e 8 quando a entrada é `inputs_embeds`. A rota automática com IDs literais compara todos os logits para comprimentos 1, 4 e 8. As comparações passaram bit a bit no ambiente descrito acima.

## Descoberta e limites do produto

`compileFixedF16ScalarModelFromDirectory` usa o catálogo e o construtor de forward já existentes no projeto. Ele lê configuração e tensores, verifica a ordem e as dependências de cada operação e compila somente o padrão F16 de RMSNorm direta, Q/K/V, RoPE completo, atenção causal, projeção O, dois residuais, MLP SiLU, normalização final e projeção de logits. O compilador escalar recebe nomes de pesos descobertos no forward, sem conter nomes de tensores Llama. Um teste sem nomes fornecidos manualmente reproduziu todos os logits para sequências de IDs de 1, 4 e 8 tokens. Padrões adicionais são rejeitados até receberem semântica numérica e validação própria.

Atenção, Q/K/V, escores, contexto, RMSNorm, gate/up e neurônios da MLP agora são calculados e reutilizados por coordenada escalar. Embedding e projeção final selecionam linhas de pesos literais por condicionais, preservando a ordem da redução. Essas fatorações reduziram a fórmula antes da composição com a operação seguinte e preservaram a paridade bit a bit.

Safetensors sozinho contém tensores e metadados; a ordem do forward vem da configuração e dos adaptadores de arquitetura existentes. `writeFixedF16ScalarArtifact` salva as fórmulas e constantes em um JSON que pode ser lido e avaliado sem reabrir o checkpoint; o teste de round-trip confirmou os valores finais. Esse JSON ainda usa a biblioteca numérica genérica do projeto para executar F16, portanto não é um módulo standalone. A validação numérica demonstrada continua limitada ao checkpoint, backend e entradas descritos acima. Faltam um módulo executável standalone, validação com outras entradas/arquiteturas e uma contagem de todos os operadores aritméticos executados.

## Medição exploratória do fixture

Ambiente: macOS arm64, Node 24.13.0, PyTorch 2.12.1, Transformers 5.5.0, CPU eager F16; checkpoint `artifacts/tiny-random-llama`. A fonte reduzida tem 15.309.179 caracteres e o JSON serializado 15.406.007 bytes, contra 87.655.527 caracteres antes da fatoração por operação. Para os IDs `[1,17,109,31999]`, uma execução mediu 0,18 s para compilar o checkpoint em fonte, 0,08 s para preparar JavaScript e 0,31 s para um forward de quatro tokens com contadores ligados; a mediana de dez forwards PyTorch após aquecimento foi 1,108 ms. As medições são de processos distintos e não são um benchmark estatístico estabilizado.

Em geração gulosa sem KV cache a partir de `[1]`, ambos produziram `[5971,10641,15242,14848]`. No Node padrão, os quatro forwards escalares levaram aproximadamente 165 ms, 91 ms, 146 ms e 304 ms (706 ms no total), contra 23,9 ms no PyTorch em uma execução que incluiu seu primeiro forward frio. A pausa de 243 s vista antes da fatoração deixou de ocorrer nesse teste. Para o forward escalar de quatro tokens, os contadores registraram 4.557.264 chamadas de `Math.fround`, 2.087.936 de `f16`, 1.152 de `Math.exp` e 80 de `Math.sqrt`. Esses números contam chamadas explícitas das funções geradas; ainda não incluem cada operador `+`, `*` e `/` executado.

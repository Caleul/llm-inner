# Forward fixo com arredondamento explícito

## Contrato implementado

`src/fixed-f16-projection.ts` lê uma matriz F16 densa de um diretório
Safetensors com `config.json`. Cada linha vira uma função escalar independente:
os pesos são substituídos por seus bits originais e termos com peso zero são
eliminados. Cada produto e cada adição do acumulador são arredondados para F32,
em ordem crescente de índice; a saída passa por uma árvore de decisões que
escolhe o F16 adjacente. O empate usa o bit par; overflow vai a infinito. Não há
dependência do checkpoint ao avaliar o programa compilado.

`src/fixed-f16-scalar-functions.ts` converte cada linha em uma expressão
escalar e substitui integralmente os inputs de uma projeção pelas funções da
anterior. Para o caso de um token, a composição `V -> O` produz 16 funções
somente dos 16 valores de entrada, sem referência a Q/K/V, nomes de matrizes ou
checkpoint. Os arredondamentos F32 da soma e F16 da fronteira entre as
projeções continuam explícitos. Um produto de dois valores F16 finitos cabe
exatamente em F32, por isso seu arredondamento F32 redundante é eliminado.

O `if/else` é implementado por busca binária sobre os valores F32/F16 positivos,
seguida de comparação com o ponto médio exato dos vizinhos. Isso representa
as decisões sem materializar dezenas de milhares de ramos no artefato. Valores
não finitos nos pesos ou na entrada são recusados nesta primeira versão.

## Evidência e limite

O checkpoint `HuggingFaceM4/tiny-random-LlamaForCausalLM`, revisão
`a4e23ca0227158bb70c644bd81f9ffb93120d96e`, foi baixado em
`artifacts/tiny-random-llama/` (ignorado pelo Git). SHA-256 do Safetensors:
`d1de8b249c2069f0a63b456b69c9b56146a2292a2aca86f094000f167162fccc`.
O teste automatizado compara
os 16 bits de saída da `q_proj` da camada 0 para uma entrada F16 fixa com a
referência PyTorch. Um fixture capturado do forward real de dois tokens (`[1,2]`)
compara Q/K/V nas duas camadas: 192 de 192 coordenadas coincidiram bit a bit.
A versão anterior, que arredondava apenas a soma final, divergiu em uma
coordenada de `v_proj` da camada 0 (1 ULP F16). O arredondamento F32 do
acumulador reproduziu a fronteira observada. Em uma comparação isolada,
`q_proj`, `k_proj`, `v_proj` e `o_proj` das duas camadas coincidiram em 128 de
128 coordenadas para uma mesma entrada artificial. Para um token, a softmax da
atenção tem uma única chave e vale 1; um segundo fixture confirma a composição
real `V -> O` nas duas camadas, incluindo a concatenação das quatro heads, com
igualdade bit a bit na entrada e na saída de `O`. Um teste sintético cobre
substituição por dimensão e composição de duas projeções, com arredondamento
entre elas. Essa expansão ainda aumenta o tamanho: as expressões separadas
`V` e `O` têm 1.280 nós cada, enquanto as 16 funções substituídas somam 21.504
nós (cerca de 811 kB em JSON). Portanto a substituição está demonstrada, mas
**a compressão de tamanho e a aceleração ainda não estão demonstradas**. A
simplificação adicional precisa ser medida sem remover fronteiras numéricas
nem alterar outputs.

Esses resultados **não** provam igualdade para qualquer entrada ou para a
atenção com dois ou mais tokens, que precisa de RoPE, máscara e softmax não
trivial. A hipótese observada para estas projeções é acumulação F32
sequencial; outros kernels podem usar FMA ou outra ordem. Nesses casos a árvore
precisa incluir as fronteiras do kernel escolhido. A expressão atual não
representa RMSNorm, RoPE, scores, softmax, máscara, residual, MLP nem logits.
A semântica de zero com sinal e
NaN/Inf também precisa ser estabelecida antes de alegar paridade geral.

## Próxima extensão verificável

1. Capturar RoPE, scores, probabilidades e saída da atenção no forward de dois
   tokens, inclusive dtype, ordem das reduções e máscaras.
2. Incorporar RoPE, produto QK, máscara, softmax e produto AV, com operações
   de arredondamento explícitas; comparar os bits em cada fronteira.
3. Conectar o resultado da atenção à projeção `o_proj`, residual,
   RMSNorm e MLP. Repetir a comparação em cada fronteira da camada 0.
4. Usar a saída validada da camada 0 como entrada da camada 1. Só então
   compor a função dos logits e medir tamanho, custo e paridade por token.

O critério é igualdade bit a bit nas fronteiras cujo runtime foi caracterizado.
Quando o backend não garante a mesma ordem ou função transcendente, registrar
a diferença e o backend de referência em vez de declarar equivalência.

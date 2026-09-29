# Fechamento do forward F16 fixo — 2026-09-29

## Estado verificado após a implementação

- Checkpoint local: `artifacts/tiny-random-llama`, Llama de duas camadas, hidden size 16, quatro heads de dimensão 4, MLP intermediária 64, vocabulário 32.000, F16.
- `fixed-f16-attention-program.ts` compila Q/K/V, RoPE, QK, máscara, softmax, AV e O num DAG escalar compartilhado com 32 raízes de saída. A avaliação recebe apenas as 32 entradas F16 da camada; pesos e RoPE são literais. Cada nó refere somente nós anteriores. A softmax é um operador binário do DAG com cálculo interno de máximo, exp, soma e divisão, ainda não expandido em nós elementares.
- `fixed-f16-two-token-model.ts` compõe embedding, as duas camadas completas, norm final e `lm_head`. A avaliação recebe apenas dois IDs de token; a saída calculada da camada 0 alimenta a camada 1. RMSNorm e MLP são programas por estágio dentro desse executor, ainda não substituídos num único DAG por dimensão de logit.
- A referência completa foi capturada para `[1,2]`, `[0,1]`, `[5,7]` e `[22144,30091]`, com PyTorch 2.12.1 e Transformers 5.5.0 CPU eager. Outros 32 pares determinísticos cobrem IDs de todo o vocabulário, com SHA-256 dos 64.000 logits F16 e de saídas ocultas. Os hashes e o próximo token coincidiram nos 32 pares. Isto prova os casos capturados, não equivalência universal.
- As projeções CPU deste fixture exigem quatro acumuladores F32 intercalados e fold par a par. A acumulação sequencial divergiu em 27 de 64.000 logits de `[1,2]`; também produziu uma divergência em `v_proj` para `[22144,30091]`. A ordem nova zerou essas divergências nos casos capturados.
- A softmax por intervalo racional agora cobre diferenças finitas até -128, com underflow explícito abaixo desse valor. Um fixture independente compara 212 pares F16 com PyTorch CPU, inclusive -1, -2, -10, -100 e -128. Diferenças especiais de NaN/Inf seguem fora do contrato.
- A versão compacta guarda embedding e `lm_head` como payloads F16 base64. O JSON compilado caiu de 20.959.582 para 3.411.616 bytes, mantendo os resultados dos 32 pares. O checkpoint Safetensors tem 2.066.960 bytes.

## Medição local

`node scripts/benchmark-fixed-forward.mjs` faz três aquecimentos e vinte execuções de `[1,2]` em cada runtime, excluindo carregamento e compilação do tempo por forward. Uma execução registrou mediana de 117,95 ms em Node.js para a função fixa e 1,24 ms no PyTorch CPU eager; o próximo token foi 20141 em ambos. O número é específico desta máquina e desta execução. As contagens de nós e termos estão em `FIXED-FORWARD-METRICS.json`; não são uma contagem completa de instruções CPU nem equiparam automaticamente um nó escalar a uma operação vetorizada.

## Trabalho restante para a formulação literal máxima

1. Substituir os operadores internos de softmax, RMSNorm e SiLU em um único DAG por coordenada de logit, preservando as fronteiras F16/F32 e a estrutura compartilhada. O executor completo atual já não recebe intermediários, mas ainda usa programas por estágio internamente.
2. Ampliar o teste diferencial para mais comprimentos e geração autoregressiva além do primeiro token previsto a partir de dois IDs. O programa atual é fixado a duas posições e mede o argmax de cada uma; ele não calcula o passo de três tokens.
3. Investigar paridade em outros kernels, máquinas, NaN/Inf, zero com sinal e casos próximos dos pontos médios transcendentais. A igualdade de `exp` corretamente arredondado com todo build de PyTorch não é uma propriedade provada pelos 212 casos.
4. Reduzir ainda mais o tamanho e custo. A compactação reduziu bytes, mas o executor escalar permanece muito mais lento que o forward vetorizado.

## Critério de conclusão

Para o escopo fixo de dois tokens, a entrada de dois IDs determina os logits das duas posições e o próximo token por argmax sem receber tensores intermediários do forward. A prova atual cobre 32 pares de IDs. O critério mais amplo de uma função escalar literal única por logit, fidelidade para outros comprimentos e geração de múltiplos tokens permanece aberto. As comparações de tamanho, contagem e tempo se referem somente ao artefato e ao forward identificados acima.

# Gemma 4 E4B: programa global real simplificado

## Resultado

O artefato `artifacts/gemma4-e4b-dense.literal.json` foi regenerado no schema
61 com duas representações simultâneas:

- o programa IEEE existente, mantido como baseline reproduzível; e
- `realSimplifiedProgram`, com semântica
  `gemma4-exact-real-simplified-v1`, sem arredondamento intermediário.

O artefato final possui `21.680.670.333` bytes e SHA-256
`869e9081c23a5cf150891dc50878a4dc7d2f3dd9cffe57c9982e07ffe4e23001`.
A seção `realSimplifiedProgram` possui `269.892.666` bytes canônicos, SHA-256
`4a422034fc955fc9ff558368820ce2c6cfd24ccafe390ddee700debaffce7ff8`,
e é vinculada pelo manifesto estrutural.

## Closure e cobertura

- nós de expressão content-addressed: `542.655`;
- funções de operação: `1.321`;
- funções públicas: `264.704`;
- dimensões finais de hidden state: `2.560`;
- logits terminais: `262.144`;
- atribuições do grafo fonte: `2.708`;
- leituras paramétricas de tabelas racionais aprendidas: `662`;
- reduções `runtime-defined` convertidas em soma real finita: `100`;
- reduções não resolvidas: `0`;
- nós de arredondamento IEEE intermediário: `0`.
- quantização das 264.704 fronteiras públicas: BF16 RNE.

As raízes públicas chamam definições content-addressed. Essa composição é a
forma serializável da substituição global: expandir fisicamente todas as
chamadas repetiria exponencialmente as mesmas subexpressões e excedeu o heap de
4 GiB, sem alterar a função matemática denotada.

## Regeneração

```bash
npm run build
node --max-old-space-size=6144 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

O reader abriu e validou o artefato somente pelos bytes incorporados. O
manifesto confirmou a seção simplificada, as 264.704 raízes e toda a cobertura
acima.

## Diferencial inicial

Foi compilada, a partir do artefato source-removed, uma amostra de 32 saídas de
`layer_0_q_proj` sobre um vetor BF16 determinístico de 2.560 entradas. O
compilador autenticou 81.920 escalares aprendidos (163.840 bytes), substituiu
todos os pesos por racionais e avaliou a soma em F64 sem casts intermediários.

Comparação contra o kernel IEEE paginado declarado, após o único
arredondamento BF16 final:

- 32/32 valores idênticos (`taxa de divergência = 0%`);
- erro absoluto máximo final: `0`;
- erro absoluto médio final: `0`;
- maior diferença observada antes da quantização final na amostra:
  `0,0006163858086944973`.

Esse resultado valida a mecânica e mostra que, nessa amostra local, o erro cru
foi absorvido pelo arredondamento final.

## Logits, argmax e geração

O script `scripts/gemma4-real-differential.py` executa as camadas textuais com
operações tensoriais F64 a partir das mesmas fronteiras BF16 de embedding e
PLE, promove os pesos BF16 do checkpoint identificado pelo artefato, remove os
casts F32/BF16 intermediários e arredonda somente os logits terminais para
BF16. O baseline é Transformers eager BF16 com `use_cache=true`, inclusive o
compartilhamento KV declarado. F64 é usado como aproximação computável dos
reais; não é apresentado como aritmética real simbólica infinita.

Nos quatro prompts naturais abaixo, cada comparação cobriu os 262.144 logits:

- `Hello` (`[2,9259]`);
- `The capital of France is` (`[2,818,5279,529,7001,563]`);
- `A soma de dois mais dois é` (`[2,236776,105017,569,25047,4522,25047,1559]`);
- `Once upon a time` (`[2,14946,3324,496,990]`).

Resultado agregado:

- argmax igual: `4/4` (`100%` na amostra);
- divergência média de valores BF16: `85,68382263183594%`;
- erro absoluto máximo: `0,5`;
- erro absoluto médio entre prompts: `0,05749302437804005`.

Também foi executado greedy de três passos, recomputando o prompt completo a
cada passo, para `The capital of France is`. Baseline e candidato produziram a
mesma sequência `[496,3207,600]`; a divergência média dos logits foi
`88,66221110026041%` e o erro absoluto máximo foi `0,5`.

Portanto remover os arredondamentos não preserva os logits bit a bit, mas nesta
amostra preservou a decisão de argmax e a sequência greedy. Isso é uma taxa
medida, não prova universal: prompts maiores, modalidades e gerações longas
continuam podendo cruzar uma fronteira decisória.

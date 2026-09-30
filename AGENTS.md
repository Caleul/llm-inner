# Direção do projeto

Este projeto transforma um checkpoint Safetensors e sua configuração em **funções de cálculo explícitas**, independentes do checkpoint na execução. A função final deve calcular o valor de cada dimensão a partir da entrada, para um número variável `n` de tokens. Os índices de posição e dimensão são parâmetros da função; não são valores fixados por um prompt de teste.

## Sequência de trabalho

1. Leia a configuração e os tensores do modelo. Identifique a arquitetura e as operações efetivamente usadas; rejeite uma operação cuja semântica numérica ainda não esteja definida.
2. Exploda a primeira camada em cálculos escalares por posição e dimensão. Substitua cada acesso a peso por seu valor literal decodificado do Safetensors, preservando dtype, ordem das reduções e pontos de arredondamento.
3. Substitua as expressões da camada nas expressões consumidoras. Simplifique imediatamente com identidades, eliminação de zeros, fatorações e trocas equivalentes **somente quando preservarem o valor sob a política numérica declarada**. Expresse escolhas e limites por condicionais `if/else` quando necessário.
4. Avance para a camada seguinte usando a expressão já substituída e simplificada da camada anterior. Repita até a normalização e a projeção finais. O resultado deve ser uma função por dimensão final, parametrizada por `n` e pela posição solicitada, sem receber ativações intermediárias.
5. Valide o **valor final de cada dimensão**, não apenas o ID do token escolhido. Compare com o forward de referência para várias entradas e comprimentos de sequência, localize a primeira divergência numérica e ajuste a transformação antes de prosseguir. Verifique também que a função gerada executa sem consultar o Safetensors original.
6. Só depois da paridade final, meça o tamanho da função, o número de operações, o tempo e os tokens produzidos. Registre o modelo, runtime, backend, entradas e limites da equivalência observada.

## Caminho incorreto já observado

O experimento `fixed-f16-attention-program.ts` / `fixed-f16-two-token-model.ts` compilou um DAG para **exatamente dois tokens** e validou logits e IDs em fixtures. Mesmo com paridade nesses casos, ele mantém uma arquitetura de nós, programas por estágio e um executor que percorre ativações intermediárias. `fixed-f16-logit-audit.ts` apenas expõe outra vista desse DAG. Compactar seus pesos ou medir seu tempo não transforma esse caminho na função pedida. Trate esses arquivos e fixtures como material diagnóstico, não como a solução arquitetural nem como prova de generalidade.

Não substitua a função final por DAG, árvore de etapas, lista de operações de camada, chamada opaca a atenção/MLP, wrapper do forward, lookup de saídas conhecidas ou função especializada para `[1,2]` ou qualquer `n` fixo. Não declare conclusão com base em argmax, hashes isolados, testes de uma única sequência ou igualdade somente em fronteiras intermediárias.

## Critério de aceitação

Para uma arquitetura suportada, o compilador parte dos arquivos Safetensors, gera funções por dimensão com pesos concretos e cálculo parametrizado pelo comprimento da entrada, e essas funções reproduzem os valores finais do forward nas condições numéricas declaradas. A implementação não depende de um grafo ou de ativações de camada na execução. Quando a expansão ou uma equivalência numérica não estiver demonstrada, registre exatamente o limite e mantenha a etapa como pendente.

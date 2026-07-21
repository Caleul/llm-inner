# Gemma 4: arquitetura do runtime compilado

## Objetivo

O produto não executará 262.144 árvores independentes. O compilador conserva um
DAG global, calcula o tronco de 42 camadas uma vez por token e expõe cada
`calc_final_n` como uma raiz navegável do mesmo programa. A cabeça terminal é
dividida em tiles de logits para execução multithread.

```text
prompt -> tokenizer -> embeddings e PLE -> tronco compartilhado
       -> hidden final[2560] -> tiles de logits[262144]
       -> argmax -> token -> cache -> próximo passo
```

Expandir cada dimensão como uma árvore textual duplicaria as mesmas atenções,
normalizações e MLPs 262.144 vezes. A string plana continua sendo uma visão de
auditoria; o formato executável é o DAG com common-subexpression elimination.

## Closure real da dimensão zero

`terminal_logit[0]`, para batch zero, sequência final cinco e comprimento seis,
foi recomposto em 1.221 passos e depois fechado em um único vetor externo:

- raiz: `sha256:9f3e49cd04a0224b649b2e49fd26882e49ddef80776329216f0efd840a601e6e`;
- nós: 1.684.006;
- chamadas de função restantes: zero;
- `x` possui 79.872 valores;
- `x[0..15359]`: `hidden_states_0`, shape `[1,6,2560]`;
- `x[15360..79871]`: `ple_inputs`, shape `[1,6,42,256]`;
- posição e máscaras full/sliding foram dobradas no grafo;
- nenhuma folha de input diferente de `x` permaneceu;
- 36.206 nós aprendidos ainda apontam para o constant pool imutável.

O arquivo `artifacts/gemma4-terminal-logit-0.x-closed.json` tem 622.557.503
bytes e SHA-256
`81e74ffa4e763c98952bde94338de3a2709a9cac9f947dc0cb1e033cc587b831`.

## Etapas do compilador

1. fechar posições, máscaras e demais entradas estruturais;
2. empacotar os pesos como constant pool autenticado, fora do vetor `x`;
3. fazer CSE global entre todas as dimensões finais;
4. baixar reduções para kernels vetoriais fundidos;
5. agendar tiles independentes da cabeça de logits em múltiplas threads;
6. preservar cache KV entre passos autoregressivos;
7. integrar tokenizer e detokenizer;
8. comparar o runtime compilado com Transformers para qualquer prompt.

## Métricas obrigatórias

A interface deve registrar por passo: igualdade do token, estabilidade do
argmax, divergência e erros dos logits, latência original/compilada, aceleração,
threads, tokens por segundo e memória. Nenhuma aceleração é presumida: ela será
aceita somente a partir dessas medições. Simplificação algébrica reduz o IR, mas
não elimina automaticamente produtos densos separados por attention, softmax,
RMSNorm e ativações não lineares.

## Bundle global e runtime persistente

O bundle `artifacts/gemma4-compiled-global-runtime-bundle` contém:

- `global-formulas.ssa.json`: 430.678.285 bytes e todas as 262.144 raízes
  `calc_terminal_logit_n` sobre o DAG compartilhado;
- `formula.graph.json`: closure algébrica fechada da dimensão zero;
- `constants.literal.json`: constant pool autenticado;
- tokenizer, configurações e pesos de compatibilidade do backend PyTorch;
- `manifest.json`: tamanhos e SHA-256 de oito arquivos.

O caminho absoluto do constant pool no SSA foi substituído por
`constants.literal.json`. Assim, servidor, tokenizer e pesos podem ser abertos
diretamente do bundle sem consultar o diretório-fonte. Os pesos Safetensors
duplicam temporariamente o constant pool lógico para compatibilidade com
Transformers; a cópia local é clone-on-write e deixa de ser necessária quando o
backend tensorial consumir diretamente os bytes literais.

O servidor usa um worker JSONL persistente. Modelo e tokenizer são carregados
uma vez, múltiplos prompts reutilizam o mesmo processo, e prefill é seguido de
decode incremental com cache KV em ambos os caminhos. A interface permite
configurar threads, F32/F64 e checkpoints BF16 experimentais.

## Evidência de desempenho e fidelidade

Com `The capital of France is`, F32, uma thread e três tokens, original e
compilado produziram `[496,3207,600]`. O prefill compilado foi mais rápido nessa
execução, mas o decode incremental permaneceu mais lento. Uma varredura sem
cache residente mostrou `1,43x` para o candidato com uma thread e regressão com
mais threads; logo não há evidência de aceleração geral ainda.

Caches residentes de pesos foram rejeitados por medição:

- F64 materializou 36,94 GB e ficou mais lento;
- F32 materializou 18,47 GB, elevou o pico para 27,89 GB e atingiu apenas
  paridade instável depois do aquecimento.

Com `Write one short sentence about the Moon:`, o primeiro token coincidiu, mas
o segundo divergiu (`818` no original e `3689` no compilado) tanto em F32 quanto
em F64. BF16 por camada e por operação não corrigiram essa decisão. Portanto,
remoção de arredondamentos ainda não preserva qualidade para qualquer prompt;
o próximo backend deve calibrar checkpoints sensíveis e medir uma matriz ampla
de prompts antes de selecionar sua política padrão.

A matriz persistida `artifacts/gemma4-compiled-calibration-f32.json` executou
oito prompts, com dois tokens por prompt, F32, uma thread e sem arredondamentos
intermediários. Seis prompts mantiveram toda a sequência, 14 de 16 decisões de
argmax coincidiram e a mediana de desempenho foi `1,19x` (`0,70x` a `2,31x`).
A divergência média elemento a elemento dos logits foi `88,51%` e o erro
absoluto máximo foi `0,71875`. A política experimental BF16 por operação repetiu
os mesmos 14 de 16 argmax e não recuperou os dois prompts divergentes. Esses
números são evidência de funcionamento ponta a ponta, não de equivalência de
qualidade nem de aceleração universal.

O caminho chamado de compilado na comparação ainda é um backend de
compatibilidade: ele recompõe as operações sem arredondamentos usando os pesos
do bundle e PyTorch. O arquivo SSA e o constant pool são empacotados e
autenticados, mas ainda não são interpretados diretamente por um executor
próprio. Remover `model.safetensors` e medir a execução do DAG são condições
necessárias para alegar a LLM efetivamente compilada no formato final.

A UI real foi exercitada via Playwright contra o bundle. Ela gerou dois tokens,
mostrou texto, ids, igualdade do argmax, divergência dos logits, erro máximo,
tokens/s, razão de desempenho, threads e pico RSS.

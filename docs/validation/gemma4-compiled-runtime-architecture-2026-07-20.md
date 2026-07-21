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

## Primeiro replay sem checkpoint

O executor existente `gemma4-paged-text.ts` foi aplicado ao artefato literal
real com a entrada mínima `[2]`. Ele acessou somente
`artifacts/gemma4-e4b-dense.literal.json`, produziu logits completos de shape
`[1,1,262144]` e caches das 24 camadas produtoras de KV. O relatório declara
explicitamente `sourceCheckpointAccessed: false`.

Esse replay levou `126.290 ms` e atingiu `2.639.136 KiB` de RSS máximo. A causa
observada está em `pagedLinearF32`: produtos densos são executados como loops
escalares JavaScript, embora os pesos sejam corretamente lidos em páginas.
Conectar esse caminho diretamente à interface provaria independência do
checkpoint, mas violaria o objetivo de desempenho. O próximo backend deve
preservar o leitor literal e substituir os lineares por tiles enviados a um
kernel nativo persistente e multithread; a mesma API paginada continua sendo a
fronteira de constant pool.

O replay foi repetido com relatório de seleção terminal. Ele retornou
`terminalArgmax: 184`; Transformers BF16 também selecionou `184`. Os dez maiores
logits coincidiram valor por valor, inclusive o empate `18,875` dos tokens `184`
e `198`. A repetição levou `126.147 ms` e `2.742.592 KiB` de RSS máximo. Isso
prova paridade do topo para esta entrada mínima, mas não amplia essa evidência
para prompts arbitrários nem resolve o status de fidelidade não verificada
declarado pelo artefato.

## Kernel nativo e constant pool binário

Um worker PyTorch persistente passou a receber tiles por protocolo binário. O
Node conserva a leitura indexada e a validação de tensor; o worker executa
`F32 input × BF16 weight` depois de ampliar o peso dentro do runtime nativo. O
canal aceita threads configuráveis e não serializa arrays numéricos como JSON.

Resultados para `[2]`, sempre com `argmax 184`:

- JSON base64 + linear escalar: `126,15 s`;
- JSON base64 + tile PyTorch F32: `70,65 s`;
- pool Safetensors indexado + tile F32 expandido no Node: `22,21 s`;
- pool Safetensors indexado + bytes BF16 crus: `7,96 s`.

O último caminho é `15,85x` mais rápido que o replay escalar. Uma página de
64 MiB regrediu para `9,30 s`; 16 MiB permanece o melhor valor observado. A
entrada textual `[2,818,5279,529,7001,563]` (`The capital of France is`) gerou
`496` e, após o forward incremental com cache, selecionou `3207` como próximo
argmax. Isso coincide com `a city` no original. Os dois forwards levaram
`17,34 s` e o pico RSS foi `2.607.712 KiB`.

O pool binário atual é o `model.safetensors` interno do bundle e o relatório
marca `compatibilityBinaryWeightsAccessed: true`; ele não deve ser confundido
com eliminação dos pesos. O formato final deve manter apenas um pool binário
autenticado, eliminando a duplicação do payload base64 e o rótulo de
compatibilidade.

## Interface com três executores persistentes

O servidor agora abre dois processos de longa duração: o comparador
Transformers mantém modelo/tokenizer e o worker literal mantém programa,
constant pool e kernel linear. A mesma entrada tokenizada alimenta original,
compatibilidade sem arredondamentos e backend direto. Os IDs diretos retornam
ao tokenizer persistente para decodificação, fechando prompt → token IDs →
forward → argmax → KV cache → decode → texto.

Na interface real, `The capital of France is` produziu `[496,3207]` e
`The capital of France is a city` nos três caminhos. O direto levou `14,05 s`
para os dois tokens na execução visual aquecida. Uma segunda requisição, sem
reinicializar processos, gerou `four.` para `Two plus two equals` com IDs
`[2390,236761]` em todos os caminhos e `14,77 s` no direto. A UI apresenta
textos, IDs, igualdade, primeira divergência, tokens/s, tempo, threads e o
relatório JSON completo de cada caminho.

A UI real foi exercitada via Playwright contra o bundle. Ela gerou dois tokens,
mostrou texto, ids, igualdade do argmax, divergência dos logits, erro máximo,
tokens/s, razão de desempenho, threads e pico RSS.

## Calibração três-vias e pool binário mapeado

`npm run calibrate:gemma4-real` executa um corpus inteiro através da mesma API
persistente usada pela interface. O relatório preserva, por prompt e etapa,
IDs e texto dos três executores, primeira divergência, top logits, tempos de
forward, throughput e custo de inicialização. Isso evita usar dois prompts
isolados como evidência de equivalência global.

O primeiro perfil amplo revelou que o worker direto transportava todos os
tiles de peso do Node para o PyTorch em cada forward. O backend
`persistent-pytorch-mmap-f32-tile` agora valida nome, dtype e shape contra o
catálogo do pool compilado e envia somente vetor de entrada, shard e intervalo
binário. O processo PyTorch abre cada shard uma vez com `mmap`; nenhuma matriz
linear atravessa o pipe ou é materializada no JavaScript.

Com oito prompts e dois tokens por prompt, a matriz
`gemma4-three-way-calibration-8x2-mmap.json` registrou:

- executor direto igual ao baseline em 7/8 prompts e 15/16 posições de token;
- modo compatível igual em 6/8 prompts e 14/16 posições;
- throughput direto agregado de 0,5234 token/s contra 0,5453 token/s do
  baseline, razão 0,9599x;
- antes do `mmap`, o mesmo corpus direto entregava 0,1226 token/s e razão
  0,2441x, portanto a remoção do transporte de matrizes produziu 4,27x sobre
  o backend direto anterior;
- uma execução direta isolada de `[2]` preservou token `184`, os mesmos top
  logits e SHA-256 terminal nas passagens fria e repetida; o forward caiu de
  9,713 s para 0,755 s quando as páginas já estavam residentes.

A única divergência direta ocorreu no segundo token de
`Write one short sentence about the Moon:`. O baseline empatou os tokens `818`
e `3689` em `23,375`; seu desempate escolheu `818`. O cálculo direto produziu
`3689 = 23,375` e `818 = 23,25`, escolhendo `3689`. Portanto a amostra localiza
uma fronteira de arredondamento sensível a empate, não uma diferença grande de
distribuição, mas ela continua sendo divergência observável e não pode ser
declarada equivalente.

O resultado ainda não atende a meta de ser centenas de vezes mais rápido. O
próximo custo dominante é a conversão BF16 para F32 de praticamente todo o
pool dentro de cada forward e a fragmentação do grafo em muitas chamadas de
kernel. O próximo limite coerente é fundir mais do programa compilado dentro do
worker nativo sem reintroduzir arredondamentos intermediários implicitamente.

## Backend MLX/Metal experimental

O protocolo referenciado agora inclui, além do shard e intervalo binário, o
nome do tensor e a coordenada inicial do tile. Isso permite ao worker
`persistent-mlx-metal-mmap-f32-tile` localizar diretamente o tensor BF16 no
Safetensors, manter os shards carregados de forma preguiçosa e executar
`F32 × BF16 → F32` no Metal sem transportar matrizes pelo JavaScript.

Na validação isolada de `[2]`, MLX e PyTorch escolheram o token `184`. O MLX
reduziu o primeiro forward de `9,913 s` para `2,989 s`, mas o forward repetido
ficou em `0,838 s`, contra `0,783 s` no PyTorch. O SHA-256 terminal também
divergiu (`f91b...` contra `39dd...`) porque as reduções não são bit a bit
idênticas; por exemplo, o token `198` mudou de `18,75` para `18,875`.

No corpus completo `gemma4-three-way-calibration-8x2-mlx.json`, o MLX manteve
7/8 prompts e 15/16 posições iguais ao baseline, a mesma taxa de acordo do
backend direto PyTorch. A divergência passou a ser o segundo token da tradução
para português; o empate do prompt da Lua voltou a escolher o token do
baseline. Na repetição final persistida, o throughput direto agregado foi
`0,5023 token/s`, contra `0,8860 token/s` do baseline aquecido, razão
`0,5669x`. O PyTorch mmap havia alcançado `0,5234 token/s` em sua execução
separada; como o baseline variou com aquecimento e cache entre ensaios, a
comparação mais segura entre os backends é o throughput direto absoluto, não a
razão cruzada. O processo MLX também atingiu cerca de `2,5 GB` de RSS no ensaio
isolado. Por isso a interface aceita
`--direct-linear-backend mlx`, exibe o backend efetivo, mas preserva PyTorch
como padrão até que fusão de operações torne o Metal vantajoso no agregado.

Comando reprodutível:

```bash
npm run calibrate:gemma4-real -- \
  --source ./artifacts/gemma4-compiled-global-runtime-bundle \
  --output ./artifacts/gemma4-three-way-calibration-8x2-mmap.json \
  --tokens 2 --request-threads 1 \
  --direct-threads 8 --direct-max-read-mib 16 \
  --direct-fused-mlp off --direct-final-head f32 \
  --precision f32 --rounding-policy none
```

## Paginação independente da projeção terminal

Depois das fusões de attention, MLP e PLE, o `lm_head` passou a responder por
81 das 85 leituras lineares referenciadas de cada forward com páginas gerais de
16 MiB. Uma varredura isolada de 4, 8, 16, 24, 32 e 64 MiB preservou os tokens
`[184,3910]` e o SHA-256 terminal
`1f65253f2759208e828a710a3f8f574d713b5b0b170f1c6174f20aee95c4bdd0`.
Como alterar o limite global também mudava as quatro leituras do prelude, o
executor ganhou `finalHeadMaxReadBytes`: somente a operação `lm_head` usa esse
limite, enquanto todas as demais matrizes mantêm `maxReadBytes`.

Em dois workers persistentes, com seis pedidos `[2]` de dois tokens e o primeiro
descartado como aquecimento, `16/16 MiB` teve mediana total de `1,0088 s` e
`16/32 MiB` teve `0,9868 s`, redução de 2,18%. Ambos mantiveram os mesmos tokens
e hash terminal; os despachos lineares caíram de 170 para 90 por pedido. A
mediana do segundo forward foi praticamente estável (`0,4873 s` contra
`0,4870 s`), portanto a decisão também foi validada no corpus completo.

No corpus 8×2 sob a mesma revisão:

- controle `16/16 MiB`: `1,1607 token/s`, 7/8 prompts e 15/16 tokens;
- candidato `16/32 MiB`: `1,1713 token/s`, 7/8 prompts e 15/16 tokens;
- ganho de throughput do candidato: 0,91%;
- contagem por prompt de dois tokens: 170 → 90 despachos referenciados.

Assim, PyTorch usa por padrão 16 MiB para o grafo e 32 MiB apenas para o head.
MLX conserva o limite geral porque não participou desta calibração. As flags
`--max-read-mib` e `--final-head-read-mib` no worker, e
`--direct-max-read-mib` e `--direct-final-head-read-mib` no comparador, deixam
o A/B explícito. A API, os relatórios e a interface mostram ambos os valores.

Evidência persistida:

- `artifacts/gemma4-three-way-calibration-8x2-split-head-16-control.json`;
- `artifacts/gemma4-three-way-calibration-8x2-split-head-32.json`.

## Prelude PLE fundido experimental

As quatro lineares referenciadas que restam além do `lm_head` pertencem a uma
única projeção do hidden state para `[42,256]`, seguida por escala, reshape,
RMSNorm, soma com o embedding PLE do token e escala final. O modo
`fusedPlePreludeRounding` compõe essas seis operações numa travessia do worker.
Ele autentica a projeção `[10752,2560]` e a norma `[256]` no `mmap`, recebe os
dois vetores de entrada e devolve diretamente `ple_inputs`.

A primeira implementação ampliava e multiplicava os 55 MB de pesos de uma vez.
Como o `lm_head` já mostrara perda de localidade em matrizes grandes, uma segunda
implementação manteve uma única travessia de processo, mas reproduziu dentro do
worker os blocos de até 16 MiB usados pelo executor paginado.

No caso isolado `[2]` com dois tokens, o modo BF16 cacheado preservou tokens,
top logits e o SHA-256 terminal
`1f65253f2759208e828a710a3f8f574d713b5b0b170f1c6174f20aee95c4bdd0`.
As referências caíram de 170 para 162 e dois preludes fundidos substituíram as
oito páginas. O forward quente isolado mediu `0,5000 s`, contra `0,5106 s` sem
a fusão.

O corpus 8×2, porém, contradisse a amostra isolada:

- controle promovido, prelude `off`: `1,1907 token/s`;
- prelude `bf16` cacheado: `1,1662 token/s`;
- PLE e prelude `real`: `1,1666 token/s`;
- todos mantiveram 7/8 prompts e 15/16 tokens iguais ao baseline.

Logo, eliminar três travessias por forward não compensou o custo de compor a
projeção e o pós-processamento nesse worker. O prelude fica disponível como
experimento explícito em
`--direct-fused-ple-prelude off|bf16|real`, mas o padrão é `off`. Isso é
independente de `--direct-fused-ple`, cujo `bf16` continua promovido para as 42
camadas. A interface separa política e contagem do prelude para não confundir
uma redução estrutural com aceleração comprovada.

Relatórios:

- `artifacts/gemma4-three-way-calibration-8x2-fused-ple-prelude-bf16-experimental.json`;
- `artifacts/gemma4-three-way-calibration-8x2-fused-ple-prelude-real-experimental.json`.

## Subgrafo PLE completo por camada

Depois das fusões de attention e MLP, cada uma das 42 camadas ainda executava
o caminho de per-layer input em oito operações separadas:

```text
select_per_layer -> gate -> GELU -> multiply(ple_input) -> projection
                 -> RMSNorm -> residual -> layer_scalar
```

As duas projeções atravessavam Node/PyTorch separadamente, totalizando 84
despachos lineares por forward. O modo `fusedPleRounding` reconhece a cadeia
por dependências, shapes, função de ativação, norma, residual e scalar — não
por nomes fixos — e envia o hidden state, o slice PLE e quatro referências
`mmap` autenticadas para o worker multithread. O retorno é diretamente
`hidden_states_{layer+1}`.

As políticas disponíveis são:

- `off`: executor anterior, operação por operação;
- `bf16`: preserva todas as fronteiras BF16 declaradas dentro da cadeia;
- `real`: remove os arredondamentos internos da cadeia e conserva apenas as
  fronteiras externas.

Para `[2]` com dois tokens, `bf16` preservou `[184,3910]`, todos os top logits
e o SHA-256 terminal
`1f65253f2759208e828a710a3f8f574d713b5b0b170f1c6174f20aee95c4bdd0`.
As 168 projeções PLE separadas tornaram-se 84 subgrafos, e os despachos
lineares de referência caíram de 338 para 170. O forward quente caiu de
`0,5347 s` para `0,5106 s`.

No corpus 8×2:

- `bf16`: `1,1907 token/s`, 7/8 prompts e 15/16 tokens iguais ao baseline;
- `real`: `1,1731 token/s`, os mesmos 7/8 prompts e 15/16 tokens;
- controle anterior sem fusão PLE: `1,1194 token/s`;
- ganho de `bf16` contra esse controle: 6,37%;
- razão de `bf16` contra o baseline da própria execução: `2,1262x`.

O modo `bf16` foi promovido como padrão PyTorch porque é bit a bit equivalente
no teste isolado, não adicionou divergência ao corpus e foi mais rápido que
`real`. MLX permanece em `off`. A interface mostra `fusedPleRounding` e
`fusedPleDispatches`; `--direct-fused-ple off|bf16|real` mantém o A/B.

O perfil promovido agora possui somente 85 lineares referenciadas por forward:
81 páginas do `lm_head` e quatro operações do prelude. O PLE de camada deixou
de aparecer nessa contagem. Somando 42 MLPs, 42 attentions e 42 PLEs fundidos,
o runtime executa aproximadamente 211 travessias do worker por forward, contra
253 antes desta fusão.

Comando promovido:

```bash
npm run calibrate:gemma4-real -- \
  --source ./artifacts/gemma4-compiled-global-runtime-bundle \
  --output ./artifacts/gemma4-three-way-calibration-8x2-fused-ple-bf16.json \
  --tokens 2 --request-threads 1 \
  --direct-threads 8 --direct-max-read-mib 16 \
  --direct-linear-backend pytorch --direct-fused-mlp real \
  --direct-fused-ple bf16 --direct-final-head native-bf16 \
  --direct-native-attention real --direct-fused-attention bf16 \
  --precision f32 --rounding-policy none
```

Para reproduzir a variante MLX/Metal, acrescente
`--direct-linear-backend mlx`; o relatório registra a implementação efetiva em
`initialization.direct.linearBackend` e a seleção em
`configuration.directLinearBackend`.

## Fusão de projeções com entrada compartilhada

O executor reconhece agora duas operações lineares adjacentes com a mesma
entrada, shape e paginação. No Gemma 4 isso cobre `gate_proj + up_proj` em cada
MLP. As duas saídas continuam sendo tensores independentes e preservam suas
fronteiras F32, mas o vetor de entrada, as identidades do pool e os pedidos dos
tiles atravessam o protocolo em um único lote. PyTorch agenda os dois GEMMs no
mesmo ciclo do worker; MLX avalia os dois resultados em uma única barreira.

O relatório direto expõe `linearReferenceDispatches`,
`linearBatchDispatches` e `linearBatchedProjectionTiles`, e a interface mostra
os lotes efetivamente usados. Para dois tokens por prompt foram 966 despachos
isolados, 336 despachos em lote e 672 tiles de projeção dentro desses lotes.
Sem a fusão seriam 1.638 despachos; com ela são 1.302, redução de 20,5%.

A calibração `gemma4-three-way-calibration-8x2-batched.json` preservou os
mesmos 7/8 prompts e 15/16 decisões de token do backend PyTorch anterior. O
SHA-256 terminal isolado de `[2]` permaneceu
`39ddaa04d2e4b36d66f72ba1be4db9420387a48a05f6757ae9683e67a4fe73cf`.
O throughput agregado, porém, ficou em `0,5221 token/s`, contra
`0,5234 token/s` antes: diferença compatível com ruído. Portanto o transporte
foi reduzido, mas os GEMMs continuam sendo o custo dominante. Uma tentativa de
concatenar os dois pesos em um único GEMM MLX também foi descartada porque
piorou o forward quente isolado de `0,789 s` para `1,140 s`, apesar de manter o
mesmo hash. O próximo limite deve fundir a região MLP inteira — projeções,
GELU, multiplicação e `down_proj` — evitando materializar intermediários fora
do worker.

## Subgrafo MLP compilado

O runtime reconhece agora a sequência estrutural
`gate_proj → GELU(tanh)`, `up_proj`, multiplicação e `down_proj` e a executa
como uma única chamada nativa por camada. O protocolo transmite somente o
vetor de entrada e as identidades autenticadas das três matrizes. Gate, up,
ativação e produto deixam de ser materializados no JavaScript; apenas o vetor
final da MLP retorna ao executor.

Há três políticas A/B:

- `off` mantém as operações separadas;
- `bf16` reproduz as quatro fronteiras BF16 internas;
- `real` elimina os arredondamentos internos e avalia a composição matemática
  em F32, arredondando somente nas fronteiras posteriores do grafo.

O modo BF16 preservou, para `[2]`, o SHA-256 terminal anterior
`39ddaa04d2e4b36d66f72ba1be4db9420387a48a05f6757ae9683e67a4fe73cf`.
O forward quente caiu de `0,7597 s` para `0,6393 s`. O modo real escolheu o
mesmo token `184`, mas produziu o hash `35dc338f...afb81f`, evidenciando a
remoção das fronteiras. Seu forward quente foi `0,6364 s`.

Nos oito prompts × dois tokens:

- `off`: `0,5221 token/s`, 7/8 prompts e 15/16 tokens iguais;
- `bf16`: `0,7272 token/s`, os mesmos 7/8 e 15/16, ganho de 39,3%;
- `real`: `0,7314 token/s`, os mesmos 7/8 e 15/16, ganho de 40,1% e razão
  `1,3141x` contra o baseline medido na mesma execução.

A única divergência dos dois modos fundidos permaneceu no segundo token de
`Translate to Portuguese: Good morning` (`1217` no baseline, `564` no
direto). Cada geração de dois tokens passou de 1.302 despachos lineares para
630 despachos lineares mais 84 subgrafos MLP, redução de 45,2% nas chamadas ao
worker. O RSS isolado do processo direto aumentou de aproximadamente 280 MB
para 807 MB porque três matrizes completas são ampliadas temporariamente para
F32. Esse é um trade-off explícito de memória por throughput.

Naquela etapa, o modo `real` foi promovido por corresponder à simplificação sem
arredondamentos internos e não reduzir o acordo de tokens. Ele continua como
controle matemático explícito. A interface mostra `fusedMlpRounding` e
`fusedMlpDispatches`; o worker MLX também executou o subgrafo real, preservou o
token `184` e produziu hash próprio `aa9d1d10...03669e04`.

Comando da MLP real com head F32 usado na comparação acima:

```bash
npm run calibrate:gemma4-real -- \
  --source ./artifacts/gemma4-compiled-global-runtime-bundle \
  --output ./artifacts/gemma4-three-way-calibration-8x2-fused-mlp-real.json \
  --tokens 2 --request-threads 1 \
  --direct-threads 8 --direct-max-read-mib 16 \
  --direct-linear-backend pytorch --direct-fused-mlp real \
  --direct-final-head f32 \
  --precision f32 --rounding-policy none
```

## MLP compilada sobre storage BF16 nativo

A telemetria acumulada por classe mostrou que, depois do aquecimento, a MLP
real ainda consumia cerca de `0,61 s` dos `1,00 s` necessários para dois tokens.
O motivo era estrutural: cada uma das três matrizes BF16 de cada camada era
ampliada integralmente para F32 antes dos GEMMs. O modo `native-bf16` mantém os
pesos diretamente no `mmap`, converte somente o vetor de entrada, executa
`gate_proj`, `up_proj`, GELU aproximada, produto e `down_proj` dentro de uma
única chamada PyTorch BF16 e devolve o resultado ampliado para F32. Nenhuma
cópia F32 completa dos pesos é materializada.

Em quatro pedidos persistentes `[2]` de dois tokens, o primeiro serviu de
aquecimento. Nos pedidos seguintes, a classe MLP caiu de aproximadamente
`0,61 s` para `0,135 s`; o pedido completo caiu de aproximadamente `1,00 s`
para `0,51–0,53 s`. Os tokens continuaram `[184,3910]`. O hash terminal mudou
de `1f65253f...c4bdd0` para
`9ff19b2b063fc96c7cec7e970749afc2b7b8cbc4611c3a894b60e069be2c582d`,
registrando que o kernel nativo possui ordem de acumulação distinta do modo
real.

No corpus 8×2 instrumentado sob a mesma revisão:

- controle `real`: `1,1694 token/s`, 7/8 prompts e 15/16 tokens;
- `native-bf16`: `1,5618 token/s`, os mesmos 7/8 prompts e 15/16 tokens;
- ganho direto: 33,55%;
- razão do nativo contra o baseline da própria execução: `2,8140x`;
- tempo MLP acumulado: `6,0648 s → 2,5183 s`, redução de 58,48% incluindo o
  primeiro prompt frio.

`native-bf16` passa a ser o padrão PyTorch porque preservou a taxa de acordo e
remove a ampliação que dominava o runtime. O modo `real` permanece disponível
para a função sem arredondamentos intermediários; `bf16` mantém o controle
anterior com boundaries explícitas; `off` conserva as operações separadas.
MLX permanece em `real`. A seleção é reproduzível por
`--fused-mlp off|bf16|real|native-bf16` no worker e
`--direct-fused-mlp ...` no comparador.

Os relatórios e a interface agora também expõem tempo acumulado de referências,
attention, MLP e PLE. Evidência persistida:

- `artifacts/gemma4-three-way-calibration-8x2-real-mlp-profile-control.json`;
- `artifacts/gemma4-three-way-calibration-8x2-native-bf16-mlp.json`.

## Subgrafo Q/K/V, RoPE, attention e projeção O

O perfil do runtime promovido mostrou 315 despachos lineares por forward:
146 pertenciam às projeções Q/K/V/O, 84 ao PLE, 81 ao `lm_head` e quatro ao
prelude. Somadas às 42 chamadas do núcleo nativo, as projeções e a attention
atravessavam a fronteira Node/PyTorch 188 vezes por forward.

O compositor reconhece agora, por dependências e shapes em vez de IDs fixos, a
região `Q → reshape → QNorm → RoPE`, o ramo produtor `K/V` quando presente, o
cache KV compartilhado quando não presente, `attention → O`. O worker recebe
os pesos como referências mmap autenticadas, posições, máscara e cache anterior;
devolve a projeção O e, somente para camadas produtoras, o cache completo
atualizado. Assim as 24 produtoras e as 18 consumidoras de KV compartilhado
usam uma chamada nativa cada.

Há três políticas A/B:

- `off` conserva as projeções separadas e o núcleo nativo anterior;
- `bf16` funde a região, mas preserva as fronteiras BF16 de Q/K/V,
  normalizações, RoPE e O, mantendo o núcleo de attention em F32 real;
- `real` remove também essas fronteiras BF16 dentro da região fundida.

Para `[2]` e dois tokens, `bf16` preservou os tokens `[184,3910]` e o SHA-256
terminal do controle,
`1f65253f2759208e828a710a3f8f574d713b5b0b170f1c6174f20aee95c4bdd0`.
Os despachos lineares caíram de 630 para 338, o núcleo separado de attention
de 84 para zero e o novo subgrafo executou 84 vezes.

No corpus de oito prompts × dois tokens:

- controle `off`: `1,0507 token/s`, 7/8 prompts e 15/16 tokens iguais;
- fundido `bf16`: `1,1173 token/s`, os mesmos 7/8 e 15/16, ganho de 6,34%;
- `bf16` atingiu `1,9678x` o baseline medido na mesma execução;
- fundido `real`: `1,1141 token/s`, mas caiu para 6/8 prompts e 14/16 tokens.

O modo real reintroduziu a divergência do segundo token de
`Write one short sentence about the Moon:` (`818` no baseline, `3689` no
direto), além da divergência já observada na tradução. Como a meta exige baixa
divergência, ele permanece experimental. `bf16` foi promovido como padrão
PyTorch: reduz as fronteiras de processo sem degradar o corpus atual. MLX
permanece em `off`. A interface exibe separadamente o núcleo e o subgrafo, suas
políticas, despachos e lineares restantes.

Comando promovido:

```bash
npm run calibrate:gemma4-real -- \
  --source ./artifacts/gemma4-compiled-global-runtime-bundle \
  --output ./artifacts/gemma4-three-way-calibration-8x2-fused-attention-bf16.json \
  --tokens 2 --request-threads 1 \
  --direct-threads 8 --direct-max-read-mib 16 \
  --direct-linear-backend pytorch --direct-fused-mlp real \
  --direct-final-head native-bf16 --direct-native-attention real \
  --direct-fused-attention bf16 \
  --precision f32 --rounding-policy none
```

## Head terminal BF16 nativo

O tensor compartilhado por embedding e `lm_head` possui shape
`[262144,2560]` e 1.342.177.280 bytes BF16. Ele é mais de 128 vezes maior que
cada projeção Q/O da primeira camada. O caminho anterior paginava esse tensor
em aproximadamente 80 tiles por forward, ampliava cada tile para F32 e então
executava o GEMM.

O modo `native-bf16` mantém os pesos em BF16, converte somente o vetor final
para BF16 e usa o GEMM BF16 nativo do PyTorch. O resultado volta como F32 para
softcap, ranking, hash e seleção do token. A mudança é limitada ao
`lm_head`, que já constitui a fronteira final de arredondamento permitida; as
funções compostas internas continuam no modo MLP `real`.

Para `[2]`, o token `184` e os top logits permaneceram, o forward quente caiu
de `0,6364 s` para `0,6015 s` e o hash terminal passou para
`ad4c001b...7cce850`, coerente com a redução BF16 final. Na matriz 8×2,
`gemma4-three-way-calibration-8x2-native-bf16-head.json` registrou:

- `0,7676 token/s`, 4,9% acima do head F32 fundido;
- 47,0% acima dos `0,5221 token/s` anteriores à fusão MLP;
- razão `1,3670x` contra o baseline na mesma execução;
- os mesmos 7/8 prompts e 15/16 tokens, sem nova divergência.

O primeiro passo da tradução reproduziu vários logits exatamente iguais ao
baseline. No segundo, `1217` e `564` empataram em `22,5` no direto e o critério
greedy escolheu o menor ID `564`; o baseline ainda separou os dois por `0,125`.
Isso mantém a divergência localizada antes do head terminal.

`native-bf16` foi promovido como padrão quando o backend é PyTorch; MLX mantém
head F32. A interface exibe `finalHeadCompute`, e
`--direct-final-head f32|native-bf16|native-bf16-whole` preserva o A/B
explícito. O terceiro modo é a variante integral experimental descrita abaixo.

Comando promovido:

```bash
npm run calibrate:gemma4-real -- \
  --source ./artifacts/gemma4-compiled-global-runtime-bundle \
  --output ./artifacts/gemma4-three-way-calibration-8x2-native-bf16-head.json \
  --tokens 2 --request-threads 1 \
  --direct-threads 8 --direct-max-read-mib 16 \
  --direct-linear-backend pytorch --direct-fused-mlp real \
  --direct-final-head native-bf16 \
  --precision f32 --rounding-policy none
```

## Head terminal BF16 integral experimental

O modo `native-bf16-whole` envia a matriz completa do `lm_head`, com shape
`[262144,2560]`, a uma única GEMM BF16 multithread por forward. O protocolo
continua referenciando o `mmap`: nenhum peso é serializado pelo JavaScript e a
saída permanece um vetor F32 de 262.144 logits. O modo anterior
`native-bf16` permanece disponível e divide a mesma matriz em 81 páginas de
até 16 MiB.

Para `[2]` com dois tokens, os dois modos produziram `[184,3910]` e o mesmo
SHA-256 terminal completo,
`1f65253f2759208e828a710a3f8f574d713b5b0b170f1c6174f20aee95c4bdd0`.
O integral reduziu os despachos lineares de 338 para 178: foram dois despachos
integrais no lugar de 162 páginas do head. O forward quente mediu `0,5310 s`
no integral e `0,5347 s` no controle paginado.

No corpus 8×2, ambos preservaram 7/8 prompts e 15/16 tokens contra o baseline.
Entretanto, o integral levou `14,8215 s` (`1,0795 token/s`) e o paginado
`14,2936 s` (`1,1194 token/s`). Assim, a chamada integral ficou 3,56% mais
lenta nesta máquina, apesar de reduzir 160 travessias de processo no corpus.
A inferência observada é que os tiles menores aproveitaram melhor cache e
localidade ao percorrer os 1,342 GB de pesos.

Por essa razão, `native-bf16` continua sendo o padrão PyTorch. O modo integral
é mantido como experimento reproduzível e aparece na interface com a métrica
`wholeNativeBf16Dispatches`; ele não é apresentado como aceleração comprovada.
Relatórios A/B:

- `artifacts/gemma4-three-way-calibration-8x2-whole-native-bf16-head.json`;
- `artifacts/gemma4-three-way-calibration-8x2-tiled-native-bf16-head-control.json`.

## Núcleo de attention nativo sem arredondamentos internos

O executor direto passou a reconhecer a operação compilada de attention como
um subgrafo único. Query, key, value e a máscara topológica materializada
atravessam o protocolo binário uma vez; o worker PyTorch executa
`QKᵀ → escala → máscara → softmax → PV` e devolve somente o contexto. Causalidade,
sliding window, cache KV e grouped-query attention continuam sendo derivados da
topologia declarada pelo programa literal, sem consultar o checkpoint-fonte.

Há três políticas reprodutíveis:

- `off` conserva o executor JavaScript;
- `bf16` reproduz as fronteiras BF16 internas do eager;
- `real` compõe score, softmax e contexto em F32, sem arredondamentos
  intermediários no subgrafo.

Na entrada isolada `[2]`, o modo `bf16` preservou byte a byte o hash terminal
promovido pelo head nativo,
`ad4c001ba081c40685082e45a2c88dfa0372915f966de052a9de80cee7cce850`, e
selecionou o token `184`. O forward quente caiu de `0,6015 s` para `0,5419 s`.
O modo `real` também selecionou `184`; com sequência unitária, onde softmax é
trivial, produziu o mesmo hash e levou `0,5686 s`. A execução do CLI sem flag
confirmou `nativeAttentionRounding: real` e 42 despachos nativos para um forward.

Nas matrizes 8×2:

- `bf16`: `1,0379 token/s`, 7/8 prompts e 15/16 tokens iguais ao baseline;
- `real`: `1,0585 token/s`, os mesmos 7/8 prompts e 15/16 tokens;
- `real` atingiu razão `1,8430x` contra o baseline medido na mesma execução;
- cada prompt de dois tokens executou 84 subgrafos nativos de attention.

A única divergência permaneceu no segundo token de
`Translate to Portuguese: Good morning`: `[236764,1217]` no baseline e
`[236764,564]` no direto. Como baseline, cache de páginas e estado térmico
variaram entre ensaios, os throughputs absolutos acima devem ser lidos junto
dos próprios relatórios, e não como uma garantia universal de aceleração.

O modo `real` foi promovido como padrão do backend PyTorch por corresponder à
composição matemática sem arredondamentos internos solicitada, não perder
acordo no corpus e ser o mais rápido dos dois modos nativos na repetição final.
MLX permanece em `off`; `--direct-native-attention off|bf16|real` mantém o A/B
explícito. A interface mostra a política e a contagem efetiva de despachos.

Comando promovido:

```bash
npm run calibrate:gemma4-real -- \
  --source ./artifacts/gemma4-compiled-global-runtime-bundle \
  --output ./artifacts/gemma4-three-way-calibration-8x2-native-attention-real.json \
  --tokens 2 --request-threads 1 \
  --direct-threads 8 --direct-max-read-mib 16 \
  --direct-linear-backend pytorch --direct-fused-mlp real \
  --direct-final-head native-bf16 --direct-native-attention real \
  --precision f32 --rounding-policy none
```

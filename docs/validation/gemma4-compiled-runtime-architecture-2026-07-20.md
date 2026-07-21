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

## Prelude PLE compilado com redução reproduzível

O backend MLX absorve agora o prelude PLE completo em uma única requisição
binária por forward: projeção de contexto, escala, reshape por camada, RMSNorm,
soma com a identidade aprendida por token e escala final. A interface ativa
`bf16` por padrão no backend MLX e publica tanto a política quanto a contagem e
o tempo desses despachos.

A primeira composição experimental executou a projeção `[10752,2560]` como uma
única GEMM e usou a redução genérica de `mx.mean`. Embora reduzisse o transporte,
ela alterou o segundo token do prompt sobre a Lua de `818` para `3689`. A versão
promovida mantém uma única fronteira externa, mas preserva internamente os
quatro blocos de projeção definidos por `maxReadBytes` e reproduz a árvore
`pytorch-cpu-f32-cascade-sum` declarada pelo artefato. No caso limítrofe, os
logits de `818` e `3689` voltaram a empatar em `23,375`; o desempate greedy pelo
menor token selecionou `818`, como no original.

Na calibração 8x2 final:

- o compilado preservou 8/8 prompts e 16/16 tokens do original;
- o throughput direto foi `4,9591 token/s`, `5,4130x` o baseline da mesma
  execução;
- contra o forward integral anterior (`4,6476 token/s`), houve ganho de 6,70%;
- o tempo agregado direto caiu de `3,4427 s` para `3,2264 s`, redução de 6,28%;
- 64 despachos lineares referenciados foram substituídos por 16 preludes PLE;
- o cache de constantes passou de 403 para 404 entradas e de `222.440.608` para
  `222.441.632` bytes, sem ampliar a matriz de projeção BF16 para F32;
- os forwards incrementais dos oito prompts ficaram entre `0,0792 s` e
  `0,0918 s`.

Evidência promovida:

- `artifacts/gemma4-three-way-calibration-8x2-mlx-ple-prelude-cascade.json`.

## Forward textual único de IDs de token até o vetor final

O caminho textual compilado passou a aceitar diretamente `input_ids`, posições,
máscaras e cache KV em uma única requisição binária. Dentro do worker persistente,
a mesma chamada executa:

1. lookup e escala BF16 do embedding textual;
2. lookup e escala BF16 da identidade PLE por camada;
3. projeção e normalização reproduzível do prelude PLE;
4. as 42 camadas decoder com RoPE, attention, MLP, PLE e cache;
5. norma final, vocabulary head e tanh softcap;
6. publicação somente do vetor de logits da última posição e do novo cache KV.

O embedding PLE possui `5.637.144.576` bytes, acima do limite de um `uint32`.
Por isso, a nova variante do protocolo usa comprimento `uint64` nos quatro
descritores do prelude, sem alterar os descritores de 36 bytes dos kernels
anteriores. Os dois embeddings são lidos por linhas BF16 autenticadas diretamente
do shard, arredondados RNE após a escala e mantidos em um LRU de no máximo 4.096
linhas efetivamente visitadas. Isso evita pedir ao Metal que indexe a matriz completa de
5,64 GB e não materializa o embedding em F32.

Como a geração greedy só consome `logits[:, -1, :]`, a chamada integral não
transporta mais hidden states nem logits das posições anteriores. No corpus 8x2,
o tráfego teórico de logits caiu de `79.691.776` para `16.777.216` bytes
(`-78,95%`), além de eliminar `778.240` bytes de hidden states.

Na calibração final 8x2:

- 8/8 prompts e 16/16 tokens do compilado coincidiram com o original;
- cada token executou exatamente um forward `token IDs -> logits/cache`;
- o throughput direto foi `5,9126 token/s`, `6,4307x` o baseline da mesma rodada;
- contra o prelude separado (`4,9591 token/s`), houve ganho de `19,23%`;
- o tempo agregado caiu de `3,2264 s` para `2,7061 s`, redução de `16,13%`;
- os incrementais ficaram entre `0,0749 s` e `0,0803 s`;
- não houve despacho separado de embedding, prelude PLE ou vocabulary head.

Uma prova contínua de 16 tokens gerou, nos três runtimes,
`" a city that is full of history and culture. It is a city that is"`, com os
16/16 IDs iguais. O compilado levou `1,6583 s` (`9,6486 token/s`); seus 15
forwards incrementais permaneceram entre `0,0755 s` e `0,0815 s`.

Evidências promovidas:

- `artifacts/gemma4-three-way-calibration-8x2-mlx-token-forward-terminal.json`;
- `artifacts/gemma4-three-way-france-16-token-forward.json`.

## Geração inteira em uma chamada com KV residente

O caminho textual compilado agora pode executar também o controle
autoregressivo dentro do worker MLX. A primeira requisição autentica e compila
os descritores do artefato; as requisições seguintes contêm somente os IDs do
prompt, `maxNewTokens`, `topK` e o EOS opcional. Embeddings, PLE, as 42 camadas,
norma/head/softcap, argmax com desempate pelo menor ID e append do próximo token
permanecem no mesmo processo. Os caches K/V nunca voltam ao Node durante a
geração.

O resultado compacto contém os IDs emitidos, duração de cada forward, top-K de
cada seleção, SHA-256 dos logits BF16 terminais e o tamanho do KV residente. A
interface expõe `externalForwardRequests`, `kvCacheTransportBytes`,
`residentKvBytes`, `residentGeneration` e
`fusedTokenGenerationDispatches`. Assim, uma geração textual inteira registra
uma chamada externa e zero bytes de transporte do KV.

O corpus real de oito prompts por dois tokens em
`gemma4-three-way-calibration-8x2-mlx-resident-generation.json` registrou:

- 8/8 prompts e 16/16 tokens diretos iguais ao Transformers eager BF16;
- `8,2855 token/s` no compilado contra `0,9255 token/s` no baseline, razão
  `8,9521x`;
- uma única chamada externa por prompt e zero bytes de KV transportado;
- somente um despacho de geração residente por prompt após o warm-up.

No ensaio contínuo aquecido de 16 tokens, a rota residente produziu os mesmos
16 IDs e o mesmo SHA-256 terminal
`5ed4dc16c6b7f7828844552fff90aaf5680bfbdff9e4be69ea730e5a6dba1f3e`
da rota não residente. O tempo caiu de `1,2479 s` (`12,8217 token/s`) para
`0,9083 s` (`17,6160 token/s`), redução de `27,22%` e ganho de `37,39%` em
throughput. Os 15 incrementais residentes ficaram entre `0,0521 s` e
`0,0531 s`.

A prova três-vias persistida em
`gemma4-three-way-france-16-resident-generation.json` confirmou os mesmos
16/16 IDs e o texto completo nos três runtimes. Incluindo o primeiro prefill
daquele tamanho após o warm-up, o compilado levou `0,9986 s`
(`16,0220 token/s`) contra `5,9252 s` (`2,7003 token/s`) do Transformers,
razão `5,9333x`.

Comando reprodutível:

```bash
npm run calibrate:gemma4-real -- \
  --source ./gemma-4-E4B-dense \
  --literal-artifact ./artifacts/gemma4-compiled-global-runtime-bundle/constants.literal.json \
  --binary-pool ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-three-way-calibration-8x2-mlx-resident-generation.json \
  --tokens 2 --request-threads 1 \
  --direct-linear-backend mlx --direct-threads 10 \
  --direct-fused-decoder-stack native-bf16 \
  --direct-fused-token-forward bf16 \
  --direct-resident-generation on \
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
divergência, ele permanece experimental. Naquela etapa, `bf16` foi promovido:
reduziu as fronteiras de processo sem degradar o corpus. Ele continua como
controle com pesos ampliados. MLX permanece em `off`. A interface exibe
separadamente o núcleo e o subgrafo, suas políticas, despachos e lineares
restantes.

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

## Projeções de attention sobre storage BF16 nativo

Depois da MLP nativa, a telemetria mostrou que o subgrafo de attention era a
maior classe aquecida restante, com aproximadamente `0,163 s` por pedido de
dois tokens. Embora Q/K/V/O já estivessem fundidos com normas, RoPE, máscara,
softmax e cache KV, suas matrizes BF16 ainda eram ampliadas integralmente para
F32 a cada camada.

O modo `native-bf16` mantém Q/K/V/O no `mmap` e executa somente esses quatro
GEMMs em BF16 nativo. Os resultados de cada projeção voltam a F32 nas mesmas
boundaries BF16 do modo anterior. RMSNorm, RoPE, scores, máscara, softmax,
contexto e cache KV permanecem no caminho já validado; portanto a otimização
não transforma softmax ou cache em uma aproximação nova.

No ensaio persistente `[2]` de dois tokens, a attention aquecida caiu de cerca
de `0,163 s` para `0,070 s`; o pedido completo típico passou de `0,51–0,53 s`
para `0,41–0,42 s`. Os tokens permaneceram `[184,3910]`. O SHA-256 terminal
passou para
`71eeb041bc97c6674686eb4dc99887cca8a50a7cb4947e42c3a63a0bb58e9e2d`,
identificando a ordem de acumulação do kernel nativo.

No corpus 8×2:

- controle com attention `bf16`: `1,5618 token/s`, 7/8 prompts e 15/16 tokens;
- attention `native-bf16`: `1,7380 token/s`, os mesmos 7/8 e 15/16 tokens;
- ganho direto: 11,28%;
- razão contra o baseline da própria execução: `3,1651x`;
- tempo acumulado de attention: `2,0630 s → 1,0379 s`, redução de 49,70%.

`native-bf16` passa a ser o padrão PyTorch. `bf16` conserva o controle anterior,
`real` conserva a composição sem boundaries internas e `off` conserva o caminho
separado. A flag é
`--fused-attention off|bf16|real|native-bf16` no worker ou
`--direct-fused-attention ...` no comparador. Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-native-bf16-mlp.json`;
- `artifacts/gemma4-three-way-calibration-8x2-native-bf16-attention.json`.

## Bloco FFN compilado até o residual

Depois das projeções BF16 nativas, a MLP isolada ainda devolvia o tensor
`down_proj` ao JavaScript. O executor paginado calculava separadamente
`pre_ffn_norm`, `post_ffn_norm` e o residual, materializando oito resultados
intermediários por camada mesmo quando somente `layer_n_after_mlp` era usado
pela continuação do grafo.

O subgrafo FFN fecha, em um único despacho persistente:

```text
pre_ffn_norm -> gate/up -> GELU -> multiply -> down
             -> post_ffn_norm -> residual
```

Ele recebe o vetor anterior e referências autenticadas para as duas normas e
as três matrizes da MLP. Gate, up e down continuam executando sobre storage
BF16 nativo; as duas RMSNorm e cada boundary BF16 permanecem explícitas. O
retorno é diretamente `layer_n_after_mlp`, portanto gate, up, ativação, produto,
down e as duas normas não atravessam mais o canal JavaScript.

Também foi medido um caminho que tratava gate/up contíguos como uma matriz
única. Embora o microbenchmark do GEMM isolado mostrasse 7–11% de ganho e saída
bit a bit igual, no corpus ele caiu para `1,6932 token/s` e aumentou o tempo da
MLP para `2,5379 s`. A variante foi descartada e não integra o runtime.

No ensaio persistente `[2]` de dois tokens, o FFN completo preservou os tokens
`[184,3910]` e o SHA-256 terminal
`71eeb041bc97c6674686eb4dc99887cca8a50a7cb4947e42c3a63a0bb58e9e2d`.
Pedidos aquecidos típicos passaram de aproximadamente `0,415–0,416 s` para
`0,401–0,406 s`.

No corpus 8×2:

- controle: `1,7380 token/s`, 7/8 prompts e 15/16 tokens;
- FFN completo: `1,8092 token/s`, os mesmos 7/8 e 15/16 tokens;
- ganho direto: 4,10%;
- razão contra o baseline da própria execução: `3,3341x`;
- tempo direto total: `9,2062 s -> 8,8439 s`.

`native-bf16` passa a ser o padrão PyTorch para `--fused-ffn`; `off` preserva
o caminho anterior para A/B e MLX permanece em `off`. A interface mostra
política, despachos e tempo do FFN separadamente. Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-native-bf16-attention.json`;
- `artifacts/gemma4-three-way-calibration-8x2-fused-ffn-native-bf16.json`.

## Camada decoder completa BF16 nativa

O compositor reverso agora reconhece o subgrafo integral de cada uma das 42
camadas e substitui suas 32 operações IR por um único despacho persistente:

```text
input norm -> Q/K/V -> RoPE -> attention -> O -> post-attention norm/residual
           -> pre-FFN norm -> gate/up/GELU/multiply/down
           -> post-FFN norm/residual -> PLE -> scalar
```

O despacho recebe o estado da camada, a entrada PLE, posições, máscara, cache
KV e referências autenticadas para todas as matrizes e normas. As fronteiras
BF16, RoPE, softmax, resíduos e atualização do cache continuam explícitas, mas
os tensores intermediários deixam de atravessar o canal JavaScript. A saída é
diretamente o escalar final da camada, acompanhada apenas do novo cache K/V.

No ensaio persistente `[2]` de dois tokens, a camada completa preservou os
tokens `[184,3910]` e o mesmo SHA-256 terminal
`71eeb041bc97c6674686eb4dc99887cca8a50a7cb4947e42c3a63a0bb58e9e2d`.
Em um prompt real, `The capital of France is`, original e direto produziram os
tokens `[496,3207]`, isto é, ` a city`.

No corpus 8x2:

- controle com FFN completo: `1,8092 token/s`, 7/8 prompts e 15/16 tokens;
- camada decoder completa: `1,9194 token/s`, os mesmos 7/8 e 15/16 tokens;
- ganho sobre o controle: 6,09%;
- razão contra o baseline da própria execução: `3,5741x`;
- tempo direto total: `8,8439 s -> 8,3361 s`;
- 672 despachos de camada completa e zero despachos separados de atenção,
  FFN ou PLE.

`native-bf16` passa a ser o padrão PyTorch para `--fused-decoder-layer`;
`off` preserva o caminho anterior para A/B e MLX permanece em `off`. A
interface mostra política, contagem e tempo da camada completa. Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-fused-ffn-native-bf16.json`;
- `artifacts/gemma4-three-way-calibration-8x2-fused-decoder-layer-native-bf16.json`.

## Pilha decoder integral em um despacho

As 42 camadas consecutivas agora também são reconhecidas como uma única pilha
compilada. O runtime envia uma vez o estado inicial, as posições e o tensor PLE
`[B,S,42,P]`; cada camada seleciona sua fatia PLE e alimenta diretamente a
seguinte dentro do worker nativo. Produtores de KV recebem o cache anterior e
publicam o cache atualizado; camadas consumidoras reutilizam esse resultado
internamente pelo índice explícito do produtor.

Assim, um forward completo passa de 42 despachos de camada para um despacho da
pilha, sem remover nenhuma RMSNorm, fronteira BF16, RoPE, softmax, residual,
PLE ou transição de cache. O matcher falha fechado se a sequência de camadas,
dependência residual, largura PLE, índice ou propriedade de cache divergir do
programa Gemma 4 declarado.

No ensaio `[2]` de dois tokens, a pilha preservou `[184,3910]` e o SHA-256
terminal
`71eeb041bc97c6674686eb4dc99887cca8a50a7cb4947e42c3a63a0bb58e9e2d`,
usando dois despachos de pilha e zero despachos individuais de camada.

No corpus 8x2:

- controle por camada: `1,9194 token/s`, 7/8 prompts e 15/16 tokens;
- pilha integral: `1,9945 token/s`, os mesmos 7/8 e 15/16 tokens;
- ganho sobre o controle: 3,91%;
- razão contra o baseline da própria execução: `3,8221x`;
- tempo direto total: `8,3361 s -> 8,0221 s`;
- despachos do corpo decoder: `672 -> 16`.

`native-bf16` passa a ser o padrão PyTorch para `--fused-decoder-stack`;
`off` mantém o caminho por camada para A/B e MLX permanece em `off`. A
interface expõe política, despachos e tempo da pilha. Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-fused-decoder-layer-native-bf16.json`;
- `artifacts/gemma4-three-way-calibration-8x2-fused-decoder-stack-native-bf16.json`.

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

## Head BF16 paginado em fluxo único

O modo `native-bf16-stream` mantém a paginação de 32 MiB que superou o GEMM
integral, mas move o loop de tiles para dentro do worker persistente. O
JavaScript envia uma única entrada e uma referência autenticada para a matriz;
o worker percorre o `mmap` em tiles BF16 e devolve cada bloco de logits pelo
mesmo fluxo binário assim que o GEMM termina. Não há matriz de 1,342 GB
materializada nem vetor terminal duplicado no worker.

Essa organização conserva exatamente os mesmos GEMMs e boundaries do modo
`native-bf16`. Para `[2]` com dois tokens, ambos produziram `[184,3910]` e o
SHA-256 terminal
`71eeb041bc97c6674686eb4dc99887cca8a50a7cb4947e42c3a63a0bb58e9e2d`.
O novo caminho realizou dois despachos de head e reduziu os demais despachos
referenciados de 90 para 8.

No controle térmico 8x2:

- head paginado pelo JavaScript: `2,0087 token/s`;
- head paginado no fluxo único: `2,0272 token/s`;
- ganho direto: 0,92%, com os mesmos 7/8 prompts e 15/16 tokens;
- tempo direto total: `7,9655 s -> 7,8926 s`;
- travessias referenciadas/stream do corpus: `720 -> 64 + 16`, redução de
  88,89% nas chamadas totais desse caminho;
- tempo atribuído ao head permaneceu praticamente igual, cerca de `0,60 s`;
- pico RSS permaneceu na mesma faixa: `2.583.456 -> 2.596.608 KiB`.

`native-bf16-stream` passa a ser o padrão PyTorch. `native-bf16` mantém o A/B
paginado pelo JavaScript, `native-bf16-whole` mantém o experimento integral e
MLX continua em `f32`. A interface mostra separadamente despachos e tempo do
head em fluxo único. Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-paged-native-bf16-head-current-control.json`;
- `artifacts/gemma4-three-way-calibration-8x2-streamed-native-bf16-head.json`.

## FFN unido e perfil interno da pilha decoder

O worker passou a devolver, no mesmo despacho binário, a decomposição do tempo
da pilha em attention, FFN e PLE. No corpus 8x2 com dez threads, os totais do
controle fiel foram `0,8717 s`, `2,1746 s` e `0,4225 s`, respectivamente. O
FFN representa aproximadamente 62% dos `3,5212 s` da pilha e é o alvo
dominante das próximas fusões.

As matrizes BF16 `gate_proj` e `up_proj` de cada uma das 42 camadas são
adjacentes no pool autenticado. Para forwards com mais de uma linha, o worker
as enxerga como uma única matriz `[20480,2560]` e executa uma GEMM, separando o
resultado em gate/up sem copiar pesos. Ensaios independentes com uma e seis
linhas confirmaram igualdade bit a bit com as duas GEMMs. Para a continuação
unitária, onde a GEMM unida foi ligeiramente pior, o runtime conserva as duas
operações. O corpus registrou 336 pares unidos: 42 por prompt inicial.

Um sweep quente no M3 Max preservou os tokens e o SHA terminal em 4, 8, 10 e
14 threads. Dez threads minimizaram a pilha (`~0,349 s`, contra `~0,376 s` em
oito e `~0,383 s` em quatorze), coerente com os dez núcleos de desempenho da
máquina. Por isso o padrão direto passou de oito para dez threads, mantendo a
flag explícita para outras máquinas.

Contra o relatório anterior de oito threads, o controle 8x2 atualizado passou
de `2,0272` para `2,0953 token/s` (+3,36%) e reduziu a pilha agregada de
`3,8130` para `3,5212 s` (-7,65%), mantendo 7/8 prompts e 15/16 tokens iguais
ao baseline. O pico RSS foi `2.609.664 KiB`.

O modo experimental `native-bf16-ple` também executa as duas projeções PLE em
BF16 nativo. Ele reduziu a fase PLE agregada de `0,4225` para `0,3155 s`
(-25,3%) e atingiu 8/8 prompts e 16/16 tokens no corpus, mas o ensaio pareado
não melhorou o tempo total (`2,0373 token/s`) por variação nas fases FFN/head.
Consequentemente ele permanece disponível para A/B, enquanto `native-bf16`
continua sendo o padrão fiel.

Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-stack-gate-up-10t-control.json`;
- `artifacts/gemma4-three-way-calibration-8x2-stack-native-ple-10t.json`.

## Cache das constantes compiladas da pilha

As constantes BF16 pequenas que a pilha fiel usa em aritmética F32 agora são
alargadas uma única vez e mantidas pelo worker persistente. A chave liga shard,
intervalo autenticado, shape e dtype; assim, dois tensores diferentes nunca
compartilham um valor apenas por terem o mesmo nome ou tamanho. Matrizes com
payload de origem acima de 2 MiB continuam diretamente no `mmap`, impedindo
que o cache replique os pesos grandes de attention, FFN e vocabulary head.

O perfil binário da pilha passou a publicar acertos, entradas residentes e
bytes F32. Esses dados atravessam o worker JSONL, o relatório de calibração e a
interface. No corpus 8x2, o estado estabilizou em 402 entradas / 222.430.368
bytes (212,13 MiB) e registrou 6.030 acertos. A validação de finitude continua
checando o resultado de cada camada, mas cada cache KV passa a ser verificado
somente quando sua camada proprietária o produz. Consumidores shared-KV não
repetem a varredura do mesmo cache já aprovado.

O caminho preservou os mesmos 7/8 prompts e 15/16 tokens do controle. Na prova
isolada repetida, `[2,818,5279,529,7001,563]` continuou gerando `[496,3207]` e
o SHA-256 terminal permaneceu
`4dd07a5fd3052fba67ea38f0359a85dbf7e97e80b3e4710bb13f615dcdf34aa8`.
No corpus final, a fase PLE caiu de `0,4225 s` para `0,3381 s` (-20,0%). A
variação simultânea de attention/FFN absorveu esse ganho: a pilha ficou em
`3,5269 s` contra `3,5212 s` e o throughput total em `2,0864` contra `2,0953
token/s`. Portanto esta etapa comprova a eliminação de alargamentos repetidos,
mas não é apresentada como aceleração global; os números absolutos ficaram
dentro do ruído do ensaio.

Uma alternativa híbrida que enviava cada FFN ao Metal/MLX foi rejeitada: o
microbenchmark de GEMM isolado parecia favorável, mas 42 sincronizações
CPU/GPU elevaram o FFN quente de cerca de `0,21 s` para `0,44 s`. Buffers BF16
reutilizáveis para os FFNs também foram removidos da versão final porque o
corpus não demonstrou ganho estável.

Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-stack-cached-constants-10t.json`;
- `artifacts/gemma4-three-way-calibration-8x2-stack-gate-up-10t-control.json`.

## Pilha decoder integral no Metal

O backend MLX agora recebe a pilha completa das 42 camadas em um único
despacho binário. O worker resolve as matrizes autenticadas diretamente do
bundle fechado, compõe norms, Q/K/V, RoPE, máscaras, attention, FFN/GELU, PLE,
resíduos e escalares em um grafo lazy e sincroniza o Metal somente uma vez no
fim da pilha. Os caches KV proprietários também são devolvidos no mesmo
despacho; consumidores shared-KV reutilizam o nó já produzido no grafo.

Essa é a forma executável da composição matemática global. Ela remove
fronteiras de orquestração e materializações intermediárias, mas não elimina
as matrizes densas arbitrárias: uma matriz aprendida full-rank ainda precisa
participar de sua multiplicação, salvo se aceitarmos aproximação, esparsidade
ou nova parametrização. Portanto o ganho demonstrado aqui vem da composição e
do agendamento integral, não de uma alegação incorreta de que todos os pesos
podem ser reduzidos a poucos coeficientes escalares.

As projeções BF16 permanecem nativas, enquanto constantes pequenas usadas em
F32 são alargadas uma vez e armazenadas no worker. A vocabulary head inteira
também passou a executar em uma única GEMM BF16 no Metal. O protocolo valida
topologia, shapes, dtype, limites do shard, máscaras, relações shared-KV e
finitude antes de publicar qualquer resultado. O backend PyTorch continua
disponível por flag como controle e fallback explícito.

Uma prova quente de duas gerações para
`[2,818,5279,529,7001,563]` produziu `[496,3207]` nas duas execuções. A segunda
atingiu `3,4740 token/s`, com `0,1705 s` nas duas pilhas, dois despachos de head
e somente dez despachos referenciados totais. O SHA-256 terminal MLX foi
`01bd2dc9a6806578b9d6a3e7cae67600d9bafd327841a64b6f3e2731a8f56498`.
Ele difere do controle PyTorch porque a ordem de acumulação dos kernels Metal
não é bit a bit idêntica; a validade de qualidade é medida separadamente pelos
tokens e pelo corpus diferencial.

No corpus final 8x2, executado com os novos padrões da interface:

- o direto MLX produziu os mesmos 8/8 prompts e 16/16 tokens do original;
- o throughput direto foi `2,2250 token/s`, contra `2,0864 token/s` do controle
  PyTorch com cache, ganho de 6,64%;
- a pilha agregada caiu de `3,5269 s` para `2,3563 s`, redução de 33,19%;
- o direto ficou 4,3064 vezes acima do baseline medido na mesma execução;
- foram 16 pilhas integrais, 16 heads integrais e 80 despachos referenciados em
  todo o corpus;
- o cache estabilizou em 402 entradas / 222.430.368 bytes e registrou 6.030
  acertos;
- o pico RSS subiu de `2.624.240` para `2.835.168 KiB`, custo de `210.928 KiB`
  (aproximadamente 206 MiB) para manter constantes e o grafo no backend Metal.

`mlx + native-bf16 + native-bf16-whole` passa a ser o caminho direto padrão
do CLI, servidor, calibração e interface. A interface identifica a pilha
integral, mostra o backend real, despachos, cache, RSS, throughput e igualdade
de tokens, e não apresenta os tempos internos de attention/FFN/PLE como zeros:
no MLX essas fases estão fundidas no mesmo grafo e não são sincronizadas
separadamente.

A superfície HTTP foi exercitada de ponta a ponta com os defaults promovidos.
Para o prompt `Hello`, original, compatibilidade e direto geraram
`[236764,108]`. O direto identificou
`persistent-mlx-metal-full-stack`, executou duas pilhas integrais e duas heads
integrais e mediu `1,9988 token/s` nessa primeira requisição fria. Assim, a
prova cobre a mesma rota usada pelo botão da interface, incluindo tokenização,
geração greedy, decodificação de tokens e publicação das métricas.

Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-mlx-full-stack.json`;
- `artifacts/gemma4-three-way-calibration-8x2-stack-cached-constants-10t.json`.

## Forward compilado até os logits

A fronteira compilada passou a reconhecer também o epílogo textual exato que
sucede a 42ª camada: `final_norm -> lm_head -> tanh_softcap`. A fusão só é
ativada quando o programa literal prova todas as dependências, shapes, pesos
BF16, ausência de bias, norma direta, escala positiva e os três casts BF16 do
softcap. O worker recebe os dois novos tensores pelo mesmo protocolo autenticado
e devolve diretamente `softcapped_logits`; o JavaScript não executa mais a
norma final nem abre um segundo despacho para a matriz `[262144,2560]`.

Decoder e epílogo formam um único despacho de forward. Internamente, o worker
sincroniza o decoder antes de materializar a head, e então sincroniza os logits.
Essa divisão preserva a composição matemática, mas impede que o grafo lazy
retenha simultaneamente todas as matrizes das 42 camadas e os 1,25 GiB da head.
Uma versão experimental com uma única sincronização Metal preservou os mesmos
tokens, porém elevou o pico transitório quando o Transformers original estava
residente; ela foi substituída pela execução em duas fases dentro do mesmo
despacho para a interface permanecer utilizável na máquina de validação.

O carregamento do servidor agora segue a mesma restrição de memória de forma
determinística. Primeiro o backend MLX executa um warm-up real com `[2]`,
preenche o cache das 403 constantes e compila os kernels; somente depois o
worker Transformers é iniciado. `/api/status` publica `warmupComplete` e
`warmupSeconds`. Carregar o original antes da primeira compilação Metal levou o
processo comparativo a ultrapassar a pressão de memória do host; a ordem
promovida foi validada com os dois workers residentes e não altera as métricas
por requisição, que continuam calculadas por deltas.

No corpus final 8x2:

- o direto preservou 8/8 prompts e 16/16 tokens do original;
- o throughput passou de `2,2250` para `4,6476 token/s`, ganho de 108,88%
  contra a pilha Metal anterior;
- o tempo direto agregado caiu de `7,1911 s` para `3,4427 s`, redução de
  52,13%;
- a razão contra o baseline medido na mesma execução foi `5,0308x`;
- os 16 despachos separados de head foram eliminados: `whole head 16 -> 0`;
- os despachos referenciados caíram de 80 para 64, enquanto 16/16 forwards
  publicaram logits pelo despacho integral;
- o pico RSS direto foi `3.148.112 KiB`, aproximadamente 306 MiB acima do
  controle Metal anterior, em troca da head e dos kernels residentes;
- o warm-up isolado levou `14,0874 s`, fora das métricas de geração.

A rota HTTP real também foi executada por oito tokens com crescimento do KV
cache. Para `The capital of France is`, original, compatibilidade e compilado
geraram exatamente `[496,3207,600,563,2587,529,4083,532]`. O direto completou
em `1,0406 s` (`7,6878 token/s`); depois do prefill de `0,4622 s`, os sete
forwards incrementais permaneceram entre `0,0761` e `0,0919 s`. Foram oito
forwards integrais, nenhuma head separada e nenhuma divergência de token.

Evidência:

- `artifacts/gemma4-three-way-calibration-8x2-mlx-full-forward.json`;
- `artifacts/gemma4-three-way-calibration-8x2-mlx-full-stack.json`;
- prova HTTP de oito tokens preservada como artefato de entrega.

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

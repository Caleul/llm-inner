# Model Decompiler v2

Reescrita do conversor de pesos para um IR matemático estruturado e verificável.

## Mudança principal

`MAX_FEATURES` não altera a rede. Ele limita apenas o preview embutido no JSON. Uma projeção continua sendo representada como:

```text
y[o] = Σ(i=0..in_features-1) x[i] * W[o,i] + b[o]
```

O IR guarda `inFeatures`, `outFeatures`, referência ao tensor, dtype e quantização. Os primeiros termos podem ser anexados para inspeção, mas nunca substituem a soma completa.

## Fontes

- Diretórios Hugging Face/Safetensors densos.
- Diretórios MLX/Safetensors quantizados, lendo `config.json`, incluindo overrides por módulo.
- Arquivos GGUF v2/v3 por um leitor nativo estrito de header, metadata e diretório; o subconjunto denso Llama também pode chegar ao IR e à materialização F32 sem bridge Python.

A leitura numérica é delegada ao runtime de referência:

- `mlx.core.dequantize` para MLX (`affine`, `mxfp4`, `mxfp8`, `nvfp4` e modos suportados pela versão instalada).
- `GGML_TYPE_Q4_0` tem um decodificador local estrito: cada bloco contém um
  `ggml_half d` little-endian e 16 bytes com os nibbles baixos para `q[0..15]`
  e altos para `q[16..31]`, materializando `F32[i] = d * (q[i] - 8)`.
- `GGML_TYPE_Q4_1` tem um decodificador local estrito e distinto: cada bloco
  contém `ggml_half d`, `ggml_half m` e 16 bytes com os nibbles baixos para
  `q[0..15]` e altos para `q[16..31]`, materializando `F32[i] = d * q[i] + m`
  e preservando a proveniência `gguf/q4_1`.
- `GGML_TYPE_Q4_K` tem um contrato local estrito de 144 bytes por bloco de 256
  valores: `ggml_half d`, `ggml_half dmin`, 12 bytes que codificam oito campos
  de escala e oito campos de mínimo de seis bits, e 128 bytes de códigos de
  quatro bits. Cada grupo de 32 valores é `F32[i] = d * scale[group] * q[i] -
  dmin * minimum[group]`; os grupos 4..7 unem campos altos e baixos em bytes
  diferentes. A proveniência é `gguf/q4_k`.
- `GGML_TYPE_Q5_0` tem um decodificador local estrito: cada bloco contém
  `ggml_half d`, um plano de quatro bytes com o quinto bit de cada valor e 16
  bytes de nibbles baixos para `q[0..15]` e altos para `q[16..31]`,
  materializando `F32[i] = d * ((low4[i] | high1[i] << 4) - 16)` e preservando
  a proveniência `gguf/q5_0`.
- `GGML_TYPE_Q5_1` tem um decodificador local estrito e distinto: cada bloco
  contém `ggml_half d`, `ggml_half m`, um plano de quatro bytes com o quinto
  bit de cada código e 16 bytes de nibbles baixos para `q[0..15]` e altos para
  `q[16..31]`, materializando `F32[i] = d * (low4[i] | high1[i] << 4) + m` e
  preservando a proveniência `gguf/q5_1`.
- `GGML_TYPE_Q6_K` tem um contrato local estrito de 210 bytes por bloco de 256
  valores: `ql[128]` fornece quatro bits baixos, `qh[64]` fornece dois bits
  altos em oito planos de 32 valores, `scales[16]` contém escalas `int8` por
  grupo de 16 valores e o `ggml_half d` final é a escala base. Cada valor é
  `F32[i] = d * scales[floor(i/16)] * ((ql4[i] | qh2[i] << 4) - 32)`, com
  proveniência `gguf/q6_k`; ordem, sinal e hierarquia das escalas são
  verificadas pelo layout, não inferidas de um nome de seis bits.
- `GGML_TYPE_Q8_0` tem um decodificador local estrito: cada bloco contém um
  `ggml_half d` little-endian seguido de 32 `int8` assinados e materializa
  `F32[i] = d * qs[i]`, preservando a proveniência `gguf/q8_0` no tensor.
- `GGML_TYPE_Q8_1` tem um contrato local estrito e distinto: cada bloco contém
  `float d`, o campo auxiliar `float s = d*sum(qs)` e 32 `int8` assinados.
  A reconstrução de cada elemento é `F32[i] = d * qs[i]`; `s` faz parte do
  layout de bloco para kernels de produto interno, mas não é escala ou offset
  por elemento. A proveniência preservada é `gguf/q8_1`.
- `GGML_TYPE_BF16` é armazenamento denso escalar, não uma quantização: cada
  elemento é o `bfloat16` IEEE-754 little-endian formado pelos 16 bits mais
  significativos de um `float32`; o leitor os amplia para F32 sem alterar
  subnormais, infinitos ou NaNs.
  Os demais tipos GGML quantizados continuam rejeitados até cada layout ter um
  decodificador local verificado; o catálogo nunca trata um `GGML_TYPE_Q*` como
  um F32 genérico.

Isso evita implementar uma falsa “dequantização genérica por número de bits”. Q4_K, IQ2, MXFP4 e affine-4bit têm layouts e fórmulas diferentes.

## Instalação

```bash
npm install
npm run build
python3 -m pip install safetensors
# Em Apple Silicon, para modelos MLX:
python3 -m pip install mlx
```

## Uso

```bash
node dist/src/cli.js \
  --source ./gemma-4-E4B-it-MLX-4bit \
  --output ./model.ir.json \
  --equations ./model.equations.txt \
  --max-features 10 \
  --max-terms 10 \
  --include-weights
```

Sem `--include-weights`, a compilação não dequantiza previews; ela apenas cataloga os tensores e gera o grafo.

## Loop autônomo sequencial

O loop usa **um Codex por vez**. Ao fim de um ciclo, o agente cria um handoff
atômico em `.agent-loop/handoffs/completed`; somente o runner externo o valida
e inicia o próximo. O agente nunca aciona seu sucessor diretamente.

Antes do primeiro uso, configure uma identidade Git local para os commits de
cada ciclo:

```bash
git init
git config user.name "Seu nome"
git config user.email "seu-email@exemplo.com"
git add .
git commit -m "chore: bootstrap autonomous agent loop"
```

Então execute, a partir da raiz do projeto:

```bash
npm run loop:start
```

O limite é de 150 handoffs aceitos, configurado em `agent-loop.config.json`.
O runner exige árvore Git limpa, um novo commit por ciclo, testes configurados,
um handoff válido e nenhuma flag `.agent-loop/STOP` antes de iniciar o próximo.

```bash
npm run loop:status
npm run loop:stop
```

`loop:stop` não mata o Codex ativo; ele evita que o sucessor seja iniciado e
permite que o ciclo atual termine de forma coerente. Os logs e mensagens finais
de cada ciclo ficam em `.agent-loop/runs/`; estado, handoffs e logs são
ignorados pelo Git para não violar a exigência de árvore limpa.

Cada instância recebe contexto novo a partir do repositório e do último
handoff. O protocolo exige que ela complete um milestone substancial e
validado — não uma sequência de microalterações — e que o handoff final retenha
somente resultado, evidência, gargalos e a próxima frente de alto impacto.
Ao terminar, cada ciclo também registra uma lista ordenada de até três próximos
passos e seus critérios de aceite; o sucessor os reavalia contra o repositório
atual, em vez de tratá-los como uma fila cega.

O runner inicia os ciclos com acesso completo e sem confirmações interativas,
por autorização explícita do operador. Cada ciclo deve atuar como responsável
pelo objetivo final: atacar uma fronteira estratégica de fidelidade ou
validação, conectar as camadas necessárias e evitar encerrar apenas por uma
microalteração isolada.

## Política de fidelidade

O compilador falha quando:

- não existe `config.json` para Safetensors;
- a arquitetura não tem adaptador registrado;
- um tensor crítico está ausente ou ambíguo;
- um tensor de camada só coincide por substring/sufixo com um papel conhecido:
  cada adaptador aceita apenas convenções completas de nome registradas, e uma
  nova convenção exige mapeamento explícito e revisado;
- há QKV fundido sem layout conhecido;
- `hidden_act` não está entre as fórmulas de ativação explicitamente registradas;
- `attention_bias` ou `mlp_bias` declarado não coincide com a presença dos
  tensores `.bias` das projeções correspondentes (ou a flag não é booleana);
- `tie_word_embeddings` não é booleano, ou um `lm_head` independente está
  ausente quando o config não declara o peso de saída amarrado ao embedding;
- campos declarados de topologia de atenção (`layer_types`, janela deslizante,
  compartilhamento de KV) ou escala/softcap numéricos são inválidos, em vez de
  cair silenciosamente na atenção global padrão;
- RoPE declara uma variante, dimensão ou parâmetros fora do contrato
  `default`/`rotate_half` atualmente modelado; variantes como `linear`,
  `dynamic`, YaRN ou proporcional exigem fórmula e validação próprias;
- shapes de embedding, normas, Q/K/V/O, MLP ou biases não coincidem com a topologia declarada;
- há MoE/AltUp/LAuReL ou outra semântica ainda não implementada.

Ele não cria bypasses nem zeros para componentes desconhecidos.

Os adaptadores atuais registram `silu` (e o alias matematicamente idêntico
`swish`), `gelu` por `erf`, e `gelu_pytorch_tanh`/`gelu_new`/`gelu_fast` por
GELU-tanh. Qualquer outro rótulo falha na construção do IR até que a fórmula,
os limites numéricos e a implementação do executor sejam adicionados e
validados.

## Limites atuais

O adaptador atual cobre blocos decoder-only auditáveis de Llama, Mistral, Qwen 2/3 e Gemma 1/2/3-text. Modelos com código remoto, state-space layers, linear attention, MoE, multimodal completo, Gemma 3n/4 ou layouts QKV especiais precisam de adaptadores próprios ou extração do grafo do runtime oficial. Gemma 4 é rejeitado de propósito porque sua topologia inclui PLE, heads por tipo de camada, KV sharing e outras semânticas que o bloco genérico não representa.

Para Qwen 3 e outros adaptadores que declarem `q_norm`, `k_norm` ou `v_norm`,
o IR faz `reshape_heads` antes da RMSNorm. Esses pesos precisam ter exatamente
`head_dim` elementos: normalizar a projeção achatada misturaria cabeças e é
rejeitado como semântica incompatível.

Quando `tie_word_embeddings=true`, o IR usa explicitamente o tensor de
embedding como peso do `lm_head`, mesmo que o contêiner retenha uma cópia de
`lm_head.weight`. Um eventual `lm_head.bias` continua pertencendo ao módulo de
saída e é preservado; ele não é procurado como se fosse `embed_tokens.bias`.

A equivalência real deve ser confirmada por um validador diferencial: mesma entrada, mesmo dtype, comparação de embeddings, saída por camada, KV cache e logits contra Transformers/MLX/llama.cpp.

`compareExecutionTrace` em `src/differential.ts` define o contrato serializável
para essa comparação. Um hook do runtime autoritativo precisa fornecer a saída
de **cada** operação pelo `operationId` estável do IR, mais o KV pós-RoPE por
camada em BHSD, e identificar runtime, modelo, revisão/checksum, formato,
quantização, tokens e política de dtype. O relatório mede erro absoluto e
relativo, cosseno, sobreposição top-k, argmax e a primeira divergência. Captura
ausente, operação extra, shape incompatível ou cache faltante é `incomplete`,
nunca uma aprovação numérica. O contrato em si não é uma integração com
Transformers/MLX/llama.cpp nem constitui comparação de checkpoint real.
Quando o IR declara `final_logit_softcap`, as métricas de logits comparam a
saída terminal `softcapped_logits` com a captura `final_logit_softcap` (e não
o `lm_head` pré-softcap). A ausência dessa saída terminal também torna o
relatório `incomplete`.

`materializeReferenceF32Constants` em `src/materialize.ts` fecha a fronteira
entre o catálogo e o executor F32: reúne todos os `TensorRef` do IR, confirma
nome, shape, dtype e contrato de quantização contra o catálogo de origem e
materializa cada constante uma única vez. Safetensors densos usam o leitor de
intervalos F32/F16/BF16; MLX quantizado exige explicitamente o bridge com
`mlx.core.dequantize` e preserva a proveniência. GGUF denso F32/F16/BF16 usa o
seu próprio leitor de intervalos. Os únicos tipos GGML empacotados materializáveis
hoje são `Q4_0`, sob o contrato exato de bloco de 18 bytes (`F16` scale + 16
nibbles, baixo `q[0..15]`, alto `q[16..31]`, código centrado por `-8`), `Q4_1`,
sob o contrato distinto de 20 bytes (`F16` scale, `F16` minimum e 16 nibbles,
com `d*q+m`), `Q4_K`, sob o contrato de 144 bytes por grupo de 256 (`F16 d`,
`F16 dmin`, 12 bytes com oito escalas/mínimos de seis bits e 128 bytes de
códigos; `d*scale*q - dmin*minimum`), `Q5_0`, sob o contrato de 22 bytes (`F16` scale + plano de 32
high bits + 16 nibbles, código centrado por `-16`), `Q5_1`, sob o contrato
distinto de 24 bytes (`F16` scale, `F16` minimum, plano de 32 high bits e 16
nibbles, com código unsigned de cinco bits e `d*q+m`), `Q6_K`, sob o contrato
de 210 bytes por grupo de 256 (`ql[128]`, `qh[64]`, 16 escalas `int8` por 16
valores e `F16 d` final, com código de seis bits centrado por `-32`), `Q8_0`, sob o contrato
de 34 bytes (`F16` scale + 32 `int8`), e `Q8_1`, sob o contrato distinto de
40 bytes (`F32 d`, `F32 s=d*sum(qs)` e 32 `int8`, reconstruídos por `d*q`).
Q4_K e Q6_K exigem que a primeira dimensão GGML seja múltipla de 256; os demais acima
exigem múltiplos de 32. Os demais não são
reinterpretados como F32 até existir um decodificador por tipo verificado.

`GgufCatalogReader` em `src/gguf.ts` não depende de `gguf-py`: valida magic,
versão v2/v3, contagens seguras, metadata tipada (incluindo arrays), diretório,
alinhamento e intervalos de payload. Hoje expõe armazenamento GGML F32/F16/BF16 e os
contratos quantizados explícitos Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q6_K, Q8_0 e Q8_1 de tamanho verificável, mantendo as dimensões
na ordem declarada pelo GGML; qualquer outro encoding empacotado é rejeitado com
seu tipo GGML até haver contrato de layout e dequantização específico. A leitura do catálogo não inventa papéis de
tensor nem uma convenção de layout de arquitetura. A exceção executável atual
é o adaptador separado `adaptGgufLlamaCatalog`: ele exige
`general.architecture=llama`, reconhece apenas os nomes GGUF Llama registrados
(`token_embd`, `output` e projeções `blk.N`) e converte somente essas matrizes
da ordem GGML declarada `[in,out]` para a forma IR `[out,in]`.

`compareGenerationTrace` cobre a evidência que não cabe em um forward isolado:
ele exige uma captura autoritativa do prompt e suas posições absolutas, cada
token greedy emitido, a posição em que cada token foi avaliado, logits terminais
e o cache KV pós-RoPE por camada. Tokens iguais sem posições, logits ou cache
equivalentes não recebem aprovação. Captura ausente ou com shape incompatível é
`incomplete`; divergência medida é `approximate`; somente a captura completa
dentro da tolerância declarada pode ser `numerically-equivalent` (ou
`lossless-within-dtype` quando todos os valores comparados são exatos).

Além da máscara causal/janela declarada pelo IR, o executor de referência pode
receber uma máscara aditiva canônica `[batch, 1|heads, query, key]`. Zero
preserva o score e `-Infinity` exclui a chave antes do softmax; valores finitos
negativos representam viés aditivo. Sem máscara, a entrada é explicitamente
sem padding. Shapes inválidos, `NaN` e `+Infinity` falham fechado.

## Executor de referência F64 (slice atual)

`src/executor.ts` interpreta o subconjunto denso do IR de decoder (embedding,
RMSNorm, linear, RoPE `rotate_half` padrão, atenção causal/GQA, SiLU/GELU,
MLP gated e residuais) com tensores F64 explícitos. É um executor de referência
determinístico para fixtures e validação por operação. `SafetensorsCatalogReader`
mantém handles e agora carrega F64 denso diretamente do intervalo declarado pelo
header para esse executor; F32/BF16/F16 não são silenciosamente promovidos a F64,
pois isso perderia a política real de arredondamento/acumulação. Políticas de dtype
implícitas (`model-configured`), quantização e variantes de RoPE ainda
falham fechadas. Em particular, os executores aceitam somente `ropeType=default`,
layout `rotate_half` e nenhuma `rope_scaling` explícita; eles nunca aplicam
silenciosamente a fórmula padrão a uma variante declarada. Isso torna a fronteira de fidelidade observável antes de conectar
runtimes autoritativos.

O adaptador aplica a mesma fronteira já no lowering: uma configuração RoPE
declarada precisa ter `rope_type=default`, dimensão par dentro de `head_dim` e
parâmetros finitos positivos. `layer_types`, quando presente, precisa cobrir
todas as camadas com `full_attention` ou `sliding_attention`; valores
malformados não são reinterpretados como atenção global. Assim, o IR não
transporta rótulos de RoPE ou atenção cujo significado o executor/adaptador não
conhece.

O executor F64 exige que cada operação declare `computeDtype`,
`accumulationDtype` e `outputDtype` como `F64`; campos ausentes não são tratados
como defaults. `inputDtype`, quando declarado, também deve ser `F64`. Isso evita
que um IR incompleto seja executado com fronteiras de cast inventadas.

O executor também aceita e devolve cache KV por camada para decoder incremental.
O contrato canônico é um `Map` indexado pela camada, com `key` e `value`
post-RoPE no layout `[batch, kv_heads, cached_sequence, head_dim]`. Ao receber
cache, ele precisa conter todas as camadas não compartilhadas e é concatenado às
projeções atuais antes da atenção; `position_ids` continua explícito e deve usar
as posições absolutas do token novo. A máscara aditiva, se presente no decode,
tem chave com o comprimento completo (cache + tokens atuais). Shapes incompletos
ou incompatíveis falham fechados. Cache com `kvSharing` entre camadas continua
rejeitado até existir uma semântica de ownership validada.

Há também um caminho separado para F32 denso: `readDenseF32` preserva os bytes
do Safetensors em `Float32Array`, e `executeReferenceF32` requer que cada operação
declare explicitamente `computeDtype`/`accumulationDtype`/`outputDtype` como `F32`
(e softmax `F32`; `inputDtype`, se declarado, também precisa ser `F32`). O interpretador aplica `Math.fround` nas fronteiras
escalares de armazenamento e aritmética, sem converter os pesos para arrays de
`number` ou para o executor F64. Funções transcendentais usam a `libm` do host e
são arredondadas de volta para F32; portanto isto é uma política escalar declarada,
não uma alegação de equivalência bitwise com kernels de PyTorch, MLX, CUDA ou BLAS.
Enquanto o IR produzido pelo adaptador ainda declarar `model-configured`, ambos os
executores rejeitam a execução: a política deve vir de metadados/runtime
verificáveis, não ser inferida do dtype de armazenamento.

Checkpoints Safetensors densos em `F16` e `BF16` também podem ser lidos por
intervalo como valores `Float32Array` (`readDenseF16AsF32` e
`readDenseBF16AsF32`). A conversão preserva exatamente cada valor já
arredondado finito (e infinito) do armazenamento; ela não permite que o
executor escolha F32 por conta própria, nem afirma que o runtime original
acumulava em F32.

Os geradores `generateReferenceF64` e `generateReferenceF32` exercitam o
mesmo contrato de cache em geração greedy. Eles aceitam um único prompt sem
padding, escolhem o argmax dos logits terminais, alimentam cada token escolhido
de volta com a próxima posição absoluta e devolvem um cache que inclui prompt e
todos os tokens gerados. O token EOS também é avaliado antes de interromper, de
forma que esse cache continua utilizável. Batch, padding, estratégias de
stopping e sampling não são aproximados: exigem uma política de máscara/cache
declarada e continuam fora deste slice de referência.

Para o caminho F32 explícito, `readDenseAsF32` faz somente o despacho estrito
entre storage `F32`, `F16` e `BF16`; inteiros e tensores quantizados são
rejeitados até terem um decodificador de formato específico. Um fixture de
decoder completo executa pesos armazenados tanto em F16 quanto em BF16, mas
isso continua sendo validação de valores de storage em um interpretador escalar
F32 — não uma comparação diferencial contra MLX, PyTorch ou outro runtime.

Para um peso MLX quantizado, `TensorBridge.readMlxDequantizedF32` é a fronteira
explícita de materialização: exige catálogo `mlx-safetensors`, storage `U32`,
modo/bits/group size/scales declarados, e shapes 2D que coincidam com o
catálogo. O bridge repassa esse contrato a `mlx.core.dequantize`, verifica o
shape de saída e devolve bytes F32 (não uma lista JSON de números), marcados
com a proveniência completa da quantização. `executeReferenceF32` só aceita um
tensor quantizado se essa proveniência for idêntica ao `TensorRef` do IR;
buffer sem proveniência ou de outro esquema falha fechado. Isso habilita uma
execução F32 de fixtures/materializações declaradas, mas não estabelece a
política de dtype do modelo nem equivalência com um runtime MLX real. Um modo
MLX omitido também é rejeitado: `U32` + bit width não define um algoritmo.

A inspeção Safetensors também valida a fronteira do contêiner antes de expor
qualquer tensor: nomes de shards do índice não podem sair do diretório de
origem, o header tem o limite de 100 MiB da especificação, e cada intervalo
`data_offsets` deve ser ordenado, caber no payload e não sobrepor outro tensor.
Isso evita que um checkpoint malformado seja interpretado como pesos válidos ou
provoque uma alocação de header descontrolada.

Para MLX, `quantization` no `config.json` também não basta para reinterpretar
um tensor: apenas um `.weight` em `U32` com `bits`, `group_size`, `mode` e uma
matriz `.scales` 2D de mesmo número de linhas recebe o rótulo MLX. A dimensão
lógica é derivada de `scales.columns * group_size` e conferida contra a
capacidade do packing. Configuração quantizada aplicada a peso denso permanece
densa; U32 declarado como MLX sem scales compatíveis falha fechado.


## O que “agnóstico” significa

O núcleo é agnóstico ao contêiner e à quantização por usar backends separados, mas a semântica da arquitetura não pode ser deduzida apenas pelos nomes dos tensores. Cada família precisa de um adaptador validado que leia `config.json`/metadados GGUF e construa o grafo exato. Arquiteturas não reconhecidas falham de forma explícita.

## Regra de igualdade

O IR não declara igualdade bitwise apenas por ter os mesmos pesos. Para reproduzir o runtime original também são necessários dtype de cálculo, dtype de acumulação, ordem das reduções, máscara, RoPE, cache KV e arredondamentos. O campo `fidelity` mantém a validação diferencial como requisito obrigatório.

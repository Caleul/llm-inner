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

A leitura numérica usa contratos explícitos por formato:

- materialização nativa por ranges para MLX `affine` U32 (códigos 2/3/4/5/6/8-bit, `scale * code + bias` por grupo); `mxfp4`, `mxfp8`, `nvfp4` e outros modos continuam na fronteira explícita de `mlx.core.dequantize`.
- `GGML_TYPE_Q4_0` tem um decodificador local estrito: cada bloco contém um
  `ggml_half d` little-endian e 16 bytes com os nibbles baixos para `q[0..15]`
  e altos para `q[16..31]`, materializando `F32[i] = d * (q[i] - 8)`.
- `GGML_TYPE_Q4_1` tem um decodificador local estrito e distinto: cada bloco
  contém `ggml_half d`, `ggml_half m` e 16 bytes com os nibbles baixos para
  `q[0..15]` e altos para `q[16..31]`, materializando `F32[i] = d * q[i] + m`
  e preservando a proveniência `gguf/q4_1`.
- `GGML_TYPE_Q2_K` tem um contrato local estrito de 84 bytes por bloco de 256
  valores: 16 bytes em que cada nibble baixo é a escala e cada nibble alto é o
  mínimo de um grupo contíguo de 16 valores, 64 bytes com quatro planos de
  códigos unsigned de dois bits e `F16 d`/`F16 dmin` finais. Cada valor é
  `F32[i] = d * scale[floor(i/16)] * q[i] - dmin * minimum[floor(i/16)]`;
  a proveniência é `gguf/q2_k`. Os planos, a ordem de grupos e os mínimos não
  são inferidos pela largura de dois bits.
- `GGML_TYPE_Q3_K` tem um contrato local estrito de 110 bytes por bloco de 256
  valores: `ggml_half d`, `hmask[32]`, `qs[64]` com quatro planos de dois
  bits e 12 bytes que empacotam 16 escalas assinadas de seis bits por grupo de
  16 valores. Cada valor é `F32[i] = d * (scale[group] - 32) * (low2[i] -
  (highMask[i] ? 0 : 4))`; a posição do bit alto e os campos de escala são
  derivados do layout GGML, não do nome ou da largura de três bits. A
  proveniência é `gguf/q3_k`.
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
- `GGML_TYPE_Q5_K` tem um contrato local estrito de 176 bytes por bloco de 256
  valores: `ggml_half d`, `ggml_half dmin`, os mesmos 12 bytes de oito campos
  de escala e mínimo de seis bits do `Q4_K`, `qh[32]` com o quinto bit de cada
  valor lógico e 128 bytes de códigos baixos em planos de nibbles. Cada grupo
  de 32 valores é `F32[i] = d * scale[group] * (low4[i] | high1[i] << 4) -
  dmin * minimum[group]`; a proveniência é `gguf/q5_k`.
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
- `GGML_TYPE_Q8_K` tem um contrato local estrito de 260 bytes por bloco de 256
  valores: uma escala `float32 d` little-endian seguida por 256 códigos `int8`
  assinados, materializando `F32[i] = d * qs[i]`. Ele não possui o `F16 d` de
  `Q8_0` nem o campo auxiliar `s` de `Q8_1`; a proveniência é `gguf/q8_k`.
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

### Comparação com captura autoritativa

`npm run compare:trace -- --source <checkpoint> --trace <captura.json> --report <relatorio.json>` executa a fronteira completa de validação declarada pela captura (`F32` ou `F64`): reabre o contêiner, confere SHA-256 de `config.json` e de cada shard/arquivo que participa do checkpoint, reconstrói o IR, materializa os pesos por range, executa o interpretador correspondente e compara cada operação e cache KV com a captura. O relatório só é escrito depois de todas essas verificações.

A captura é JSON `schemaVersion: 1`, `kind: "execution"`, e exige:

- `source.files`: lista exata de caminhos relativos seguros e checksums SHA-256 do checkpoint;
- `irFingerprint`: SHA-256 do IR serializado antes da política declarada;
- `candidatePolicy`: `dtype: "F32"` ou `"F64"` e o identificador do executor candidato;
- `reference`: identidade imutável do runtime/modelo/revisão, tokens de entrada e cada operação por `operationId`, além do cache KV pós-RoPE BHSD;
- cada tensor com o mesmo `dtype` da política (`"F32"` ou `"F64"`), `shape` e `valuesBase64` com bytes IEEE-754 little-endian — não arrays decimais sujeitos a arredondamento JSON nem mistura silenciosa de precisão.

O comando recusa arquivo ausente/extra, checksum divergente, fingerprint de IR diferente, dtype implícito, operação duplicada, shape/payload inválido e evidência incompleta. A captura ainda precisa ser produzida por hooks verificados no runtime autoritativo; esse mecanismo não transforma o executor escalar em uma referência de Transformers, MLX ou llama.cpp.

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

Cada agente também atua como responsável técnico temporário do repositório:
deve usar princípios de Clean Code e SOLID pragmaticamente, preservar fronteiras
claras entre contêiner, quantização, arquitetura, execução e validação, reduzir
duplicação relevante e deixar contratos, erros e testes mais fáceis de evoluir
nas sessões seguintes. O handoff é uma proposta técnica curta para o próximo
ciclo, não apenas a confirmação de uma tarefa concluída.

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

O adaptador atual cobre blocos decoder-only auditáveis de Llama, Mistral, Qwen 2/3 e Gemma 1/2/3-text. Modelos com código remoto, state-space layers, linear attention, MoE, multimodal completo, Gemma 3n/4 ou layouts QKV especiais precisam de adaptadores próprios ou extração do grafo do runtime oficial. Gemma 4 é rejeitado de propósito porque sua topologia inclui PLE, heads por tipo de camada, KV sharing e outras semânticas que o bloco genérico não representa. A recusa usa os metadados/tensores declarados: `hidden_size_per_layer_input`, `global_head_dim`, RoPE proporcional para `full_attention`, `num_kv_shared_layers` e pesos PLE (`embed_tokens_per_layer`, `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`, `layer_scalar`) aparecem no diagnóstico quando presentes.

Um pacote composto não é convertido implicitamente em um checkpoint textual. Quando o `config.json` externo declara `audio_config` ou `vision_config` e tokens de modalidade, enquanto `text_config` descreve outro `model_type`, o compilador falha antes da seleção do adaptador: a injeção desses tokens, os towers e sua ordem no forward pass fazem parte da função. Um futuro adaptador de texto só pode reutilizar esse submodelo depois que um adaptador do pacote composto declarar e validar explicitamente essa fronteira.

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

`npm run compare:generation-trace -- --source <checkpoint> --trace <captura.json> --report <relatorio.json>` consome uma captura distinta com `kind: "generation"`. Além da mesma ligação obrigatória a `source.files`, checksums e `irFingerprint`, ela exige prompt, posições absolutas do prompt, limite/EOS, cada token gerado e sua posição, os logits binários usados para selecionar **cada** token, um snapshot KV BHSD pós-decode para **cada** token, logits terminais e o cache KV BHSD final. Cada snapshot e o cache final devem cobrir exatamente cada camada de atenção com KV próprio declarada pelo IR; um consumidor `kvSharing` usa exclusivamente a entrada do seu produtor e é rejeitado se serializar uma cópia própria. Assim, duas partes não podem omitir o mesmo estado nem inventar uma segunda propriedade do mesmo cache. Cada token também precisa ser o argmax dos logits do último passo de sequência, com empate decidido pelo menor ID e nenhum valor não-finito; portanto uma captura não pode chamar uma sequência de sampling de geração greedy. O executor repete prefill e cada decode incremental com pesos materializados por range; uma posição, token, logits de seleção, cache intermediário/final, payload ou checksum ausente/divergente torna o relatório `incomplete` ou `approximate`, nunca aprovação. Assim, a captura ainda deve vir de hooks de um runtime autoritativo; a fixture local apenas valida o contrato de consumo.

### Captura independente por kernels MLX (Llama, Mistral, Gemma 1/2 e Qwen 2/3 densos F32)

Quando há um Python com `mlx`, `numpy` e `safetensors`, o comando abaixo cria
uma captura de referência independente para o subconjunto explicitamente
suportado: Safetensors denso F32 com os adaptadores `llama`, `mistral`,
`gemma`, `gemma2`, `qwen2` ou `qwen3`. Ele recebe o IR já validado e executa cada operação por kernels MLX;
não detecta arquitetura, não dequantiza e não aceita variantes de RoPE fora de
`default/rotate_half`. A lista é fechada: Mistral cobre máscara causal de janela
deslizante no prefill e decode; Gemma 1 cobre escala de embedding, RMSNorm
`1 + weight` e GELU-tanh; Gemma 2 cobre quatro RMSNorms, normas Q/K por cabeça,
`query_pre_attn_scalar`, softcaps de atenção/logit e janela local; Qwen 2 cobre bias de atenção sem normalização Q/K;
Qwen 3 cobre normas Q/K em BHSD e bias nas projeções. Outros IRs aparentemente compatíveis continuam recusados até terem
um contrato e regressão independentes próprios.

Há também um caminho separado e fechado para `llama` cujos pesos quantizados
usam MLX `affine` U32 (bits 2/3/4/5/6/8, `group_size` 32 ou 64 validado, um único
contrato no checkpoint e `scales`/`biases` no mesmo dtype F32, F16 ou BF16,
com `biases` opcionais). O helper entrega os
bytes U32, parâmetros e `bits/group_size/mode="affine"` diretamente a
`mlx.core.dequantize(..., dtype=mlx.float32)` antes de executar os kernels; ele
nunca chama o materializador nativo nem o bridge candidato. Modos MLX como
`mxfp4`, `mxfp8` e `nvfp4`, parâmetros mistos, contratos por-módulo que mudam
bits/grupo e adaptadores quantizados não-Llama continuam recusados até terem
um contrato de captura e regressão próprios. Quando o formato não traz
`biases`, o helper passa uma matriz MLX F32 de zeros — a identidade explícita
da fórmula afim — porque a API de dequantização exige esse argumento.

```bash
npm run capture:mlx-trace -- --source ./model --output ./mlx-execution.json \
  --input-tokens 1,2 --python ./venv/bin/python \
  --model my-dense-llama --revision immutable-checksum
npm run compare:trace -- --source ./model --trace ./mlx-execution.json \
  --report ./mlx-execution-report.json --max-absolute-error 1e-5 --max-relative-error 1e-4
```

Adicionar `--max-new-tokens 4` produz uma captura `generation` contendo
logits completos de cada forward (a escolha usa somente a última posição),
snapshots KV pós-decode e logits terminais, que deve ser consumida por
`compare:generation-trace`. A evidência é independente do
executor escalar, mas ainda é limitada ao adaptador/precisão declarados; não
é uma alegação de equivalência com `transformers` nem habilita checkpoints
quantizados além do contrato affine Llama, outras famílias além das seis declaradas ou o pacote multimodal
Gemma 4.

Há também uma regressão cruzada de contêiner que grava o mesmo microcheckpoint
Llama denso em Safetensors e GGUF v3. O caminho GGUF reconstrói as dimensões
`[in,out]` declaradas pelo GGML para os pesos IR `[out,in]`, verifica o checksum
do arquivo `.gguf`, e consome a evidência F32 produzida pelo fixture
Safetensors para forward completo, cada operação, cache KV, logits e geração
greedy. Ela é uma prova lossless dentro da política escalar F32 do fixture —
não uma alegação de equivalência com um runtime externo ou de cobertura de
modelos reais/quantizados.

Uma segunda regressão executável cobre o `GGML_TYPE_Q8_0`: um Llama de uma
camada e largura 256 armazena todas as matrizes em blocos Q8_0 de 32 valores
com `d=0.5`. Um pacote GGUF F32 separado é construído diretamente da fórmula
declarada `F32[i] = F16(d) * int8(q[i])`, sem reutilizar o leitor candidato, e
fornece a evidência para cada operação, KV cache, logits e duas etapas greedy.
O comparador também rejeita um único bit alterado no payload quantizado pelo
checksum da captura. Isto prova uma fronteira sintética lossless dentro do
executor escalar F32 para Q8_0; não é uma comparação com llama.cpp nem uma
prova para os demais layouts GGML quantizados.

Uma regressão complementar cobre o `GGML_TYPE_Q8_1`: cada matriz de um Llama
de largura 256 usa blocos de 32 códigos `int8` assinados com escala `F32
d=0.25`, incluindo o domínio completo de `-128` a `127`. O writer constrói
independentemente o campo auxiliar obrigatório `F32 s=d*sum(qs)` para cada
bloco, enquanto o GGUF F32 pareado aplica somente a reconstrução declarada
`d*q` (sem reinterpretar `s` como offset ou escala F16). A evidência cobre
cada operação, cache KV pós-RoPE, logits e duas etapas greedy; uma alteração
em um campo `s` empacotado é recusada pelo checksum. É uma prova sintética
lossless na política escalar F32, não uma comparação com llama.cpp nem uma
validação de checkpoint real.

Uma terceira regressão executável percorre o `GGML_TYPE_Q3_K` pela fronteira
completa. Cada matriz de um Llama de largura 256 contém os quatro planos de
códigos de dois bits, os dois estados do `hmask[32]` e as 16 escalas assinadas
de seis bits por grupos de 16, incluindo ambos os sinais. Um GGUF F32 pareado
calcula diretamente `d*signedScale*(low2-(hmask?0:4))` com `d=0.5`, sem chamar
o leitor candidato, e fornece operações, cache KV pós-RoPE, logits e duas
etapas greedy. Uma alteração no payload empacotado é recusada pelo checksum da
captura. Isto é uma prova sintética lossless sob a política escalar F32 para
Q3_K, não uma comparação externa nem uma validação de checkpoint real.

Uma regressão complementar percorre o `GGML_TYPE_Q2_K` pela mesma fronteira
completa. Cada matriz do Llama de largura 256 usa os 16 nibbles independentes
de escala e mínimo, todos os quatro planos de códigos de dois bits e as bases
`d=0.5` e `dmin=0.25`. O GGUF F32 pareado calcula diretamente
`d*scale*code - dmin*minimum`, sem usar o leitor candidato, e fornece cada
operação, o cache KV pós-RoPE, logits e duas etapas greedy. Uma alteração no
payload compactado é recusada pelo checksum da captura. Isto demonstra somente
uma fronteira sintética lossless na política escalar F32 para Q2_K — não uma
comparação com llama.cpp nem validação de checkpoint real.

Uma quarta regressão executável percorre o `GGML_TYPE_Q4_K` por essa mesma
fronteira. Cada matriz de um Llama de largura 256 contém blocos de 256 valores
com `d=0.5`, `dmin=0.25`, códigos de nibble variados e todos os oito grupos de
escala/mínimo; os grupos 4..7 usam também os dois bits altos espalhados pelos
primeiros oito bytes. Um GGUF F32 pareado aplica diretamente a fórmula
declarada `d*scale*q - dmin*minimum`, sem chamar o decodificador candidato, e
fornece operações, KV, logits e duas etapas greedy. A alteração de um byte
empacotado é rejeitada pelo checksum. A classe continua sendo uma prova
sintética lossless na política escalar F32 — não uma comparação externa nem
uma prova para os outros tipos GGML.

Uma quinta regressão executável cobre o `GGML_TYPE_Q5_K` pela fronteira
completa. Cada matriz de um Llama de largura 256 usa o mesmo `d=0.5`,
`dmin=0.25` e os oito campos de escala/mínimo de seis bits, mas constrói
diretamente no pacote F32 pareado os códigos unsigned de cinco bits da fórmula
`d*scale*(low4 | high1<<4) - dmin*minimum`. Os 32 valores possíveis de cada
código aparecem no fixture, portanto tanto `qh[32]` quanto os nibbles baixos
e ambos os ramos de campos altos dos grupos 4..7 influenciam as operações,
cache KV, logits e duas etapas greedy comparadas. O pacote F32 não chama o
leitor candidato e uma alteração de byte empacotado é rejeitada pelo checksum.
Esta é uma prova sintética lossless dentro da política escalar F32, não uma
comparação com llama.cpp nem uma validação de checkpoint real.

Uma sexta regressão executável percorre o `GGML_TYPE_Q6_K` pela mesma
fronteira completa. Cada matriz do Llama de largura 256 contém os 64 códigos
centrados possíveis, distribuídos pelos oito planos de 32 valores de `ql` e
pelos dois bits altos de `qh`, e as 16 escalas `int8` por grupos de 16 incluem
ambos os sinais. O pacote GGUF F32 pareado calcula diretamente
`d*scale*(code-32)` com `d=0.5`, sem chamar o leitor candidato; dele vêm cada
operação, o cache KV pós-RoPE, logits e duas etapas greedy. Uma alteração no
payload empacotado é recusada pelo checksum da captura. Isso demonstra uma
fronteira sintética lossless sob a política escalar F32 para Q6_K, não uma
comparação com llama.cpp nem uma validação de checkpoint real.

Uma sétima regressão executável percorre o `GGML_TYPE_Q8_K` pela fronteira
completa. Cada bloco de 256 valores usa a escala binária32 `d=0.25` e todos os
256 códigos `int8` assinados, de `-128` a `127`, de modo que valores negativos,
zero e positivos participam das projeções. Um GGUF F32 pareado calcula
diretamente `d*q`, sem chamar o leitor candidato, e fornece cada operação, o
cache KV pós-RoPE, logits e duas etapas greedy. Um byte alterado no payload
empacotado é recusado pelo checksum da captura. Isso é somente uma prova
sintética lossless sob a política escalar F32 para Q8_K; não é uma comparação
com llama.cpp nem validação de checkpoint real.

`materializeReferenceF32Constants` em `src/materialize.ts` fecha a fronteira
entre o catálogo e o executor F32: reúne todos os `TensorRef` do IR, confirma
nome, shape, dtype e contrato de quantização contra o catálogo de origem e
materializa cada constante uma única vez. Safetensors densos usam o leitor de
intervalos F32/F16/BF16; MLX quantizado exige explicitamente o bridge com
`mlx.core.dequantize` e preserva a proveniência. GGUF denso F32/F16/BF16 usa o
seu próprio leitor de intervalos. Os únicos tipos GGML empacotados materializáveis
hoje são `Q2_K`, sob o contrato de 84 bytes por grupo de 256 (16 bytes de
escalas/mínimos em nibbles, quatro planos de códigos de dois bits em 64 bytes e
`F16 d`/`F16 dmin`, com `d*scale*q-dmin*minimum` por grupo de 16), `Q3_K`, sob
o contrato de 110 bytes por grupo de 256 (`F16 d`,
`hmask[32]`, 64 bytes com quatro planos de códigos de dois bits e 12 bytes com
16 escalas assinadas de seis bits, com `d*(scale-32)*(code-(mask?0:4))`),
`Q4_0`, sob o contrato exato de bloco de 18 bytes (`F16` scale + 16
nibbles, baixo `q[0..15]`, alto `q[16..31]`, código centrado por `-8`), `Q4_1`,
sob o contrato distinto de 20 bytes (`F16` scale, `F16` minimum e 16 nibbles,
com `d*q+m`), `Q4_K`, sob o contrato de 144 bytes por grupo de 256 (`F16 d`,
`F16 dmin`, 12 bytes com oito escalas/mínimos de seis bits e 128 bytes de
códigos; `d*scale*q - dmin*minimum`), `Q5_0`, sob o contrato de 22 bytes (`F16` scale + plano de 32
high bits + 16 nibbles, código centrado por `-16`), `Q5_1`, sob o contrato
distinto de 24 bytes (`F16` scale, `F16` minimum, plano de 32 high bits e 16
nibbles, com código unsigned de cinco bits e `d*q+m`), `Q5_K`, sob o contrato
de 176 bytes por grupo de 256 (`F16 d`, `F16 dmin`, 12 bytes de
escalas/mínimos, `qh[32]` e 128 bytes de códigos baixos; `d*scale*q -
dmin*minimum`), `Q6_K`, sob o contrato
de 210 bytes por grupo de 256 (`ql[128]`, `qh[64]`, 16 escalas `int8` por 16
valores e `F16 d` final, com código de seis bits centrado por `-32`), `Q8_0`, sob o contrato
de 34 bytes (`F16` scale + 32 `int8`), `Q8_1`, sob o contrato distinto de
40 bytes (`F32 d`, `F32 s=d*sum(qs)` e 32 `int8`, reconstruídos por `d*q`), e
`Q8_K`, sob o contrato de 260 bytes por grupo de 256 (`F32 d` e 256 `int8`,
reconstruídos por `d*q`). Q2_K, Q3_K, Q4_K, Q5_K, Q6_K e Q8_K exigem que a primeira dimensão GGML seja múltipla de 256; os demais acima
exigem múltiplos de 32. Os demais não são
reinterpretados como F32 até existir um decodificador por tipo verificado.

`GgufCatalogReader` em `src/gguf.ts` não depende de `gguf-py`: valida magic,
versão v2/v3, contagens seguras, metadata tipada (incluindo arrays), diretório,
alinhamento e intervalos de payload. Hoje expõe armazenamento GGML F32/F16/BF16 e os
contratos quantizados explícitos Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_1 e Q8_K de tamanho verificável, mantendo as dimensões
na ordem declarada pelo GGML; qualquer outro encoding empacotado é rejeitado com
seu tipo GGML até haver contrato de layout e dequantização específico. A leitura do catálogo não inventa papéis de
tensor nem uma convenção de layout de arquitetura. A exceção executável atual
é o adaptador separado `adaptGgufLlamaCatalog`: ele exige
`general.architecture=llama`, reconhece apenas os nomes GGUF Llama registrados
(`token_embd`, `output` e projeções `blk.N`) e converte somente essas matrizes
da ordem GGML declarada `[in,out]` para a forma IR `[out,in]`.

`compareGenerationTrace` cobre a evidência que não cabe em um forward isolado:
ele exige uma captura autoritativa do prompt e suas posições absolutas, cada
token greedy emitido, a posição em que cada token foi avaliado, os logits que
selecionaram cada token, o cache KV pós-RoPE imediatamente após cada decode,
logits terminais e o cache KV final por camada. Tokens iguais sem posições,
logits de seleção, snapshots de cache, logits terminais ou cache final
equivalentes não recebem aprovação. Captura ausente ou com shape incompatível é
`incomplete`; divergência medida é `approximate`; somente a captura completa
dentro da tolerância declarada pode ser `numerically-equivalent` (ou
`lossless-within-dtype` quando todos os valores comparados são exatos).

Uma regressão executável Safetensors F32 também atravessa a cadeia inteira para
Gemma 1 de uma camada: catálogo, lowering, materialização por range, cada
operação, KV pós-RoPE, logits e duas etapas greedy. Ela verifica os dois
desvios que não podem herdar a semântica Llama: embedding multiplicado por
`sqrt(hidden_size)` e RMSNorm com peso `1 + weight`; também preserva a
tradução explícita de `gelu_pytorch_tanh` para a operação GELU-tanh do IR e o
`lm_head` amarrado ao embedding. A captura é ligada a `config.json` e
`model.safetensors` por checksum e rejeita uma mutação do payload. Como as
outras fixtures locais, a prova é lossless somente dentro do executor escalar
F32 e não é uma comparação contra runtime Gemma/Transformers autoritativo.

Há também uma fronteira executável sintética para Qwen 3 de uma camada. Além
do catálogo, lowering, materialização por range, todas as operações, KV pós-RoPE
e duas etapas greedy, ela fixa a ordem semântica que distingue esse adaptador:
as normas RMS de Q e K recebem tensores já reorganizados em BHSD e normalizam
o eixo `head_dim` antes de RoPE. As quatro projeções de atenção possuem bias
porque `attention_bias: true` é declarado pelo pacote e precisa coincidir com
os tensores. A captura é ligada aos checksums de `config.json` e
`model.safetensors`; seu resultado é lossless somente no executor escalar F32
determinístico, não uma comparação com um runtime Qwen/Transformers
autoritativo.

A fixture sintética Q5_1 de 256 dimensões também percorre a fronteira completa
de execução e geração: o lado denso independente aplica `d*q+m` com `d=0.5`,
`m=-1` e todos os 32 códigos unsigned; o GGUF empacotado grava os códigos
`q[0..15]` nos nibbles baixos, `q[16..31]` nos altos e o quinto bit no plano
`qh`. A comparação liga cada operação, o cache KV pós-RoPE, logits e dois
passos greedy à fonte Q5_1 com checksum; uma mutação no plano `qh` é recusada
antes de qualquer relatório fiel. Isso demonstra reconstrução lossless sob a
política escalar F32 contra evidência F32 construída pela fórmula declarada,
não contra um runtime externo autoritativo.

A fixture sintética Q5_0 de 256 dimensões cobre a variante de cinco bits
centrada, que não possui o mínimo afim de Q5_1: a evidência densa independente
calcula `d*(q-16)` com `d=0.5` para todos os 32 códigos. O GGUF empacotado
exercita os dois planos de nibbles e o bit alto `qh`; a fronteira de execução e
geração compara todas as operações, cache KV, logits e dois passos greedy, e
rejeita uma mutação em `qh` pelo checksum. É prova sintética
lossless-within-dtype sob F32 escalar, não comparação com runtime externo.

A fixture sintética Q4_1 de 256 dimensões fecha a variante afim de quatro
bits: a evidência F32 independente aplica `d*q+m` com `d=0.5`, `m=-1` e os 16
códigos unsigned; o pacote GGUF grava `q[0..15]` nos nibbles baixos e
`q[16..31]` nos altos. Todas as operações, o cache KV pós-RoPE, logits e dois
passos greedy são comparados contra essa evidência, e uma mutação do `F16 m`
do último bloco é recusada pelo checksum antes do relatório. Assim como as
demais fixtures quantizadas, é prova sintética lossless-within-dtype sob F32
escalar, não uma comparação com runtime externo autoritativo.

A fixture sintética Q4_0 de 256 dimensões fecha a variante centrada de quatro
bits, semanticamente distinta de Q4_1 por não possuir mínimo afim: a evidência
F32 independente calcula `d*(q-8)` com `d=0.5` para todos os 16 códigos. O
GGUF empacotado grava `q[0..15]` nos nibbles baixos e `q[16..31]` nos altos;
todas as operações, cache KV pós-RoPE, logits e dois passos greedy são
comparados contra essa evidência, e a mutação do último byte que contém ambos
os nibbles é recusada pelo checksum antes do relatório. É prova sintética
lossless-within-dtype sob F32 escalar, não comparação com runtime externo.

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

O mesmo comando de trace também percorre a fronteira F64 completa para
Safetensors densos: payloads `valuesBase64` F64, todas as operações, KV
pós-RoPE, logits terminais e geração greedy são comparados sem estreitar os
bytes para F32. A validação exige que todos os tensores capturados tenham o
dtype declarado e que o leitor do contêiner implemente materialização F64
verificada; GGUF, quantização e F16/BF16 continuam recusados nesse caminho em
vez de serem promovidos implicitamente. O fixture regressivo altera valores
F64 além da precisão F32 e também prova a recusa por checksum de um payload
modificado. Continua sendo evidência sintética escalar, não equivalência a um
runtime externo.

O executor também aceita e devolve cache KV por camada para decoder incremental.
O contrato canônico é um `Map` indexado pela camada, com `key` e `value`
post-RoPE no layout `[batch, kv_heads, cached_sequence, head_dim]`. Ao receber
cache, ele precisa conter todas as camadas produtoras de KV e é concatenado às
projeções atuais antes da atenção; `position_ids` continua explícito e deve usar
as posições absolutas do token novo. A máscara aditiva, se presente no decode,
tem chave com o comprimento completo (cache + tokens atuais). Shapes incompletos
ou incompatíveis falham fechados. Para `kvSharing`, somente a camada produtora
persiste a entrada BHSD: a consumidora precisa declarar um `producerLayer`
anterior e lê o cache completo que esse produtor acabou de atualizar na mesma
execução. Não há cópia por consumidora; produtor ausente, ordem inválida ou
cache malformado falham fechados.

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

Uma regressão de trace cobre também esta fronteira de storage denso, não só a
leitura unitária: o mesmo Llama mínimo é codificado independentemente em
Safetensors e GGUF v3, cada um em `F16` e `BF16`, enquanto a evidência vem de
um pacote F32 pareado que não chama os leitores candidatos. Para os quatro
pacotes, o comparador confere cada operação, KV pós-RoPE, logits e duas etapas
greedy sem erro sob a política escalar F32, e uma mutação do payload é recusada
pelo checksum. Isso prova somente os valores já arredondados desses formatos
de storage no executor F32; não é uma alegação sobre a precisão de acumulação
ou kernels de um runtime externo.

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

Para um peso MLX `affine` quantizado, o leitor Safetensors materializa os U32
por range sem Python: exige `bits` em 2/3/4/5/6/8, packing exato de palavras
U32, `group_size`, scales densas e biases compatíveis quando declaradas, e
reconstrói `scale * code + bias` por grupo. O fluxo de códigos é contíguo por
linha: códigos de 3, 5 e 6 bits que atravessam uma fronteira U32 são reunidos
dos dois words, nunca truncados. Outros modos MLX usam
`TensorBridge.readMlxDequantizedF32`, que repassa o contrato explícito a
`mlx.core.dequantize`, verifica o shape de saída e devolve bytes F32 (não uma
lista JSON de números), marcados com a proveniência completa da quantização.
`executeReferenceF32` só aceita um
tensor quantizado se essa proveniência for idêntica ao `TensorRef` do IR;
buffer sem proveniência ou de outro esquema falha fechado. Isso habilita uma
execução F32 de fixtures/materializações declaradas, mas não estabelece a
política de dtype do modelo nem equivalência com um runtime MLX real. Um modo
MLX omitido também é rejeitado: `U32` + bit width não define um algoritmo.

Uma regressão de execução completa também percorre o MLX `affine` nativo: um
Llama de uma camada exercita grupos de 32 e 64 valores para todos os bits
aceitos (2/3/4/5/6/8), incluindo os códigos que cruzam palavras U32 em 3, 5 e
6 bits. O caso de 4 bits cobre biases F32 e, em uma captura separada, scales
e biases F16; os demais exercitam a identidade de bias zero exigida pelo
kernel quando o pacote não traz `.biases`. Um pacote
Safetensors F32 separado calcula `scale[group] * code + bias[group]` sem usar
o leitor candidato; seus outputs por operação, cache KV pós-RoPE, logits e dois
passos greedy são a evidência consumida pelo pacote MLX. A comparação exige os
checksums de `config.json` e `model.safetensors`, portanto alterar um byte U32
também invalida o trace. É uma fronteira sintética numericamente equivalente
sob o executor escalar F32, não uma captura autoritativa do runtime MLX.

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

# Model Decompiler v2

Reescrita do conversor de pesos para um IR matemático estruturado e verificável.

## Mudança principal

`MAX_FEATURES` não altera a rede. Ele limita apenas o preview embutido no JSON. Uma projeção continua sendo representada como:

```text
y[o] = Σ(i=0..in_features-1) x[i] * W[o,i] + b[o]
```

O IR guarda `inFeatures`, `outFeatures`, referência ao tensor, dtype e quantização. Os primeiros termos podem ser anexados para inspeção, mas nunca substituem a soma completa.

## Produto final: programa JSON autocontido

O objetivo final é decompilar um pacote `.safetensors` — acompanhado de sua
evidência semântica autoritativa — para um JSON que seja um programa de cálculo
literal. Não basta catalogar ou desenhar uma LLM. O JSON recebe somente as
entradas declaradas (`x[i]`/`x[t,d]`, IDs de token, posições e controles de
geração); os pesos, constantes e todas as atribuições intermediárias em ordem
de dependência até logits e tokens gerados ficam dentro dele.

Cada etapa precisa declarar o resultado nomeado, operação, entradas nomeadas,
dimensões, dtype, layout, casts, ordem de redução e estado. Nomes como
`x_embedding[i]`, `x_med[i]`, `x_alguma_variacao[i]` e
`x_para_proxima_camada[i]` exemplificam o nível de explicitação esperado.
Assim, a entrada da segunda camada aponta explicitamente para o cálculo
produzido pela primeira; nada depende de um "bloco Transformer" implícito. O
JSON final deve reproduzir um modelo suportado mesmo sem acesso ao checkpoint
de origem.

Para isso, toda constante ou peso exigido pelo replay precisa estar presente de
forma lossless no próprio JSON. Payloads binários codificados são aceitáveis
para tensores grandes, desde que o JSON também descreva dtype, endianness,
shape, layout e, para quantização, packing, parâmetros e fórmula determinística
de dequantização. Referências externas a shards são aceitáveis apenas no IR
interno de compilação, nunca no artefato final autocontido.

Ser agnóstico ao tipo de modelo não autoriza supor que pesos brutos revelem a
arquitetura. O contêiner Safetensors e o exportador são genéricos; cada
semântica deve vir de metadata autoritativa, adaptador validado ou extração de
grafo de runtime autoritativo. Se essa evidência não existir, o compilador falha
de forma explícita. O estado final buscado é: toda semântica Safetensors que
possa ser estabelecida é transformada nesse JSON literal autocontido, e não em
referências externas, resumos ou aproximações.

## Fontes

- Diretórios Hugging Face/Safetensors densos.
- Diretórios MLX/Safetensors quantizados, lendo `config.json`, incluindo overrides por módulo.
- Arquivos GGUF v2/v3 por um leitor nativo estrito de header, metadata e diretório; os subconjuntos densos Llama e Qwen 2/3 com layout de decoder registrado também podem chegar ao IR e à materialização F32 sem bridge Python.

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

### Exportação literal autocontida (Safetensors, MLX affine-U32 e GGUF Q8_0)

`--literal` troca o artefato de IR interno por um programa de cálculo que não
retém `source.path`, shard, offset ou referência externa. Para cada tensor
usado pelas atribuições, ele incorpora o payload original `F32`, `F16`, `BF16`,
`U32` ou o bloco `GGML_Q8_0` em base64 com shape, layout declarado e byte
order little-endian. Safetensors/MLX usam `row-major`; o stream GGUF Q8_0
declara `ggml-first-axis-contiguous`, em vez de fingir que seu bloco físico é
uma matriz F32 já materializada.
O programa declara uma atribuição de storage por constante:
`ieee-f32-little-endian`, `ieee-f16-to-f32`, `ieee-bf16-to-f32` ou, no
contrato MLX verificado, `mlx-affine-u32-to-f32`. Este último incorpora também
os tensors `scales` e `biases` e declara bits, group size, packing de códigos
contíguos LSB-first por palavra U32, dtype dos parâmetros e o arredondamento
BF16 após `scale * code + bias`. Portanto o limite de decode/cast até a
política escalar F32 é explícito e o exportador não substitui silenciosamente
valores de storage por uma matriz F32 materializada. Para `GGML_TYPE_Q8_0`, a
atribuição `ggml-q8-0-to-f32` incorpora os 34 bytes de cada bloco (escala
IEEE-754 binary16 + 32 códigos `int8` assinados), declara o tamanho/packing e
reconstrói cada valor como `F32(d * q)` no próprio programa literal.
As atribuições do modelo são ordenadas em `prelude`, camadas e `epilogue`, e
cada atenção declara sua transição de cache KV (`append-post-rope` ou
`reuse-producer`).

#### Fórmulas auditáveis com valores substituídos

O grafo não pode parar em `linear(x, weight)` ou `weight[o,i]` como se isso
fosse uma fórmula manual completa. A visualização/auditoria literal precisa
explicitar os índices, limites e ordem de redução, e substituir cada peso,
bias e constante endereçável pelo seu valor numérico decodificado no local em
que participa da equação. Assim, uma linha auditável é
`y[t,0] = F32(x[t,0] * 3.456812134 + x[t,1] * -0.125 + 0.75)`, e não apenas
uma referência opaca à matriz. Para não transformar uma E4B em um JSON ainda
mais impraticável, o artefato de produção pode conservar o payload binário
lossless; porém deve declarar o decoder e o mapeamento de índice suficientes
para derivar exatamente a mesma substituição sem o checkpoint original.

O contrato e o exemplo completo de duas camadas estão em
[`docs/literal-scalar-substitution-contract.md`](docs/literal-scalar-substitution-contract.md)
e
[`docs/examples/literal-scalar-substitution.example.json`](docs/examples/literal-scalar-substitution.example.json).

```bash
node dist/src/cli.js --source ./modelo --output ./modelo.literal.json --literal
```

`executeLiteralF32` e `generateLiteralF32` reconstroem os `Float32Array`
somente desses bytes incorporados e das atribuições de decode validadas. As
regressões removem o checkpoint antes de executar forward e dois passos greedy
para F32, F16, BF16, MLX affine U32 (inclusive parâmetros BF16 e códigos de 3
bits que cruzam palavras U32) e GGUF Q8_0. A regressão Q8_0 compara o replay
literal sem arquivo com o materializador nativo já estabelecido e também rejeita
um contrato de bloco alterado. Portanto esse caminho não pode cair de volta para
um shard local. MLX `mxfp4`, `mxfp8`, `nvfp4`, affine com `global_scale` e os
outros layouts GGUF quantizados continuam falhando fechado: cada um exige
payloads, parâmetros e atribuições de dequantização específicos, não uma matriz
F32 materializada escondida.

### Literal Gemma 4 composto denso

O pacote externo `gemma4` não passa por `compileModel`: essa rota continua a
rejeitar corretamente a redução silenciosa para `text_config`. O comando
explícito abaixo escolhe o adaptador composto já auditado e grava um único JSON
autocontido com os bytes densos de todos os pesos, inclusive os escalares
`shape: []` de `Gemma4ClippableLinear`:

```bash
node dist/src/cli.js --source ./gemma-4-E4B --output ./gemma4.literal.json --gemma4-composite-literal
```

`gemma4-composite-literal.ts` mantém as atribuições do prelude externo, tower
visual, tower de áudio e camadas textuais em escopos declarados. A validação
exige uma constante incorporada compatível para toda referência, recusa qualquer
tensor do catálogo sem atribuição semântica, declara os controles
`pixel_values`, vídeo, áudio, `mm_token_type_ids` e cache KV, e exige as
transições de máscara full/sliding antes de permitir replay. O texto interno
recebe somente o identificador virtual
`embedded://gemma4-composite-literal`; nenhum caminho de checkpoint é retido.
`executeGemma4CompositeLiteralF32` e
`generateGemma4CompositeLiteralF32` decodificam apenas esses payloads e chamam
o executor composto para prefill e greedy cached decode. A regressão remove o
mapa de bytes/tensores de uma fixture multimodal registrada antes do replay e
compara logits e dois tokens greedy.

Para o checkpoint obrigatório `google/gemma-4-E4B` BF16, o mesmo comando agora
escreve o artefato completo em streaming: ranges Safetensors de 12 MiB são
codificados em base64 sem acumular o pacote ou uma string de vários GiB na
heap. O resultado local de 21,325,917,078 bytes contém os 2,130 payloads
originais (15,992,314,836 bytes), tem SHA-256
`e81feb9061cabb9a1982c0b2ce890d540ea91f22034c99b1382e6283076c61e1` e é
auditado por `npm run audit:literal`. As 54 projeções/normas K/V locais dos
consumidores compartilhados são incorporadas com proveniência explícita em
`unreachableConstants`; o grafo usa apenas os KV do produtor declarado. A
evidência reproduzível está em
[`docs/validation/gemma4-e4b-literal-export-2026-07-16.md`](docs/validation/gemma4-e4b-literal-export-2026-07-16.md).

O leitor `gemma4-composite-literal-reader.ts` abre esse JSON em streaming: ele
indexa os offsets dos payloads base64 e valida a estrutura semântica sem
`JSON.parse` do artefato completo. Cada faixa de bytes é decodificada a partir
do próprio JSON, sem abrir shard Safetensors. Para auditar o índice ou uma
faixa específica (o comando não executa o modelo):

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --tensor model.language_model.embed_tokens.weight \
  --offset 0 --byte-length 4096
```

Isso fecha a fronteira de leitura seletiva; ainda não é uma alegação de replay
numérico BF16 ou de geração da E4B real. A medição do artefato real e o limite
de memória do executor atual estão em
[`docs/validation/gemma4-e4b-literal-reader-2026-07-16.md`](docs/validation/gemma4-e4b-literal-reader-2026-07-16.md).
Isso prova materialização completa e ausência do caminho do checkpoint no
artefato, mas ainda não é replay E4B sem fonte nem comparação numérica com o
runtime autoritativo.

`paged-dense.ts` é a fronteira seguinte: aceita somente matrizes literais
row-major densas `F32`/`F16`/`BF16`, lê linhas por `readTensorBytesRange` e
executa embedding e linear na mesma ordem escalar F32 do executor de
referência. A leitura é limitada por uma janela explícita e não aceita
quantização ou um payload sem decoder declarado. Isso permite que embeddings e
projeções enormes sejam consumidos sem formar um mapa F32 do pacote inteiro,
e fornece a base de armazenamento para a orquestração textual assíncrona
descrita a seguir.

`gemma4-paged-text.ts` agora fecha essa orquestração para o ramo textual: o
interpretador assíncrono lê embeddings e projeções como faixas limitadas do
artefato indexado, lê vetores de norma/escala sob o mesmo orçamento explícito,
e executa PLE, RMSNorm, RoPE, atenção, KV por camada produtora, residual, MLP,
norma final e projeção de vocabulário com os mesmos kernels escalares F32 do
executor de referência. Ele nunca abre o checkpoint: recebe somente o leitor
do JSON literal já indexado. O comando deliberadamente não recebe entradas
visuais, vídeo ou áudio; esses ramos continuam fail-closed até terem kernels
de armazenamento paginado próprios.

```bash
npm run replay:gemma4-paged-text -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --input-ids 2,106,3 --max-new-tokens 1 --output ./paged-text-report.json
```

O relatório inclui hash dos logits, tokens greedy, produtores KV, janela de
leitura e RSS. A regressão atual prova equivalência byte-a-byte de prefill e
dois passos cached contra o executor eager apenas para a fixture Gemma 4
registrada após remover os tensores de origem. Não há ainda execução E4B real
nem comparação com runtime autoritativo; portanto o marcador de checkpoint
Gemma 4 continua proibido.

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

O limite é de 100 handoffs aceitos, configurado em `agent-loop.config.json`.
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

O runner usa `gpt-5.6-terra` com esforço de raciocínio `high`. Cada invocação
é uma sessão autônoma de engenharia: o agente reavalia o sistema, decide a
fronteira estratégica de maior impacto, executa uma entrega coesa através das
camadas necessárias e deixa uma proposta técnica para a próxima sessão — não
uma sequência de microtarefas derivadas mecanicamente do commit anterior.

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
  ausente quando o config não declara o peso de saída amarrado ao embedding
  (exceto o default semântico registrado `true` de Gemma 2);
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

## Benchmark nativo reproduzível

`npm run benchmark:native -- --output docs/benchmarks/<arquivo>.json --samples 7 --warmup-samples 2`
gera uma linha de base de wall-clock para os caminhos que hoje são executáveis
sem bridge externo. O harness cria e remove três fixtures Llama determinísticas
de uma camada e largura 256: Safetensors F32 denso, Safetensors MLX affine U32
4-bit/grupo-32 (com `scales` e `biases` F32), e GGUF v3 `Q8_0`. O relatório
persiste os SHA-256 dos payloads, a topologia, tokens de prefill/decode, versão
do Node, plataforma, arquitetura, amostras em nanossegundos, mínimo, mediana e
p95 para cada estágio. Também registra bytes exatos dos constantes F32
materializados e um snapshot de `rss`/heap/ArrayBuffer do processo:

- inspeção de catálogo;
- lowering de IR (após abrir o catálogo);
- materialização por ranges (após lowering);
- prefill F32 (após materialização);
- um decode incremental com o KV cache do prefill (após preparar esse cache).

Logo, criação de fixture, checksum, abertura para as etapas posteriores,
lowering para as etapas de materialização e materialização para as etapas de
executor ficam fora da janela medida. Os testes verificam contrato e cobertura
das três rotas, mas não impõem limite de tempo: o relatório é evidência local,
não uma alegação comparável entre máquinas ou uma prova de fidelidade contra um
runtime autoritativo. A primeira medição comprometida está em
`docs/benchmarks/native-fixture-baseline-2026-07-15.json`.

## Limites atuais

O adaptador atual cobre blocos decoder-only auditáveis de Llama, Mistral, Qwen 2/3 e Gemma 1/2/3-text, além do núcleo `gemma4_text` isolado. Este último declara PLE empacotado, projeção/contexto normalizado, fatia PLE por camada, RoPE default/proporcional, RMSNorm sem escala de V, KV por produtor e `layer_scalar` como atribuições explícitas. A política executável F32 é uma referência escalar e ainda não é uma alegação de fidelidade BF16 ao runtime oficial. Modelos com código remoto, state-space layers, linear attention, MoE, multimodal completo, Gemma 3n/4 composto ou layouts QKV especiais precisam de adaptadores próprios ou extração do grafo do runtime oficial.

Um pacote composto não é convertido implicitamente em um checkpoint textual. Quando o `config.json` externo declara `audio_config` ou `vision_config` e tokens de modalidade, enquanto `text_config` descreve outro `model_type`, o compilador falha antes da seleção do adaptador: a injeção desses tokens, os towers e sua ordem no forward pass fazem parte da função. Um futuro adaptador de texto só pode reutilizar esse submodelo depois que um adaptador do pacote composto declarar e validar explicitamente essa fronteira.

### Auditoria do contrato Gemma 4 denso

`npm run audit:gemma4 -- --source <diretório-Gemma4> --output <audit.json>` é
uma fronteira de evidência para o checkpoint obrigatório — não é um adaptador e
não produz IR. Ele exige o pacote composto `gemma4` denso em Safetensors,
`text_config=gemma4_text`, os towers de visão e áudio, e os layouts registrados
do checkpoint oficial (`model.language_model`, `model.vision_tower`,
`model.audio_tower`, `model.embed_vision`, `model.embed_audio`). Em seguida
verifica os shapes de PLE, quatro RMSNorms, projeções por tipo de atenção,
RoPE default/proporcional e a propriedade KV compartilhada. Para os towers,
valida patch/posição/pooling visual, subsampling/local-convolution/atenção por
chunks do áudio, os projetores multimodais e cada limite escalar de
`Gemma4ClippableLinear` antes e depois da projeção. O relatório declara a
camada produtora KV para cada consumidor e calcula SHA-256 dos arquivos de
identidade. A primeira evidência para o E4B BF16 está em
[`docs/validation/gemma4-e4b-source-audit-2026-07-16.json`](docs/validation/gemma4-e4b-source-audit-2026-07-16.json).

O lowering estrutural do subgrafo textual do mesmo catálogo BF16 está registrado
em [`docs/validation/gemma4-e4b-text-core-lowering-2026-07-16.md`](docs/validation/gemma4-e4b-text-core-lowering-2026-07-16.md).

O branch visual agora possui um programa de atribuições e executor F32
separados em [`src/gemma4-vision.ts`](src/gemma4-vision.ts), documentados em
[`docs/validation/gemma4-vision-lowering-2026-07-16.md`](docs/validation/gemma4-vision-lowering-2026-07-16.md).
Ele preserva o affine de pixels, lookup x/y, RoPE 2-D, `Gemma4ClippableLinear`,
atenção bidirecional, pool espacial, escala F32, projeção e a cardinalidade do
`masked_scatter` de imagens.

O branch de áudio agora possui o mesmo limite executável explícito em
[`src/gemma4-audio.ts`](src/gemma4-audio.ts), com evidência em
[`docs/validation/gemma4-audio-lowering-2026-07-16.md`](docs/validation/gemma4-audio-lowering-2026-07-16.md).
Ele não aproxima o encoder como um Conformer genérico: declara os dois Conv2d
com máscaras `::2`, LayerNorm por canal, posição relativa, atenção local por
blocos com shift relativo e softcap, FFNs/clips/resíduos, GLU/convolução causal
profunda, projeções e stripping/scatter de tokens de áudio. Também continua a
ser apenas uma feature branch F32: a montagem texto+imagem+áudio+vídeo, casts
BF16, cache e geração do pacote composto seguem pendentes.

O programa externo composto está em
[`src/gemma4-composite.ts`](src/gemma4-composite.ts) e sua evidência em
[`docs/validation/gemma4-composite-prelude-2026-07-16.md`](docs/validation/gemma4-composite-prelude-2026-07-16.md).
Ele declara a substituição inicial de todos os IDs multimodais por PAD, a PLE
de identidade desses PADs, o scatter ordenado imagem → vídeo → áudio e a
projeção/contexto PLE após os três scatters antes de entrar no núcleo textual.
O executor F32 exige todas essas saídas de prelude nomeadas para chamar as
camadas textuais; portanto não há atalho implícito de embedding. Para
`use_bidirectional_attention="vision"`, também declara `vision_block_sequence_ids`
e duas máscaras completas por camada: `full_attention` permanece causal,
enquanto `sliding_attention` é `sliding_window AND (causal OR mesmo bloco
image/video)`. O prefill multimodal usa essas máscaras; o decode incremental
remove `mm_token_type_ids` e reutiliza apenas o KV do produtor, como o runtime
autoritativo. Máscara 4-D fornecida pelo chamador junto com esses IDs, ou IDs
multimodais no decode com cache, falham fechados. A literalização desse fluxo
F32 agora incorpora cada tensor e reexecuta uma fixture multimodal sem o source;
o contrato BF16 de E4B, a captura autoritativa e o diferencial de operações/KV
ainda falham fechados. Logo o E4B denso não tem
marcador `gemma4-dense-lossless`.

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
quantização, tokens, posições absolutas de execução (quando não começam em
zero) e política de dtype. O relatório mede erro absoluto e
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
usam MLX `affine` U32 (bits 2/3/4/5/6/8 e `group_size` 32 ou 64 validados por
matriz). O checkpoint pode ter overrides explícitos por módulo; a captura
preserva a lista ordenada de cada contrato `bits/group_size/dtype` na
proveniência, em vez de resumir a quantização como uma largura única. Cada
matriz exige `scales` e `biases` opcionais no mesmo dtype F32, F16 ou BF16. O helper entrega os
bytes U32, parâmetros e `bits/group_size/mode="affine"` diretamente a
`mlx.core.dequantize(..., dtype=mlx.float32)` antes de executar os kernels; ele
nunca chama o materializador nativo nem o bridge candidato. Modos MLX como
`mxfp4`, `mxfp8` e `nvfp4`, parâmetros mistos na mesma matriz e adaptadores
quantizados não-Llama continuam recusados até terem um contrato de captura e
regressão próprios. Quando o formato não traz
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

Um replay completo contra o checkpoint público, imutável e denso F32
`hf-internal-testing/tiny-random-LlamaForCausalLM` está registrado em
[`docs/validation/tiny-random-llama-mlx-2026-07-15.md`](docs/validation/tiny-random-llama-mlx-2026-07-15.md).
Ele cobre catalogação Safetensors, lowering Llama, 41 operações, ambos os
caches KV, logits e três passos greedy, todos contra MLX 0.32. O relatório
preserva a distinção essencial: é uma comparação numérica por kernels MLX
contra o executor candidato, não uma captura do `transformers` que estabelece
de modo independente a semântica da arquitetura.

### Captura semântica nativa por Transformers (Llama, Qwen 2 e Gemma 2 F32)

`capture:transformers-trace` é o caminho separado para evidência semântica do
runtime oficial: ele abre um pacote local denso F32 por
`LlamaForCausalLM`, `Qwen2ForCausalLM` ou `Gemma2ForCausalLM` do Hugging Face, fixa `transformers==4.57.1` e
`torch==2.7.1`, força atenção `eager` e não envia o IR candidato ao helper
Python. O adaptador seleciona explicitamente `--adapter llama|qwen2|gemma2`:
Llama exige projeções sem bias, Qwen 2 exige bias em Q/K/V e ausência de bias
em `o_proj`, e Gemma 2 exige quatro RMSNorms, embedding escalado, atenção
local/global declarada, `query_pre_attn_scalar`, ambos os softcaps e a ligação
de saída declarada/default. No runtime Gemma 2 4.57.1, Q/K norms não existem no
`Gemma2Attention`; um pacote que as declare é recusado por este adaptador em
vez de ser apresentado como evidência nativa. Todos exigem RoPE padrão sem
scaling, F32 e `eager`.
Ele instala hooks para todas as
fronteiras estáveis do IR e prova, antes de escrever a captura, que a execução
instrumentada preserva bit a bit logits e `DynamicCache` canônico BHSD da
execução não instrumentada. Famílias, storage, dtypes, versões ou semânticas
fora desse contrato falham fechadas.

```bash
npm run capture:transformers-trace -- --adapter qwen2 --source ./model --output ./transformers-execution.json \
  --input-tokens 1,2,3 --position-ids 0,1,2 \
  --python /path/to/pinned-transformers/bin/python \
  --model my-llama --revision immutable-revision
npm run compare:trace -- --source ./model --trace ./transformers-execution.json \
  --report ./transformers-execution-report.json --max-absolute-error 1e-5 --max-relative-error 1e-4
```

Com `--max-new-tokens`, a mesma fronteira captura logits de seleção, cada KV
pós-decode, logits terminais e o cache final para `compare:generation-trace`.
Ao contrário da captura MLX, esta evidência não deriva a semântica de forward
do IR candidato. Ela ainda é limitada aos contratos Llama/Qwen 2/Gemma 2 F32 e às
versões declaradas; não generaliza para arquiteturas, formatos ou quantizações
não instrumentados. Os replays públicos imutáveis estão em
[`docs/validation/tiny-random-llama-transformers-2026-07-16.md`](docs/validation/tiny-random-llama-transformers-2026-07-16.md)
e [`docs/validation/tiny-dummy-qwen2-transformers-2026-07-16.md`](docs/validation/tiny-dummy-qwen2-transformers-2026-07-16.md),
além de [`docs/validation/tiny-random-gemma2-transformers-2026-07-16.md`](docs/validation/tiny-random-gemma2-transformers-2026-07-16.md).

Há também uma regressão cruzada de contêiner que grava o mesmo microcheckpoint
Llama denso em Safetensors e GGUF v3. O caminho GGUF reconstrói as dimensões
`[in,out]` declaradas pelo GGML para os pesos IR `[out,in]`, verifica o checksum
do arquivo `.gguf`, e consome a evidência F32 produzida pelo fixture
Safetensors para forward completo, cada operação, cache KV, logits e geração
greedy. Ela é uma prova lossless dentro da política escalar F32 do fixture —
não uma alegação de equivalência com um runtime externo ou de cobertura de
modelos reais/quantizados.

As mesmas fronteiras executáveis agora cobrem Qwen 2 e Qwen 3 em GGUF v3
denso. A regressão Qwen 2 preserva os biases Q/K/V e a ausência de bias em
`o_proj`,
prova que não introduz os Q/K norms que pertencem apenas ao Qwen 3; a regressão
Qwen 3 preserva esses norms no tensor BHSD após `reshape_heads`. Ambas partem
de um Safetensors F32 pareado, verificam a orientação `[in,out]` do GGML para
cada matriz registrada, todas as operações, o KV pós-RoPE, logits terminais e
duas etapas greedy contra uma captura com checksum do arquivo GGUF. O resultado
é evidência sintética lossless na política escalar F32, não uma comparação com
um runtime externo nem uma validação de checkpoint publicado.

Essa fronteira também cobre Qwen 2 e Qwen 3 com `GGML_TYPE_Q8_0`. Cada matriz
de largura 32 é empacotada em blocos de 32 códigos `int8` com `d=0.5`, enquanto
as normas e os biases de atenção declarados permanecem F32 conforme os tensores
declarados. Um GGUF F32 pareado é construído diretamente da fórmula
`F32[i] = F16(d) * int8(q[i])`, sem chamar o leitor candidato. Para ambas as
arquiteturas, a regressão confirma todas as operações, KV pós-RoPE, logits e
duas etapas greedy; Qwen 2 mantém a ausência de normas Q/K e Qwen 3 exige suas
normas BHSD. É evidência sintética lossless para a combinação declarada de
layout Qwen e Q8_0, não uma equivalência com llama.cpp nem cobertura de outros
formatos GGML quantizados ou de um checkpoint publicado.

A mesma fronteira agora cobre Qwen 2 e Qwen 3 com `GGML_TYPE_Q4_0`, cuja
semântica centrada é diferente de Q8_0: cada matriz de largura 32 usa blocos
de 18 bytes com `F16 d=0.5` e nibbles baixo/alto, e o pacote F32 pareado é
construído diretamente por `F32[i] = d * (q[i] - 8)`. O fixture percorre todos
os 16 códigos e ambos os nibbles em cada matriz, mantendo normas e os quatro
biases de atenção densos; Qwen 2 continua sem Q/K norms e Qwen 3 exige os
dois norms BHSD. Para ambas, todas as operações, KV pós-RoPE, logits e duas
etapas greedy são comparadas com a testemunha F32 independente. Isso é
evidência sintética lossless limitada ao contrato Qwen/Q4_0 e à política
escalar F32, não equivalência com llama.cpp nem validação de checkpoint
publicado.

Qwen 2 e Qwen 3 também possuem replay completo com `GGML_TYPE_Q4_K`, um
contrato afim diferente que usa blocos de 256 valores: `F16 d`, `F16 dmin`,
oito escalas e oito mínimos de seis bits, e 128 bytes de nibbles. As matrizes
de largura 256 do fixture percorrem os oito grupos, todos os códigos de quatro
bits e tanto os campos diretos dos grupos 0..3 quanto os bits altos divididos
dos grupos 4..7. A testemunha GGUF F32 é construída diretamente por
`d * scale[group] * q - dmin * minimum[group]`; normas e os biases de atenção declarados
atenção continuam F32. Para os dois layouts, a regressão compara todas as
operações, KV pós-RoPE, logits terminais e duas etapas greedy, mantendo a
ausência de Q/K norms no Qwen 2 e exigindo as normas BHSD no Qwen 3. É
evidência sintética lossless dentro da política F32 declarada para
Qwen/Q4_K, não equivalência com llama.cpp nem validação de checkpoint
publicado.

Qwen 2 e Qwen 3 também possuem replay completo com `GGML_TYPE_Q5_K`, que
preserva a mesma hierarquia afim de escala/mínimo do Q4_K mas acrescenta
`qh[32]` como quinto bit independente de cada código. As matrizes de largura
256 do fixture percorrem os oito grupos, todos os códigos unsigned de cinco
bits, nibbles baixo/alto, ambos os campos altos divididos dos grupos 4..7 e
cada posição de `qh`. A testemunha GGUF F32 calcula diretamente
`d * scale[group] * (low4 | high1 << 4) - dmin * minimum[group]`; normas e
os biases de atenção declarados permanecem F32. Para os dois layouts, a regressão
compara todas as operações, KV pós-RoPE, logits terminais e duas etapas greedy,
mantendo a ausência de Q/K norms no Qwen 2 e exigindo as normas BHSD no Qwen
3. É evidência sintética lossless dentro da política F32 declarada para
Qwen/Q5_K, não equivalência com llama.cpp nem validação de checkpoint
publicado.

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

Qwen 2 e Qwen 3 também percorrem `GGML_TYPE_Q6_K` pela fronteira completa,
sem emprestar a semântica de atenção do Llama. Os dois GGUFs de largura 256
quantizam todas as matrizes com os 64 códigos centrados, os oito planos de 32
valores de `ql`/`qh` e as 16 escalas `int8` por grupos de 16; o par F32 calcula
diretamente `d*scale*(code-32)` com `d=0.5`, sem ler o payload candidato. As
projeções de Qwen preservam seus biases densos, e somente Qwen 3 preserva as
normas Q/K em BHSD antes de RoPE. Para cada layout, a captura ligada ao checksum
compara todas as operações, cache KV pós-RoPE, logits terminais e duas etapas
greedy. Isso é evidência sintética lossless na política escalar F32, não
equivalência contra um runtime Qwen externo nem validação de checkpoint real.

Qwen 2 e Qwen 3 também percorrem `GGML_TYPE_Q3_K` pela fronteira completa.
Cada matriz de largura 256 usa as 16 escalas assinadas de seis bits, os dois
estados de `hmask[32]` e os quatro planos de códigos baixos de dois bits; o
GGUF F32 pareado calcula diretamente `d*scale*(low2-(hmask?0:4))` com
`d=0.5`, sem ler o payload candidato. As projeções preservam os biases de atenção declarados,
densos de Qwen, enquanto apenas Qwen 3 mantém as normas Q/K em BHSD antes de
RoPE. Para ambos os layouts, a captura ligada ao checksum compara todas as
operações, cache KV pós-RoPE, logits terminais e duas etapas greedy. Isto é
evidência sintética lossless na política escalar F32 para Qwen/Q3_K, não uma
equivalência contra runtime externo nem uma validação de checkpoint publicado.

Qwen 2 e Qwen 3 também percorrem `GGML_TYPE_Q2_K` pela fronteira completa.
Cada matriz de largura 256 usa os 16 nibbles independentes de escala/mínimo e
os quatro planos de códigos de dois bits; o GGUF F32 pareado calcula
diretamente `d*scale*code-dmin*minimum`, com `d=0.5` e `dmin=0.25`, sem ler o
payload candidato. As projeções preservam os biases de atenção declarados por Qwen,
enquanto apenas Qwen 3 mantém as normas Q/K em BHSD antes de RoPE. Para ambos
os layouts, a captura ligada ao checksum compara todas as operações, cache KV
pós-RoPE, logits terminais e duas etapas greedy. Isto é evidência sintética
lossless na política escalar F32 para Qwen/Q2_K, não equivalência contra um
runtime externo nem validação de checkpoint publicado.

Qwen 2 e Qwen 3 também percorrem `GGML_TYPE_Q8_K` pela fronteira completa.
Cada matriz de largura 256 usa o bloco de 260 bytes com escala `F32 d=0.25`
e os 256 códigos `int8` assinados, de `-128` a `127`; o GGUF F32 pareado
calcula diretamente `d*q`, sem ler o payload candidato. As projeções preservam
os biases de atenção declarados por Qwen, enquanto apenas Qwen 3 mantém as normas Q/K
em BHSD antes de RoPE. Para ambos os layouts, a captura ligada ao checksum
compara todas as operações, cache KV pós-RoPE, logits terminais e duas etapas
greedy. Isto é evidência sintética lossless na política escalar F32 para
Qwen/Q8_K, não equivalência contra runtime externo nem validação de checkpoint
publicado.

Qwen 2 e Qwen 3 também percorrem `GGML_TYPE_Q8_1` pela fronteira completa.
Cada matriz de largura 256 usa blocos de 40 bytes: escala `F32 d=0.25`, campo
auxiliar `F32 s=d*sum(qs)` e 32 códigos `int8` assinados, cobrindo o domínio
de `-128` a `127`. O GGUF F32 pareado calcula apenas `d*q`, sem ler o payload
candidato nem interpretar `s` como offset ou escala F16; o writer verifica que
`s` é persistido independentemente em cada bloco. As projeções preservam os
biases de atenção densos declarados de Qwen, enquanto apenas Qwen 3 mantém as normas Q/K em
BHSD antes de RoPE. Para ambos os layouts, a captura ligada ao checksum compara
todas as operações, cache KV pós-RoPE, logits terminais e duas etapas greedy;
uma mutação no campo `s` final é recusada antes do relatório. Isto é evidência
sintética lossless na política escalar F32 para Qwen/Q8_1, não equivalência
contra runtime externo nem validação de checkpoint publicado.

As três variantes restantes de bloco curto também possuem replay Qwen 2/3
completo, sem tratar sua largura de bits como uma semântica comum. `Q4_1` usa
`F16 d`, `F16 m` e nibbles unsigned para reconstruir `d*q+m`; `Q5_0` combina
nibbles e `qh` em um código centrado para `d*(q-16)`; e `Q5_1` usa o mesmo
packing de cinco bits com mínimo afim, `d*q+m`. Cada pacote F32 pareado aplica
sua fórmula declarada diretamente, enquanto o GGUF empacotado mantém as normas
e os biases de atenção declarados de Qwen densos. Para cada variante e ambos os
layouts, a captura ligada ao checksum compara todas as operações, cache KV
pós-RoPE, logits terminais e duas etapas greedy; Qwen 2 continua sem normas
Q/K e Qwen 3 mantém suas normas BHSD antes de RoPE. Isto é evidência sintética
lossless na política escalar F32 para Qwen/Q4_1, Qwen/Q5_0 e Qwen/Q5_1, não
equivalência contra runtime externo nem validação de checkpoint publicado.

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
é o adaptador separado `adaptGgufDecoderCatalog`: ele seleciona exclusivamente
`general.architecture=llama`, `qwen2` ou `qwen3`, reconhece apenas o registro
documentado de matrizes decoder (`token_embd`, `output` e projeções `blk.N`) e
converte somente essas matrizes da ordem GGML declarada `[in,out]` para a forma
IR `[out,in]`. No Qwen 3, os vetores registrados `attn_q_norm` e
`attn_k_norm` continuam vetores e são baixados depois do reshape de cabeças;
um nome coincidente sob qualquer outra arquitetura não recebe orientação nem
semântica implícita.

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
dos dois words, nunca truncados. Quando os parâmetros affine são BF16, o
leitor reproduz o arredondamento BF16 do resultado `scale*code+bias` antes do
cast de saída F32 observado no kernel MLX; ampliar apenas os parâmetros para
F32 mudaria pesos de códigos altos. Scales e biases com dtypes diferentes são
rejeitados nessa rota nativa. Outros modos MLX usam
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
6 bits. O caso de 4 bits cobre biases F32 e, em capturas separadas, scales e
biases F16 e BF16; uma regressão de contratos mistos cobre raízes 4-bit/grupo
32 F32 e overrides MLP 8-bit/grupo 64 BF16. Os demais exercitam a identidade
de bias zero exigida pelo kernel quando o pacote não traz `.biases`. Um pacote
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

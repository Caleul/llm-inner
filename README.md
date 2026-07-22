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

O leitor streaming também materializa essa vista diretamente do artefato Gemma
4 real. `--list-operations` expande a ordem composite completa: instancia a
torre visual separadamente nos pontos de chamada de imagem e vídeo, instancia
áudio no seu ponto de chamada e entra no texto preparado diretamente na camada
0, sem repetir o embedding/PLE standalone. Cada instância compartilhada tem ID
único, `definitionId` e `invocationId`; pixels e posições de vídeo possuem
predecessores flatten distintos. O índice devolve predecessor, consumidores e
vizinhos de cada atribuição, e rejeita IDs/outputs duplicados ou um produtor que
apareça depois do consumidor. `--operation` seleciona uma
coordenada escalar e decodifica do payload incorporado cada peso/escala usado,
incluindo bits de storage, decoder IEEE, literal F32 e a agenda exata de
redução/cast. Por padrão uma linear expande todos os termos; uma janela só é
permitida quando `--input-start` e `--input-count` são ambos explícitos, e o
resultado fica marcado `complete: false` e `omittedTerms > 0`:

O relatório-base resume a identidade imutável sem expandir os payloads Base64
de metadata; assim `tokenizer.json` continua integralmente incorporado, mas não
transforma toda inspeção em dezenas de megabytes. Desde o schema 59, cada shard
Safetensors também declara uma cobertura byte a byte: prefixo/header e qualquer
gap estrutural ficam incorporados, enquanto cada `data_offsets` aponta para uma
única constante literal. `--source-file` navega qualquer arquivo do pacote por
janela explícita, inclusive uma janela que cruza header e tensor, devolvendo os
bytes em Base64, o SHA-256 da janela e o compromisso do arquivo inteiro. A rota
não duplica weights: ela lê os mesmos payloads que `--tensor`.

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source-file tokenizer.json --source-offset 0 --source-byte-length 4096 \
  --output /tmp/gemma4-tokenizer-window.json

# Recompõe e re-hasheia todos os arquivos do pacote sem abrir o checkpoint.
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-source-files \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-source-reconstruction.json
```

A reconstrução e o re-hash reais de todos os arquivos com o checkpoint
fisicamente indisponível estão em
[`docs/validation/gemma4-e4b-source-file-reconstruction-2026-07-20.md`](docs/validation/gemma4-e4b-source-file-reconstruction-2026-07-20.md).

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --operation layer_0_q_proj --output-coordinate 0,0,0 \
  --input-start 0 --input-count 8 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/q-proj-scalar-view.json
```

Cada `learnedScalar` retornado por essa vista agora inclui
`payloadIntegrity`: o range exato de bytes usado pelo decoder, o SHA-256 do
payload inteiro, todos os chunks canônicos Base64 que foram autenticados antes
da leitura e ponteiros JSON para a entrada em `payloadIntegrity` e para a seção
correspondente do `integrityManifest`. A mesma cadeia é devolvida por
`--tensor`. Portanto o literal numérico não fica ligado apenas a nome, índice e
bits de storage: o relatório mostra também como esses bits chegam ao root
SHA-256 estrutural do próprio artefato. Ranges que cruzam chunks enumeram todos
os chunks cobertos em ordem; corrupção em qualquer um deles falha antes do
decode escalar.

A prova foi exercitada no artefato E4B real com o checkpoint fisicamente
indisponível em
[`docs/validation/gemma4-e4b-integrity-bound-scalar-provenance-2026-07-20.md`](docs/validation/gemma4-e4b-integrity-bound-scalar-provenance-2026-07-20.md).

Embeddings exigem `--token-id`, pois o token continua sendo uma entrada do
chamador, não um default inventado. Operações e coordenadas sem contrato
escalar registrado falham fechado. A execução contra a E4B real com a fonte
indisponível está em
[`docs/validation/gemma4-e4b-source-removed-scalar-navigation-2026-07-17.md`](docs/validation/gemma4-e4b-source-removed-scalar-navigation-2026-07-17.md).
A expansão composite corrigida e uma linha visual de 768 pesos substituídos
estão em
[`docs/validation/gemma4-e4b-expanded-calculation-order-2026-07-17.md`](docs/validation/gemma4-e4b-expanded-calculation-order-2026-07-17.md).

Desde o schema Gemma 4 literal v2, o artefato também carrega a geração greedy como programa,
em vez de depender da implementação TypeScript para completar logits em
tokens. `--show-generation-program` expõe doze atribuições ordenadas para
prefill, posição inicial, linha final de logits, argmax com desempate pelo menor
ID, append do token, avanço de posição, entrada incremental, nova execução do
programa declarado, snapshot KV, parada EOS e seleção dos logits/cache
terminais. O contrato deixa explícito que
imagem/vídeo/áudio e `mm_token_type_ids` participam apenas do prefill, enquanto
o decode usa o token único, a posição avançada e o cache pós-RoPE;
`max_new_tokens` é marcado como obrigatório especificamente no modo geração.
Uma regra,
input ou transição alterada faz o leitor falhar fechado:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --show-generation-program \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-generation-program.json
```

Para sair dos templates simbólicos e seguir uma geração finita operação por
operação, o mesmo leitor instancia `step`, resolve produtores/consumidores e
liga cada prefill/decode à expansão completa do forward declarado:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-generation-operations --generation-max-new-tokens 2 \
  --output /tmp/gemma4-generation-navigation.json

npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --generation-operation 'generation_argmax[0]' \
  --generation-max-new-tokens 2
```

A navegação mantém uma única expansão compartilhada do forward para não
duplicar milhares de IDs por passo. A vista de cálculo do argmax expõe o scan
ascendente completo do vocabulário, a rejeição de logits não finitos e o
desempate que preserva o menor token ID; append de token, avanço de posição,
entrada incremental, cache, EOS e seleção terminal também possuem fórmulas
concretas por passo. A validação E4B com a fonte ausente está em
[`docs/validation/gemma4-e4b-source-removed-generation-navigation-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-generation-navigation-2026-07-18.md).

O schema v3 incorporou `calculationDomains`: cada definição composite,
vision, audio e text declara o dtype de saída, layout, shape simbólico, eixos,
bounds de coordenada e política numérica/redução. O leitor instancia os mesmos
domínios para imagem e vídeo sem duplicar definições, portanto todas as 2.709
operações do forward navegável carregam a região exata na qual sua fórmula
escalar pode ser endereçada. Domínios nativos ainda não comprovados continuam
com `accumulationDtype=runtime-defined`; a presença de shape nunca autoriza uma
redução aproximada. A regeneração, auditoria de 2.130 payloads e replays com o
checkpoint ausente estão em
[`docs/validation/gemma4-e4b-source-removed-calculation-domains-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-calculation-domains-2026-07-18.md).

O schema v4 acrescenta `learnedOperands`, um vínculo fail-closed entre cada
atribuição que consome storage e os papéis semânticos de seus tensores. Peso,
bias, limites `input/output min/max`, tabela de posição, kernels de convolução,
gamma de normalização e escalas aprendidas declaram o `decoderId` e expressões
de índice lógico antes do cálculo do índice row-major. O scalar view e a
navegação leem esse contrato serializado; não redescobrem papel por rank,
shape, posição no array ou sufixo do nome. Alterar/remover um papel, decoder ou
expressão de índice invalida o artefato inteiro.
A regeneração E4B, a partição dos 2.130 tensores e a substituição com o
checkpoint ausente estão em
[`docs/validation/gemma4-e4b-source-removed-learned-operands-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-learned-operands-2026-07-18.md).

O schema v5 acrescenta `scalarCalculations` para todas as definições do
forward composite, vision, audio e text. Cada entrada serializa a fórmula
indexada, inputs em ordem, coordenadas de saída, papéis aprendidos, casts,
domínio completo da redução e a agenda numérica. A navegação retorna esse
contrato diretamente do JSON; ela não depende de reconstruir a equação a
partir do nome da operação no leitor. As cinco classes de batched matmul cuja
árvore interna do Apple Accelerate ainda não é autoritativamente conhecida
continuam marcadas `fail-closed-runtime-reduction`: o schema torna o limite
endereçável em todas as definições compatíveis sem inventar soma ascendente.
A regeneração, cobertura source-removed e diferenciais de modalidade estão em
[`docs/validation/gemma4-e4b-source-removed-scalar-calculations-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-scalar-calculations-2026-07-18.md).

O schema v6 estende o mesmo limite para geração. `generation.forwardCalculation`
serializa a ordem instanciada completa do forward e uma transição KV por camada,
distinguindo produtores de consumidores que reutilizam o cache do produtor. As
12 classes de controle greedy possuem fórmulas em
`generation.scalarCalculations`; a navegação source-removed lê essas fórmulas
do JSON e rejeita qualquer divergência entre a ordem serializada e as operações
navegáveis. Prefill e decode incremental referenciam explicitamente os logits
nomeados e as transições BHSD de append/reuse, sem uma chamada
`declared_*_forward` ou `generic_decoder` reconstruída pelo leitor.
A regeneração real, a navegação source-removed, as transições KV e os
diferenciais estão em
[`docs/validation/gemma4-e4b-source-removed-generation-calculations-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-generation-calculations-2026-07-18.md).

O schema v7 torna essa expansão o grafo canônico do artefato, não uma
reconstrução do leitor. `calculationGraph.assignments` armazena cada chamada
instanciada em ordem de dependência, incluindo `definitionId`, `invocationId`,
entradas e saída já vinculadas, domínio de saída, fórmula escalar vinculada,
operandos aprendidos e arestas explícitas de produtor/consumidor. As chamadas
compostas de vision, audio e text não aparecem como `declared_subprogram`:
imagem, vídeo, áudio e o core textual apontam para suas atribuições concretas.
`generation.forwardCalculation.operationOrder` é derivada do mesmo grafo e a
navegação source-removed lê diretamente o JSON. A regeneração E4B real expôs
2.709 atribuições e 3.363 arestas sem alterar os diferenciais existentes; veja
[`docs/validation/gemma4-e4b-source-removed-instantiated-calculation-graph-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-instantiated-calculation-graph-2026-07-18.md).

O schema v8 acrescenta `numericLiterals`: uma tabela deduplicada e diretamente
endereçável de todo token numérico usado pelas fórmulas forward e greedy. Cada
entrada preserva o token textual, os bits IEEE-754 F64/F32/BF16, a regra BF16
round-to-nearest-ties-to-even e todas as definições consumidoras. Assim `0.5`,
`pi`, epsilons, escalas, softcaps, limites de índice e controles de geração não
dependem do parser decimal ou das constantes matemáticas do host. O leitor
reconstrói a tabela do próprio programa e rejeita qualquer bit ou uso alterado;
`inspect:gemma4-literal --numeric-literal <token>` expõe a entrada sem abrir o
checkpoint. A evidência do E4B real está em
[`docs/validation/gemma4-e4b-source-removed-numeric-literals-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-numeric-literals-2026-07-18.md).

Desde o schema v9, o próprio programa fica vinculado ao pacote imutável que estabeleceu suas
semânticas. `sourceIdentity` carrega `modelId`, revisão commit de 40 dígitos,
adaptador semântico, tamanho e SHA-256 de cada shard/metadata. Todos os JSONs
top-level do pacote — inclusive configuração, geração, processor e tokenizer —
são incorporados byte a byte em Base64 e validados contra o digest. Os shards
Safetensors não são duplicados: seus bytes aprendidos continuam em `constants`,
enquanto o hash do arquivo inteiro também compromete cabeçalho e layout do
container. Antes da remoção da fonte, o auditor recalcula os seis arquivos e
compara a identidade inteira. Identidade móvel, metadata alterada, arquivo
repetido ou decoder de metadata divergente falha fechado. A evidência do E4B
real está em
[`docs/validation/gemma4-e4b-embedded-source-identity-2026-07-18.md`](docs/validation/gemma4-e4b-embedded-source-identity-2026-07-18.md).

O schema v10 também incorpora `formulaLanguage`, o contrato normativo da
notação `indexed-ieee754-expression-v1`. Ele aponta para as seções que possuem
ordem, domínios, bindings aprendidos, decoders e bits literais; define índices,
casts F64/F32/BF16, operadores, reduções, intrínsecos e a execução greedy; e
exige falha fechada para qualquer redução `runtime-defined`. Assim, uma leitura
source-removed não precisa obter deste repositório a interpretação escondida
das fórmulas. A evidência real está em
[`docs/validation/gemma4-e4b-embedded-formula-language-2026-07-18.md`](docs/validation/gemma4-e4b-embedded-formula-language-2026-07-18.md).

O schema v11 remove a última expressão textual do caminho
índice-lógico-para-storage. Cada `learnedOperands.logicalIndices` é agora uma
AST `gemma4-learned-index-expression-v1`: coordenadas de saída, índices de
redução e escalares de input possuem fontes distintas; `add`, `multiply` e
`modulo` têm ordem e aritmética inteira exatas; índices de redução/input
carregam bounds inclusivo/exclusivo; e cada resultado é validado contra o shape
lógico antes do cálculo row-major. Os renderers de linear, embedding/PLE,
normalização, posição 2-D, convolução, depthwise, escala por dimensão, clipping
e tensor scalar usam um único avaliador dessa AST, em vez de reconstruir
endereços por tipo de operação. A evidência real source-removed está em
[`docs/validation/gemma4-e4b-learned-index-program-2026-07-18.md`](docs/validation/gemma4-e4b-learned-index-program-2026-07-18.md).

O schema v12 fecha o restante do caminho índice-lógico-para-bits. Cada decoder
denso incorpora um programa `address` com shape, strides row-major, domínio de
cada eixo, bytes por elemento e as equações exatas de element/byte offset. O
programa `decode` adjacente descreve a transformação bit a bit de F32, F16 ou
BF16 para os bits F32 resultantes, incluindo zero, subnormal, normal,
infinito/NaN de binary16. O avaliador de substituição usa esses dois programas
serializados; não volta a inferir stride, largura ou conversão a partir do
dtype. Slices e a vista end-to-end carregam o decoder completo de cada
constante, e adulterar stride, domínio ou algoritmo IEEE falha fechado. A
evidência real está em
[`docs/validation/gemma4-e4b-dense-decoder-program-2026-07-18.md`](docs/validation/gemma4-e4b-dense-decoder-program-2026-07-18.md).

O schema v13 torna esses programas diretamente executáveis, em vez de
preservar suas equações como strings verificadas pelo leitor. `address` usa a
AST `exact-safe-integer-expression-v1` para reduzir eixos, multiplicar
strides e formar o byte range com overflow fail-closed. `decode` usa a AST
`u32-bit-expression-v1`: leitura little-endian, bindings ordenados, bitwise,
shifts, `count-leading-zeros-u32` e seleção condicional produzem os bits F32
que são então bitcast, sem uma conversão de dtype escolhida pelo host. O
contrato compartilhado `denseDecoderLanguage` incorpora a semântica e a ordem
de avaliação desses opcodes. A vista escalar, os slices e a composição
end-to-end carregam o contrato e somente os decoders usados. A evidência real
foi revisada independentemente e está preservada, junto da exportação que a
substitui, em
[`docs/validation/gemma4-e4b-exact-product-scalar-audit-2026-07-18.md`](docs/validation/gemma4-e4b-exact-product-scalar-audit-2026-07-18.md).

O schema v14 corrige a fronteira produto-redução para toda classe linear Gemma
4 compatível. Fórmulas com agendas FMA/dot agora declaram
`exact_product(a*b)`: o produto não recebe um `F32` prematuro antes da soma
fundida. As vistas escalares text, vision e audio compartilham a mesma
transcrição por `ReductionSchedule`, substituem cada peso por seu número
decodificado em `product[i]` e abrem registradores ARM, árvore 0+4/1+5/2+6/3+7,
fold horizontal, vector tail e scalar tail sem `weight[o,i]` nem intrínsecos
opacos. Uma validação fail-closed recusa qualquer vista completa com referência
aprendida simbólica, termo ausente ou literal decodificado sem uso. A evidência
real source-removed está no mesmo relatório acima.

O schema v15 fecha as reduções aprendidas não lineares cujo contrato nativo já
era conhecido. Os 637 RMSNorms instanciados agora abrem unidades, quatro níveis
de cascade, merges/clears, registradores e lanes; as duas LayerNorms de canais
abrem updates Welford low/high, merge, fold de momentos e segundo passe
`x*inv_std+bias` antes de gamma; e os dois Conv2d mais doze depthwise obedecem
seus quatro acumuladores intercalados em vez de exibirem uma soma ordenada.
`formulaLanguage` schema 3 incorpora os programas normativos de cascade e
Welford. A varredura source-removed de todas as 653 vistas decodificou 1.801
literais aprendidos sem helpers opacos ou referências simbólicas. A evidência
real está em
[`docs/validation/gemma4-e4b-explicit-normalization-reductions-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-normalization-reductions-2026-07-18.md).

O schema v16 incorpora `authoritativeExecution` no próprio programa. O
artefato fixa o forward composite autoritativo em
`Gemma4ForConditionalGeneration` CPU eager dentro de `torch.inference_mode`,
distingue as capturas diagnósticas diretas de áudio e visão e registra, sem
aproximar, as cinco classes de BMM despachadas para Apple Accelerate SGEMM cuja
árvore escalar não é publicada. Capturadores, comparadores e o leitor recusam
traces ou headers `no_grad`/runtime divergentes. A regeneração real, auditoria
source-removed e evidência do desvio dependente do contexto estão em
[`docs/validation/gemma4-e4b-authoritative-execution-contract-2026-07-18.md`](docs/validation/gemma4-e4b-authoritative-execution-contract-2026-07-18.md).

O leitor source-removed torna essa fronteira navegável sem enfraquecê-la.
`--list-runtime-reductions` enumera as 100 instâncias por classe autoritativa;
`--runtime-reduction-audit`, junto de `--operation` e
`--output-coordinate`, abre os endereços concretos de cada par de operandos,
predicados de padding e o domínio incorporado. O resultado continua marcado
`fail-closed-runtime-reduction`: `REAL_PRODUCT` é somente o termo matemático
auditável, enquanto arredondamento do produto, ordem de acumulação e saída
permanecem indisponíveis até uma agenda Apple Accelerate geral ser provada.
Assim a vista estrita continua recusando a operação, mas um leitor pode seguir
todos os termos conhecidos em vez de receber apenas uma mensagem de erro. A
prova nas cinco classes reais está em
[`docs/validation/gemma4-e4b-runtime-reduction-product-navigation-2026-07-19.md`](docs/validation/gemma4-e4b-runtime-reduction-product-navigation-2026-07-19.md).

O schema v17 fixa também a implementação de atenção como `eager` e reproduz o
contrato de máscara de áudio observado no Transformers 5.5.0 para toda a classe
compatível. A fonte cria uma máscara **aditiva**, preenche o bloqueio 5-D com
zero e aplica `masked_fill(mask.logical_not(), -1e9)`: zeros permitidos e de
padding são substituídos, enquanto entradas rejeitadas não zero preservam seu
score. Capturas SDPA, `no_grad` ou sem esse contexto são recusadas. Com essa
semântica explícita, a execução composite source-removed do input de áudio
registrado obteve áudio terminal, logits, token greedy e todos os caches KV
exatos; as 100 reduções BMM nativas continuam fail-closed e impedem declarar o
checkpoint. Causa, hashes e limites estão em
[`docs/validation/gemma4-e4b-eager-audio-mask-contract-2026-07-18.md`](docs/validation/gemma4-e4b-eager-audio-mask-contract-2026-07-18.md).

O schema v18 remove a dependência externa escondida pelos nomes
`SLEEF_EXP_F32`, `SLEEF_SIN_F32`, `SLEEF_COS_F32` e `SLEEF_TANH_F32` em todas
as classes text/vision/audio. `transcendentalPrograms` incorpora a autoridade
PyTorch/SLEEF, cada constante pelos bits binary32, as regras F32/FMA e de pares,
os subprogramas de redução/polinômio, branches de NaN/infinito/zero assinado e a
tabela `rempif` de 416 F32 com SHA-256. A linguagem de fórmulas aponta para esse
programa finito; leitor, vistas escalares, slices e cálculo end-to-end recusam
qualquer alteração ou intrínseco não resolvido, sem carregar uma biblioteca ou
checkout SLEEF. A evidência real source-removed está em
[`docs/validation/gemma4-e4b-embedded-transcendental-programs-2026-07-18.md`](docs/validation/gemma4-e4b-embedded-transcendental-programs-2026-07-18.md).

O schema v19 fecha a classe restante de intrínsecos unários usados pelo Gemma
4: `SLEEF_LOG1P_F32`, `ARM_SQRT_F32` e
`PYTORCH_POW_NEGATIVE_HALF_F32` passam a ser programas finitos incorporados,
ao lado dos quatro programas do schema v18. Isso cobre softplus aprendido do
áudio, LayerNorm/Welford e RMSNorm, e substitui também os `exp`, `tanh`,
`sin` e `cos` crus que ainda apareciam em algumas vistas escalares. A validação
agora recusa qualquer chamada host unária nas fórmulas ou vistas. A exportação
real source-removed, comparação bit a bit com PyTorch e replay composite estão
em
[`docs/validation/gemma4-e4b-embedded-unary-runtime-math-2026-07-18.md`](docs/validation/gemma4-e4b-embedded-unary-runtime-math-2026-07-18.md).

O schema v20 separa cada atenção/softmax em estágios de redução explícitos, em
vez de atribuir uma única ordem ambígua ao cálculo inteiro. As 42 atenções de
texto declaram separadamente dot de score, máximo, soma exponencial e dot de
contexto; visão e áudio declaram máximo, predicado de validade e soma sobre os
domínios completos. A linguagem de fórmulas incorpora programas executáveis
para redução F32 ordenada e para o fold vetorial PyTorch de quatro lanes, e as
vistas escalares apontam para cada estágio sem `max_k`, `score-max-*` ou outro
helper opaco. A exportação real e os replays source-removed estão em
[`docs/validation/gemma4-e4b-explicit-softmax-reductions-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-softmax-reductions-2026-07-18.md).

O schema v21 fecha os helpers de indexação e RoPE ainda escondidos nas
fórmulas. Um programa finito incorporado calcula o identificador de cada grupo
vision contíguo; substituição de tokens multimodais e posição 2D escrevem seus
predicados e índices diretamente. Todas as 66 atribuições RoPE de texto agora
declaram par, feature pareada, limite de pares ativos, expoente theta,
escalonamento proporcional, casts BF16/F32 e a soma/subtração `rotate_half`.
A validação percorre todas as fórmulas forward e de geração e recusa qualquer
chamada cujo programa ou semântica não esteja registrado. A exportação real e
os replays source-removed estão em
[`docs/validation/gemma4-e4b-explicit-indexing-rope-programs-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-indexing-rope-programs-2026-07-18.md).

O schema v22 fecha a indexação estável que liga features multimodais ao fluxo
textual. Cada scatter de imagem, vídeo e áudio carrega o token modal
autoritativo e calcula a linha exata de features por prefix rank batch-major;
os dois `strip-padding` calculam a coordenada fonte do rank válido em vez de
usar uma linha implícita. `STABLE_TRUE_COUNT`, `STABLE_TRUE_PREFIX_RANK` e
`STABLE_TRUE_COORDINATE_AT_RANK` são programas executáveis incorporados, e o
leitor falha se o programa, token ou cardinalidade divergir. As fórmulas e
vistas escalares não contêm mais `next_feature_row`, `placeholder_at` ou
`stable_batch_major_true_mask_row`. A prova no pacote real está em
[`docs/validation/gemma4-e4b-explicit-stable-selection-programs-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-stable-selection-programs-2026-07-18.md).

O schema v23 elimina a prosa de coordenadas das classes restantes que já têm
semântica autoritativa. Pooling vision calcula cada célula com os programas
executáveis `VISION_POOL_SLOT` e `VISION_POOL_CELL_HAS_PATCH`; relative shift
audio calcula a coordenada exata de `pad/view/slice/view` com
`AUDIO_RELATIVE_SHIFT_SOURCE`; score, máscara e value chunked declaram
`query_index`/`key_index` e seus bounds; e todos os 90 reshapes de heads usam
alias row-major explícito. Os executores recusam programas alterados, e as
vistas source-removed expõem o loop FMA, a seleção booleana e a coordenada
fonte concreta. A prova no pacote real está em
[`docs/validation/gemma4-e4b-explicit-coordinate-programs-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-coordinate-programs-2026-07-18.md).

O schema v24 transforma as 42 transições KV textuais em programas escalares
executáveis. Cada uma das 24 camadas produtoras declara cópia BHSD de prefill
e append incremental por coordenada, com shape, dtype, eixo de sequência e
ordem lexicográfica; cada um dos 18 consumidores compartilhados lê o cache do
produtor anterior e declara `cache_entry_present[layer]=false`, sem criar uma
cópia. Os programas fazem parte da linguagem incorporada e da tabela de bits
numéricos, e qualquer fórmula, dtype, ownership ou layer alterado falha na
validação canônica. A prova no pacote real está em
[`docs/validation/gemma4-e4b-explicit-kv-cache-programs-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-kv-cache-programs-2026-07-18.md).

O schema v25 fecha a ligação dos operandos nas fórmulas canônicas. As classes
linear, RMSNorm, ativação, elementwise, Conv2d/depthwise, LayerNorm, GLU e BMM
vision agora nomeiam cada `orderedInput` e sua coordenada completa; nenhum
leitor precisa inferir `input`, `x`, `q`, `k`, `value`, `padded_input` ou
`...`. As 32 definições RoPE 2-D vision declaram eixo, feature local, par,
feature pareada, ângulo e os branches concretos de soma/subtração, sem `+/-`
nem "follows rotate_half". `formulaLanguage` schema 11 torna essa clausura de
operandos parte do contrato e a construção falha para toda a classe quando um
alias livre reaparece. Isso não inventa a árvore Apple Accelerate: as mesmas
100 instâncias BMM continuam `fail-closed-runtime-reduction`. A prova no pacote
real, incluindo 1.574 definições operand-closed e views source-removed, está em
[`docs/validation/gemma4-e4b-explicit-operand-bindings-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-operand-bindings-2026-07-18.md).

O schema v26 substitui as descrições em prosa dos shapes dinâmicos por
`calculationDomains.dimensionPrograms`. A linguagem
`gemma4-safe-integer-dimension-expression-v1` incorpora um AST fechado para
ler eixos de tensores, somar o comprimento KV opcional, contar máscaras BOOL
em ordem row-major e executar soma, produto, ceil-division e divisão exata com
checagem de inteiro seguro. Assim `K`, os dois estágios Conv2d de áudio,
`ABLOCKS`, pooling vision, flatten de vídeo e cardinalidades após
`strip-padding` são calculáveis sem interpretar frases como “past key length
plus S” ou “number of true rows”. Shapes compostos aceitam somente literais,
dimensões registradas e multiplicação exata; ciclos, nomes ausentes, overflow,
divisor zero e pooling não divisível falham fechado. Slices, vistas escalares
e o cálculo end-to-end carregam a mesma linguagem e os mesmos programas. A
prova no pacote real está em
[`docs/validation/gemma4-e4b-executable-dimension-programs-2026-07-18.md`](docs/validation/gemma4-e4b-executable-dimension-programs-2026-07-18.md).

O schema v27 fecha a mesma dependência interpretativa dentro dos limites de
redução. Cada linear, normalização, convolução, pooling e atenção de
texto/visão/áudio passa a carregar `reduction.domains` ou
`reductionStages[*].domains`: um AST finito liga cada índice a um inteiro
positivo incorporado ou a um eixo concreto de um `orderedInput`. A instanciação
do grafo reescreve também esses nomes para o call site real, por exemplo
`composite_audio_features/audio_layer_0_attention_ac`, sem dispatch por camada.
O leitor source-removed valida 1.722 domínios para 1.462 reduções e 256 estágios,
e o diferencial de áudio avaliou 1.138 domínios ativos, incluindo os bounds das
36 BMM de áudio. Isso torna `in_features`, `width`, `patches`, `context` e
`head_dim` meros rótulos humanos: nenhum deles pode fornecer implicitamente o
extent. A ordem escalar Apple Accelerate das 100 BMM continua corretamente
fail-closed. Contrato, hashes e replay estão em
[`docs/validation/gemma4-e4b-executable-reduction-domains-2026-07-18.md`](docs/validation/gemma4-e4b-executable-reduction-domains-2026-07-18.md).

O schema v28 torna executável também a ordem interna de cada fórmula forward.
`formula` continua sendo a vista compacta output-first para auditoria humana,
mas cada definição e cada call site passam a carregar `scalarAssignments`: um
programa em que coordenadas locais, temporários de redução e `require` aparecem
antes da atribuição final. A construção cobre todas as classes de operação e
falha se um temporário for lido antes de ser declarado; o binding do grafo
instanciado reescreve a mesma sequência sem perder a ordem. No pacote real, as
2.709 atribuições forward contêm 4.146 statements, dos quais 1.437 são passos
locais/precondições em 288 programas multistep. Isso elimina a necessidade de
um leitor reordenar por conta própria `score`, `maximum`, `total`, coordenadas
RoPE/chunked, Welford e pooling. As 100 BMM Apple Accelerate continuam
explicitamente fail-closed quanto à agenda escalar. Contrato, hashes e replay
source-removed estão em
[`docs/validation/gemma4-e4b-dependency-ordered-scalar-programs-2026-07-18.md`](docs/validation/gemma4-e4b-dependency-ordered-scalar-programs-2026-07-18.md).

O schema v29 remove o último loop greedy implícito do replay literal. Além da
vista humana em `generation.scalarCalculations`, o artefato incorpora agora
`generation.controlProgram`: um programa estruturado que fixa restrições do
prompt, bindings do prefill, posição inicial, domínio e desempate do argmax,
append do token, avanço inteiro de posição, inputs deliberadamente omitidos no
decode, forward incremental, snapshot de cache, momento exato do EOS e seleção
do estado terminal. Os executores síncrono, paged-text e composite interpretam
esse mesmo objeto; alterar um campo, remover um input omitido ou trocar a
cardinalidade do vocabulário falha antes do replay. O programa preserva a lista
instanciada de 224 operações forward e as transições KV produtor/reuso já
incorporadas, sem delegar a geração a `generate`, Transformers ou outro loop de
host. A prova no pacote real está em
[`docs/validation/gemma4-e4b-structured-generation-control-2026-07-18.md`](docs/validation/gemma4-e4b-structured-generation-control-2026-07-18.md).

O schema v30 remove também a seleção modal implícita do forward. O novo
`forwardControl` é um programa estruturado que ordena imagem, vídeo e áudio,
exige cada par de inputs como `all-present-or-all-absent`, nomeia as operações
ativas e a invocação expandida de cada torre, e declara aliases identidade sem
cast quando um ramo está ausente. O mesmo objeto seleciona máscaras visual
full/sliding versus máscara aditiva do chamador ou causal/cache textual, fixa
os conflitos de `mm_token_type_ids`, declara o default escalar de
`position_ids` e distingue cache vazio de cache pós-RoPE fornecido. Replay
síncrono e paginado interpretam esse programa antes de qualquer torre ou
camada textual; adulterar um input group, alias, ordem, modo de máscara ou
default falha fechado. A evidência do pacote real está em
[`docs/validation/gemma4-e4b-structured-forward-control-2026-07-19.md`](docs/validation/gemma4-e4b-structured-forward-control-2026-07-19.md).

O schema v31 remove a validação implícita dos inputs. `inputContract` declara
as 13 entradas de forward/geração com representação, rank e eixos; fixa o
domínio dos tokens e posições, os shapes de `mm_token_type_ids` e máscara
aditiva, e enumera somente os 24 produtores que podem possuir cache KV BHSD.
Para imagem e vídeo, o mesmo programa valida largura patchificada `768`,
coordenadas/padding, pooling `3x3`, ordem de flatten e igualdade entre células
válidas e placeholders. Para áudio, valida máscara, dois strides 2, 32 canais,
largura 1.024 da projeção e a cardinalidade de soft tokens após subsampling.
Replay síncrono e paginado executam o contrato antes do `forwardControl`; ranks,
buffers, shapes relacionados, cache extra/ausente e cardinalidade divergente
falham antes de qualquer peso ser lido. A evidência do pacote real está em
[`docs/validation/gemma4-e4b-literal-input-contract-2026-07-19.md`](docs/validation/gemma4-e4b-literal-input-contract-2026-07-19.md).

O schema v32 fecha o mesmo limite na saída. `outputContract` declara os três
tensores públicos do forward (`hidden_states_0`, `ple_inputs` e logits), seus
produtores, eixos, shapes e layout; fixa a substituição modal observável em
`llmInputIds`; e enumera somente os produtores KV que podem aparecer no estado
BHSD pós-RoPE. Na geração, o contrato liga `executed_steps` a tokens, logits de
seleção e snapshots de cache, valida o crescimento exato da sequência, a parada
antecipada somente por EOS e a identidade entre cache terminal e o último
forward incremental (ou prefill quando zero passos são pedidos). Replay
síncrono e paginado executam esse objeto depois de cada forward e ao fechar a
geração; adulterar shape, produtor, propriedade de cache ou relação terminal
falha fechado. A evidência no pacote real está em
[`docs/validation/gemma4-e4b-literal-output-contract-2026-07-19.md`](docs/validation/gemma4-e4b-literal-output-contract-2026-07-19.md).

O schema v33 torna a cadeia de estados greedy verificável sem confiar na
identidade de objetos do executor. Cada forward incremental publica
`step_forward_logits[step]` junto de `step_past_key_values[step]`; o próximo
`selection_logits` deve ser bitwise idêntico ao snapshot anterior, cada token
deve ser o argmax crescente declarado, e logits/cache terminais devem pertencer
ao mesmo último estado. O programa de geração, o controle estruturado, a
navegação de 2 passos (agora 22 atribuições) e ambos os replays executam o mesmo
contrato. A evidência do pacote real está em
[`docs/validation/gemma4-e4b-greedy-state-chain-2026-07-19.md`](docs/validation/gemma4-e4b-greedy-state-chain-2026-07-19.md).

O schema v34 torna cada aresta do grafo navegável também no nível de
coordenadas. Cada `calculationGraph.assignments[*].predecessors[*]` incorpora,
em ordem da primeira leitura, os acessos `tensor-element`, `tensor-shape` ou
`whole-value` extraídos do programa escalar da própria atribuição. Uma vista
concreta devolve o mesmo vínculo em `predecessorCoordinates`, com o ID do
produtor, o template simbólico, os endereços materializados e cobertura
`complete` ou `windowed`; a auditoria BMM fail-closed usa exatamente o mesmo
contrato. A construção não despacha por layer ou ID. Ela também removeu uma
atribuição de placeholder sem consumidores e duas arestas que não participavam
do cálculo. No pacote real restam 2.708 atribuições, 3.491 predecessores, 3.488
leituras escalares endereçadas e três dependências somente de shape/controle.
A prova source-removed, hashes e a suite diferencial das três modalidades
estão em
[`docs/validation/gemma4-e4b-predecessor-coordinate-navigation-2026-07-19.md`](docs/validation/gemma4-e4b-predecessor-coordinate-navigation-2026-07-19.md).

O schema v35 fecha a navegação no sentido produtor → consumidor para a classe
forward inteira. Cada atribuição agora possui `outputCoordinate.write`, a única
coordenada no lado esquerdo realmente escrita por seu programa escalar, mais
eventuais invariantes em `shapeAssertions`. `consumerCoordinates` agrupa, por
ID de consumidor, os endereços exatos em que essa saída é lida. O mesmo parser
é usado nos dois sentidos; ele também reduz `row_major_alias(x)[...]` ao acesso
real `x[...]`, sem tabela por operação, shape, dtype ou layer. Vistas escalares
concretas e auditorias BMM fail-closed expõem `renderedOutputCoordinate`, então
o leitor pode sair de um predecessor, localizar a escrita intermediária e
seguir os consumidores seguintes sem interpretar texto livre.

No E4B real, as 2.708 atribuições possuem 2.708 escritas estruturadas; as 3.361
arestas produtor-consumidor possuem 4.200 acessos downstream (4.142 de
elemento, 55 de shape e três de valor completo). A redução de aliases também
elevou os acessos predecessores de elemento de 4.101 para 4.287 e reduziu
leituras `whole-value` de 189 para três. A prova source-removed, integridade e
suite diferencial estão em
[`docs/validation/gemma4-e4b-bidirectional-coordinate-navigation-2026-07-19.md`](docs/validation/gemma4-e4b-bidirectional-coordinate-navigation-2026-07-19.md).

O leitor também pode partir de qualquer saída instanciada e calcular o fecho
transitivo exato de seus produtores. `--calculation-slice <operation-id>`
retorna somente as atribuições necessárias ao alvo, ainda em ordem de
dependência, mais os inputs do chamador, constantes aprendidas com decoder e
consumidores, programas executáveis de dimensão, literais numéricos com bits e todas as reduções que permanecem
fail-closed. Isso permite seguir uma saída até suas entradas sem confundir a
ordem linear global com dependência real ou omitir um ramo necessário:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --calculation-slice final_logit_softcap \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-final-logits-calculation-slice.json
```

Para seguir o produto inteiro em uma única vista finita,
`--end-to-end-calculation --generation-max-new-tokens <n>` liga os inputs
declarados ao fecho de produtores dos logits, instancia todas as atribuições
greedy para o limite escolhido e inclui as transições KV. Essa vista também
prova a cobertura do storage: cada constante incorporada precisa estar no
cálculo alcançável ou na lista explícita de projeções K/V e K-norm locais que
não são executadas porque a camada reutiliza o cache de um produtor. Uma constante
órfã, um decoder ausente ou uma declaração `runtime-unreachable` contraditória
faz a inspeção falhar fechado:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-end-to-end-calculation.json
```

No E4B real, essa vista contém as 2.709 atribuições forward uma única vez,
20 atribuições de controle para dois passos, 42 transições KV, todos os 70
literais numéricos e a partição completa das 2.130 constantes incorporadas.
Veja
[`docs/validation/gemma4-e4b-source-removed-end-to-end-calculation-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-end-to-end-calculation-2026-07-18.md).

O schema v36 remove a interpretação textual que ainda restava na navegação de
coordenadas do schema v35. Cada acesso `tensor-element` incorpora agora um
`coordinatePrograms` por eixo, e cada acesso `tensor-shape` incorpora um
`axisProgram`. A linguagem fechada `gemma4-coordinate-expression-v1` cobre
inteiros assinados seguros, símbolos e campos de `STRUCT`, arrays inteiros
indexados, soma/subtração/produto, módulo não negativo, divisão inteira com
`floor`, ranges inclusivos e o rank estável de placeholders multimodais.
Bindings ausentes, overflow, divisor zero, tensor ragged, range invertido ou
helper desconhecido falham antes da navegação; o leitor não precisa analisar a
string humana `expression` para descobrir um endereço.

Como `/` também separa os segmentos dos nomes de tensores compostos, cada
operando `tensor/path.shape[axis]` usado em uma divisão `floor` é delimitado
por parênteses. Isso mantém a divisão e o caminho lexicalmente inequívocos e
faz a construção falhar fechado se um gerador emitir a forma ambígua.

No E4B real, o grafo contém 36.495 programas de coordenada para as 2.708
atribuições, incluindo sete ranges e seis ranks estáveis. A fonte foi removida
fisicamente durante a verificação integral dos 2.130 payloads, vistas escalar e
BMM, fecho end-to-end e suite diferencial das três modalidades. A prova e os
hashes estão em
[`docs/validation/gemma4-e4b-executable-coordinate-programs-2026-07-19.md`](docs/validation/gemma4-e4b-executable-coordinate-programs-2026-07-19.md).

O schema v37 fecha os bindings desses programas de coordenada. O schema v36
ainda permitia que nomes como `frames`, `patches`, `pool_cells`, `head_dim`,
`channels`, `per_layer_width` e `feature` fossem interpretados como escalares
fornecidos pelo host, mesmo quando não pertenciam aos eixos ou domínios da
atribuição. A linguagem `gemma4-coordinate-expression-v2` acrescenta o nó
explícito `tensor-axis`, que resolve somente `shape[axis]` de um input,
predecessor ou output declarado. O calculation graph schema 5 valida cada
programa contra os eixos de saída, índices de redução, locals anteriores,
campos `STRUCT` e tensores do call site; alias livre ou tensor não declarado
falha durante a construção e durante a abertura do artefato.

No E4B real, as 2.708 atribuições passaram esse fecho sem bindings do chamador.
O artefato de 21.391.466.484 bytes tem SHA-256
`88b82835fa012291b4797740dc760e3f059c85b7af816e9d758c2aa3ded311eb`
e contém 36.505 programas de coordenada, incluindo 14 leituras explícitas de
eixo de tensor. Com a fonte fisicamente ausente, todos os 2.130 payloads foram
verificados, uma vista de `layer_0_q_proj[0,0,0]` substituiu pesos BF16 por
literais numéricos, o fecho forward/greedy de um passo foi aberto e imagem,
vídeo e áudio repetiram prefill e geração com tolerância zero. As 100 BMM
Apple Accelerate continuam explicitamente `fail-closed-runtime-reduction`.
Comandos, contagens e hashes estão em
[`docs/validation/gemma4-e4b-closed-coordinate-bindings-2026-07-19.md`](docs/validation/gemma4-e4b-closed-coordinate-bindings-2026-07-19.md).

O schema v38 fecha a navegação dentro de cada operação. Embora o schema v28 já
ordenasse `scalarAssignments`, os locals como `score[key]`, `maximum`,
`angle`, `paired_feature` e `source_coordinate` ainda não declaravam qual
statement os produzia nem quais statements os consumiam. Cada cálculo agora
incorpora `statementDataflow`: writes locais/output, reads ligados ao
`producerStatementOrdinal`, arestas reversas em `consumerStatementOrdinals` e
ASTs de coordenada para todos os locals indexados. O construtor aplica o mesmo
algoritmo a todas as classes e call sites; não existe tabela por layer ou
assignment ID.

No E4B real são 4.144 entries de dataflow para 4.144 statements, 1.431 writes
locais, 2.708 writes de output, 1.958 reads/arestas de produtor, 1.931 arestas
reversas deduplicadas e 9.340 programas de coordenada local. O artefato de
21.394.548.803 bytes tem SHA-256
`63e3b8da96d56593c3935176a555614e12fde534833fae391b0e1d13aca5f628`.
Com o checkpoint fisicamente ausente, o leitor verificou os 2.130 payloads,
abriu o fecho forward/greedy, substituiu pesos BF16 reais e repetiu imagem,
vídeo e áudio a tolerância zero. As 100 BMM Apple Accelerate permanecem
`fail-closed-runtime-reduction`; dataflow explícito não inventa a agenda
nativa. Comandos, contagens e hashes estão em
[`docs/validation/gemma4-e4b-scalar-intermediate-dataflow-2026-07-19.md`](docs/validation/gemma4-e4b-scalar-intermediate-dataflow-2026-07-19.md).

O schema v39 remove a última necessidade de reparsar essas fórmulas forward.
Cada ordinal agora incorpora `statementPrograms`, uma árvore sintática fechada
com destinos, literals, identifiers, arrays, casts/calls, índices, membros,
operadores unários/binários, condicionais, ranges, argumentos nomeados,
domínios filtrados, loops ascendentes e invocações composite. A string
`scalarAssignments[ordinal]` continua como rendering humano, mas a árvore é a
autoridade executável e precisa corresponder exatamente a ela. O binding de
call site é estrutural, de modo que caminhos como
`composite_image_features/vision_layer_0_q` permanecem um identificador e não
são reinterpretados como divisão.

No E4B real, os 4.144 statements produziram 4.144 programas e 118.780 nós de
expressão, incluindo 1.718 calls de redução/FMA/dot, dois loops ascendentes e
64 domínios filtrados. A construção de todas as classes também encontrou e
corrigiu os parênteses desequilibrados na fórmula de posição relativa de áudio.
O artefato de 21.406.708.755 bytes tem SHA-256
`942a33a61a7380bf8e0f12a916fc47dc183c7b678706e78e73e3b066b19a35d5`.
Payloads, vista escalar, fecho forward/greedy e a suite diferencial das três
modalidades passaram com a fonte fisicamente ausente. As 100 BMM Apple
Accelerate continuam `fail-closed-runtime-reduction`. A prova completa está em
[`docs/validation/gemma4-e4b-executable-scalar-statement-programs-2026-07-19.md`](docs/validation/gemma4-e4b-executable-scalar-statement-programs-2026-07-19.md).

O schema v40 fecha também o ambiente léxico dessas árvores. Cada cálculo
incorpora `statementEnvironment` com coordenadas de saída, `orderedInputs`
posicionais, produtores locais, índices e aliases de extent ligados aos
domínios executáveis, papéis aprendidos aceitos por `decode`, intrinsics e
contratos de membro. Identificador livre, helper desconhecido, role aprendida
ausente, stage de redução inválido, membro host ou local estruturado sem
contrato falha durante a construção e novamente ao abrir o artefato. O binding
é reconstruído em cada call site, portanto os nomes instanciados de imagem,
vídeo e áudio não dependem do source TypeScript.

No E4B real são 2.708 ambientes: 3.491 bindings de input, 1.431 de locals,
1.722 de redução, 10.727 de intrinsic e 211 acessos de membro. A auditoria
também eliminou os aliases livres `feature` na escala por dimensão de áudio e
`head_dim` no RoPE textual, substituindo-os pelo eixo final e `rotaryDim`
declarados. O artefato de 21.409.225.548 bytes tem SHA-256
`f0e8a36902b1bbb65c1f85ca32347b5cdfc05ae694a6a74af12cd3b8179543b7`.
Com a fonte fisicamente ausente, os 2.130 payloads, a vista escalar e as três
modalidades passaram; as 100 BMM Apple Accelerate permanecem
`fail-closed-runtime-reduction`. Comandos, hashes e contagens estão em
[`docs/validation/gemma4-e4b-closed-scalar-statement-environments-2026-07-19.md`](docs/validation/gemma4-e4b-closed-scalar-statement-environments-2026-07-19.md).

O schema v41 torna a integridade parte obrigatória do programa, não uma opção
do leitor. Cada artefato incorpora um compromisso SHA-256 por payload e um
`integrityManifest` sobre 23 seções canônicas: identidade imutável, política,
inputs, metadados de constantes, decoders, programa, atribuições, fórmulas,
ASTs, navegação, controles forward/greedy, outputs e a tabela completa de
payloads. Alterar um campo ainda estruturalmente válido, remover a tabela ou
trocar um digest falha ao abrir; replay em memória também rehasha os bytes
aprendidos antes de executar.

O E4B real regenerado tem 21.409.228.758 bytes, SHA-256
`271147d1db4400e272911d603a31e02796e9793766468bbf6041602473820558` e
raiz estrutural
`eb1899dfb59bfe44eb9bc3f6f7e489b0436c2dbf95f76f2e96eb05e5d6f4b1b1`.
Com a fonte fisicamente ausente, todos os 2.130 payloads / 15.992.314.836
bytes, a substituição escalar e a suite diferencial de imagem, vídeo e áudio
passaram. As 100 BMM Apple Accelerate continuam explicitamente
`fail-closed-runtime-reduction`. Evidência completa em
[`docs/validation/gemma4-e4b-mandatory-artifact-integrity-2026-07-19.md`](docs/validation/gemma4-e4b-mandatory-artifact-integrity-2026-07-19.md).

O schema v42 incorpora um `fidelityGate` derivado do grafo canônico e coberto
pelo manifesto de integridade. Ele enumera cada redução nativa cuja agenda
escalar ainda é desconhecida, com classe de operação, ordinal, IDs de definição
e invocação, output e JSON Pointers para a redução e sua coordenada. Qualquer
divergência entre o gate, o grafo e o contrato autoritativo falha ao construir
ou abrir o artefato. Enquanto a lista não estiver vazia, o próprio programa
declara `exactReplayClaim=forbidden`; uma lista vazia seria apenas elegível para
certificação independente, nunca autocertificada pela sessão que a produziu.

No E4B real, o gate compromete exatamente as 100 BMM Apple Accelerate nas cinco
classes autoritativas. O artefato tem 21.409.281.213 bytes, SHA-256
`b5b826dae8a89001c437341c82a4bf4c828d79a065bdeb66be40667ffb90a72f`
e raiz estrutural
`ffe0578f74c82226d9eb6fb71f2035580fe8dcf52cf20cf84b685118134f5910`.
Payloads, vista escalar e a suite diferencial das três modalidades foram
executados com a fonte fisicamente ausente. Evidência completa em
[`docs/validation/gemma4-e4b-integrity-bound-fidelity-gate-2026-07-19.md`](docs/validation/gemma4-e4b-integrity-bound-fidelity-gate-2026-07-19.md).

O schema v43 remove a última reconstrução por classe da auditoria de produtos
BMM. A vista source-removed agora lê `statementPrograms`, `orderedInputs`,
`outputCoordinates` e o domínio de redução diretamente de cada atribuição do
`calculationGraph`; executa o prelude escalar serializado, localiza o único
`REDUCE` e seus predicados e materializa os dois endereços de operandos sem
consultar `headDim`, chunk, contexto ou outro metadado da torre no leitor. Os
extents arquiteturalmente fixos de score visual, score de áudio e value de
áudio são constantes do domínio serializado. Somente o número de patches do
value BMM visual permanece corretamente dependente do shape e exige uma janela
explícita sem shapes de runtime. A fronteira Apple Accelerate continua
fail-closed; esta mudança torna auditável a entrada completa dessa fronteira,
sem inventar sua árvore de acumulação.

O E4B v43 tem 21.409.277.259 bytes, SHA-256
`8dbfd9e22126b5286d94f6a2906cfe31b392e2056ef80c7559b97c51a426461e`
e raiz estrutural
`20ec2149a01250b350c174336627c33ee8c28924c5a2b1af7469a524b2c34c08`.
Os 2.130 payloads e 15.992.314.836 bytes aprendidos foram revalidados com a
fonte presente e ausente. A suite source-removed repetiu 3/3 prefill e 3/3
gerações a tolerância zero, preservando as 100 reduções nativas no gate.

Evidência da regeneração E4B real, auditoria source-removed e regressões está em
[`docs/validation/gemma4-e4b-serialized-runtime-reduction-audit-2026-07-20.md`](docs/validation/gemma4-e4b-serialized-runtime-reduction-audit-2026-07-20.md).

Os traces autoritativos vision/audio schema v2 fecham a outra metade dessa
fronteira de auditoria: cada redução BMM nativa precisa trazer sua saída e os
dois operandos exatos consumidos pelo kernel. Em vision isso inclui Q/K
**depois** do RoPE, capturados dentro de `eager_attention_forward`; Q/K somente
normalizados não satisfazem o contrato. O trace incorpora um `captureId`, os
SHA-256 de `config.json`/`model.safetensors` e
`runtimeReductionCoverage`, derivado do programa e revalidado contra o
artefato source-removed. Operando anterior, produtor ausente, shape adulterado,
output divergente ou identidade de checkpoint diferente falha antes da
comparação numérica. O E4B real registrou 32/32 fronteiras de vídeo e 36/36 de
áudio; as agendas Apple Accelerate continuam corretamente no gate como
`runtime-defined`. Evidência em
[`docs/validation/gemma4-e4b-native-bmm-operand-traces-2026-07-20.md`](docs/validation/gemma4-e4b-native-bmm-operand-traces-2026-07-20.md).

O schema v45 transforma o replay executável dessas 100 fronteiras em evidência
fail-closed. O contrato incorporado fixa não apenas `torch-2.12.1`, mas o
commit de build `7269437d655783a26cba32aa88195b741ff496aa`,
`Darwin-arm64`, `torch.inference_mode`, CPU e `BLAS_INFO=accelerate`. O helper
recusa outro build, plataforma ou BLAS antes do primeiro matmul. Cada execução
devolve e o chamador revalida um recibo com classe/ID da operação serializada,
os dois operandos ordenados e o output comprometidos por shape, bytes F32
little-endian e SHA-256. A comparação composite exige que os recibos tenham a
mesma cardinalidade das reduções ativas: 32 para imagem, 32 para vídeo e 36
para áudio.

O E4B v45 tem 21.409.278.550 bytes, SHA-256
`46f99fa94d5cd06e73ad72ce2e1a6152377e72cb577810dc98238a9b3c85a157`
e raiz estrutural
`7c30fc8d70a6c563ed30edb2d65246963ed7234949549b9b0a1ea3ae78ac97b3`.
Com a fonte fisicamente ausente, 2.130 payloads foram rehashados e imagem,
vídeo e áudio repetiram prefill, logits, cache e geração greedy a tolerância
zero, com 100/100 recibos atestados. Isso prova replay source-removed pelo
runtime fixado; não publica a árvore escalar proprietária do SGEMM, portanto o
`fidelityGate` continua corretamente com `exactReplayClaim=forbidden`.
Evidência completa em
[`docs/validation/gemma4-e4b-attested-runtime-reduction-replay-2026-07-20.md`](docs/validation/gemma4-e4b-attested-runtime-reduction-replay-2026-07-20.md).

O schema v46 remove também o arquivo helper do checkout como dependência
implícita desse replay. `authoritativeExecution` incorpora o programa Python
UTF-8 completo, seu entrypoint, 7.813 bytes e SHA-256
`6a2d8ab61c3ae2d2b5ca90c908e1d76ff3761615f0f7c662ae63323b0b611674`;
a raiz estrutural compromete esses bytes. O provider abre o artefato, valida o
programa incorporado, materializa-o somente em diretório temporário privado e
inclui o mesmo hash em cada recibo. Com checkpoint e helper do repositório
fisicamente ausentes, os 2.130 payloads foram revalidados e imagem, vídeo e
áudio repetiram prefill e geração a tolerância zero com 100/100 recibos. O E4B
v46 tem 21.409.286.964 bytes, SHA-256
`2d16f1c79553841d6b92d08216ca6987105f2e376b88e66a049dd9ff7b8c555f`
e raiz estrutural
`1b246b3b11c205119321d6bed483289b7a9e74aeb11b00fa2ecb2ebe674fe093`.
A agenda escalar proprietária continua no gate fail-closed. Evidência completa
em
[`docs/validation/gemma4-e4b-embedded-runtime-reduction-adapter-2026-07-20.md`](docs/validation/gemma4-e4b-embedded-runtime-reduction-adapter-2026-07-20.md).

O schema v48 remove a reconstrução por escopo que ainda existia na preparação
do adapter incorporado. Cada um dos cinco `invocationPrograms` schema v2 agora
declara o binding de `program.runtimeDtype`, todos os parâmetros de torre que
consome, seus caminhos de origem, domínio inteiro seguro e limite inferior. O
provider TypeScript serializa esse ambiente percorrendo somente as declarações
do programa; o adapter Python aplica os mesmos contratos antes de executar os
stages. Não há mais um branch host `vision`/`audio` que escolha parâmetros, nem
uma exceção oculta para o contexto direito de áudio.

O E4B v48 tem 21.409.324.470 bytes, SHA-256
`97f7cea6d0b2b183ae6936ea8ade402501bae2cbadc92b703335cb51a105c420`
e raiz estrutural
`eaa70f0d3a464829bd6ba19b894043160e0ffad9faf0852699e92b1eacab9894`.
Com checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads
foram revalidados e imagem, vídeo e áudio repetiram prefill, logits, 24 caches
KV e geração greedy a tolerância zero com 100/100 recibos. As 100 agendas
escalares Apple Accelerate continuam corretamente fail-closed. Evidência em
[`docs/validation/gemma4-e4b-declarative-runtime-reduction-environment-2026-07-20.md`](docs/validation/gemma4-e4b-declarative-runtime-reduction-environment-2026-07-20.md).

O schema v49 fecha uma identidade ainda mais externa desse replay. O contrato
v48 aceitava qualquer host `Darwin-arm64` com o mesmo commit de Torch e
`BLAS_INFO=accelerate`, embora o dispatch SGEMM possa mudar com o build do
macOS e a geração da CPU. `runtimeEnvironmentIdentity` agora fixa CPython
3.14.3, macOS 26.5.2 build `25F84`, kernel `25.5.0`, `Mac15,10` / Apple M3 Max
e SHA-256 `606e3853...e07a7` da configuração completa do build Torch. O adapter
incorporado deriva e compara esses campos antes do primeiro `torch.matmul`, e
cada recibo repete a identidade exata. As 100 entradas do `fidelityGate`
apontam também para esse contrato de ambiente.

O E4B v49 tem 21.409.339.988 bytes, SHA-256
`76eaf79b9eb4d8e4e0f651085aa9871767ebc454126e406fa2e90cc220e31ef8`
e raiz estrutural
`892684e8a198ceb1f28c4a57bd7020f032b2376334890fc8ac1ec78aa4ce79df`.
Com checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads
foram revalidados e as três modalidades repetiram prefill, logits, 24 caches KV
e geração greedy a tolerância zero com 100/100 recibos ligados ao ambiente.
A árvore escalar proprietária continua corretamente fail-closed. Evidência em
[`docs/validation/gemma4-e4b-runtime-platform-identity-2026-07-20.md`](docs/validation/gemma4-e4b-runtime-platform-identity-2026-07-20.md).

O schema v50 fecha a identidade binária por baixo das versões do schema v49.
O adapter agora rehasha o runtime CPython, `torch._C`,
`libtorch_python.dylib`, `libtorch_cpu.dylib` e `libc10.dylib`, comparando
tamanho e SHA-256 antes de executar. Como o Accelerate reside no dyld shared
cache, `libBLAS.dylib` é fixado por install name, arquitetura `arm64e` e Mach-O
UUID `F078C775-D8DC-3C4D-879F-A9BB228DBE06`. Esses campos estão dentro de
`runtimeEnvironmentIdentity`, da raiz de integridade e de cada um dos 100
recibos; um build byte-diferente não pode mais satisfazer apenas versões e
configuração iguais.

O E4B v50 tem 21.409.344.765 bytes, SHA-256
`4d334b8358b1f0fd19a30882d017b0eb9d0c8dd6d8fa38d60cc16346cde268d6`
e raiz estrutural
`eef690cf2c58ace5c59632598c03dd7a3d97c039a349e9992ccdd85ec73be7c2`.
Com checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads
foram revalidados e imagem, vídeo e áudio repetiram prefill, logits, 24 caches
KV e geração greedy a tolerância zero com 100/100 recibos ligados também aos
binários. A árvore escalar Apple Accelerate permanece corretamente no gate.
Evidência completa em
[`docs/validation/gemma4-e4b-runtime-binary-identity-2026-07-20.md`](docs/validation/gemma4-e4b-runtime-binary-identity-2026-07-20.md).

O schema v51 fecha a identidade do código interpretado e das bibliotecas
empacotadas que ainda ficavam fora do subconjunto binário v50. O contrato
`runtimeDependencyIdentity` fixa nove arquivos nativos: CPython, `torch._C`,
`libtorch_python`, `libtorch_cpu`, `libc10`, `libtorch`, `libshm`, `libomp` e
`libtorch_global_deps`. Ele também compromete, por árvore canônica de caminho,
tamanho e SHA-256 por folha, os 1.848 arquivos-fonte da biblioteca padrão
CPython e os 2.230 arquivos-fonte do pacote Torch. O adapter incorporado
recalcula os 4.078 arquivos / 81.903.120 bytes e todos os binários antes do
primeiro `torch.matmul`; arquivo ausente, extra dentro dos sufixos declarados,
alterado ou reordenado muda a raiz e falha fechado. A identidade do
`libBLAS.dylib` no dyld shared cache permanece presa ao UUID Mach-O.

O E4B v51 tem 21.409.350.146 bytes, SHA-256
`33070a641fdd2d891310a789e4bc7508ebb6ccb0469d29d5577e9339a4ee5099`
e raiz estrutural
`0c97380a3ae5d6599e29c4b45eda9d6c546a6c3e58259292f674a74be591a41c`.
Com checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads
foram revalidados e imagem, vídeo e áudio repetiram prefill, logits, 24 caches
KV e geração greedy a tolerância zero com 32 + 32 + 36 recibos ligados à
identidade expandida. As 100 agendas escalares Apple Accelerate continuam no
gate, com `exactReplayClaim=forbidden`. Evidência completa em
[`docs/validation/gemma4-e4b-runtime-dependency-identity-2026-07-20.md`](docs/validation/gemma4-e4b-runtime-dependency-identity-2026-07-20.md).

O schema v52 fecha o estado efetivo do processo Torch que ainda podia variar
sem alterar nenhuma identidade binária do v51. `runtimeExecutionState` fixa e
o adapter configura antes do primeiro matmul `intraopThreads=10`,
`interopThreads=14`, `deterministicAlgorithms=false`,
`mkldnnAvailable=false` e `mkldnnEnabled=true`; em seguida ele relê os cinco
campos e falha se o estado efetivo divergir. O mesmo objeto está no contrato
autoritativo, na raiz estrutural, no gate navegável e em cada recibo das 100
reduções nativas. Assim, uma execução com contagem de threads ou política de
algoritmo diferente não pode reutilizar a identidade de runtime fixada.

O E4B v52 tem 21.409.364.037 bytes, SHA-256
`4004e48c072e535af1e0fdd86f50b3c4b1bcf1fb6d9f09265db276db230e69c0`
e raiz estrutural
`5da4fad4c816f9cd782ee740f0d25c1aee4937b7d56da0ebfd6f9e75c98d4366`.
Com checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads
foram revalidados e imagem, vídeo e áudio repetiram prefill, logits, 24 caches
KV e geração greedy a tolerância zero com os 100 recibos ligados ao estado
efetivo. A agenda escalar Apple Accelerate continua corretamente fail-closed,
com `exactReplayClaim=forbidden`. Evidência completa em
[`docs/validation/gemma4-e4b-runtime-execution-state-2026-07-20.md`](docs/validation/gemma4-e4b-runtime-execution-state-2026-07-20.md).

O schema v53 fecha o estado numérico mutável que ainda podia alterar ou
reinterpretar o `torch.matmul` sem mudar os cinco campos do v52. O adapter
agora configura e relê `float32MatmulPrecision=highest`, modo determinístico
sem `warn_only`, dtype e device padrão, capability CPU, preservação de
subnormais IEEE-F32 e o modo C `FE_TONEAREST`, além dos campos de threads e
MKLDNN já existentes. A preservação de denormais é verificada por comportamento:
o menor subnormal positivo (`bits=1`) continua com `bits=1` depois da
multiplicação por um. Um teste inicia o helper com
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, que altera o estado inicial do Torch para
`high`, e prova que o programa incorporado restaura e atesta `highest` antes do
matmul.

O E4B v53 tem 21.409.366.588 bytes, SHA-256
`e1fb30baa093263dd4f5400e6fc61847748b18ebc2f851dfba80e761f184a3fe`
e raiz estrutural
`ed3cacb5a977bf4861da71003a25cfb174f1a4e46f83a33d1325bdf554c6b7a6`.
Com checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads
foram revalidados e imagem, vídeo e áudio repetiram prefill, logits, 24 caches
KV e geração greedy a tolerância zero. Todos os 100 recibos carregam o estado
numérico v2 fechado. A árvore escalar Apple Accelerate permanece corretamente
no gate, com `exactReplayClaim=forbidden`. Evidência completa em
[`docs/validation/gemma4-e4b-runtime-numeric-state-2026-07-20.md`](docs/validation/gemma4-e4b-runtime-numeric-state-2026-07-20.md).

O schema v54 fecha também o ambiente de lançamento do adapter nativo. O
provider não herda nenhuma variável do processo pai: o artefato incorpora um
mapa completo com `inheritance=none`, inicia o Python somente com esse mapa, e
o adapter o captura antes de importar Torch. Isso exclui alterações ocultas
por `DYLD_*`, `VECLIB_*`, `OMP_*`, `PYTHON*` e `TORCH*`; os 100 recibos e as
100 entradas do gate apontam para o mesmo contrato. O teste de integração
injeta `OMP_NUM_THREADS`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`,
`VECLIB_MAXIMUM_THREADS` e `DYLD_INSERT_LIBRARIES` hostis no pai e prova que
nenhuma delas alcança o helper.

O E4B v54 tem 21.409.381.135 bytes, SHA-256
`f56cd4f125b87aa094ffa2e9b57cd1f0cffef10ba1f9a360f9538b39ce2d7585`
e raiz estrutural
`0e452eaf9d365e25b9fa9206670ac29808a3f8688a4b2249af8528ae13995621`.
Os 15.992.314.836 bytes aprendidos coincidiram com a fonte imutável. Com
checkpoint e adapter do checkout fisicamente ausentes, os 2.130 payloads e a
vista escalar foram revalidados; imagem, vídeo e áudio repetiram prefill,
token greedy `184` na posição `2`, logits e 24 caches KV a tolerância zero.
Todos os 100 recibos carregam o ambiente sem herança. A agenda escalar Apple
Accelerate permanece corretamente no gate com `exactReplayClaim=forbidden`.
Evidência completa em
[`docs/validation/gemma4-e4b-closed-runtime-process-environment-2026-07-20.md`](docs/validation/gemma4-e4b-closed-runtime-process-environment-2026-07-20.md).

O replay v54 também consome esse contrato a partir do próprio artefato. O
reader entrega ao provider o objeto `executableReplay` completo, e o provider
deriva dele o source do adapter, os programas de invocação, o mapa exato do
processo e a expectativa de cada recibo. A execução não volta a importar o
mapa de ambiente ou as identidades esperadas do checkout para lançar o helper
ou validar seus recibos. O validador de compatibilidade continua recusando um
schema/runtime não suportado. O contrato público do provider expõe uma cópia isolada dessa autoridade
para que a validação de ID, SHA-256 do adapter, programa de invocação e
attestation use os mesmos dados serializados. Alteração do contrato, herança
de ambiente ou mutação de uma cópia devolvida falha fechado ou permanece
isolada. A validação está em
[`docs/validation/gemma4-e4b-artifact-owned-runtime-replay-2026-07-20.md`](docs/validation/gemma4-e4b-artifact-owned-runtime-replay-2026-07-20.md).

O schema v55 torna também o protocolo de execução parte dessa autoridade. O
JSON fixa a seleção explícita do runtime Python, os dois arquivos temporários,
o diretório de trabalho, a ordem dos argumentos, a origem do ambiente fechado,
stdin/stdout/stderr, o limite de saída e os campos exatos dos envelopes JSON e
dos tensores. O provider materializa esses valores a partir do artefato, rejeita
campos extras ou ausentes e inclui o SHA-256 do protocolo em cada recibo BMM.
Todos os 100 recibos source-removed da suite multimodal carregaram o mesmo
digest `01b0cdd4a07e2e473dd0c3ae783ae3237ea77bdbd5ff9ff738f67ca2341fd559`.
Alterar ordem de argumentos, nomes de arquivo ou envelope falha antes do
adapter. A evidência real está em
[`docs/validation/gemma4-e4b-artifact-owned-execution-protocol-2026-07-20.md`](docs/validation/gemma4-e4b-artifact-owned-execution-protocol-2026-07-20.md).

O schema v56 torna o transporte dos operandos e outputs BMM literalmente
lossless. Os arrays decimais JSON foram substituídos por payloads Base64 com
`dtype=F32`, bits IEEE-754 binary32, little-endian, layout row-major contíguo,
shape, byte length, fórmula de endereço e Base64 RFC 4648 canônico declarados
no próprio `executionProtocol`. Isso preserva inclusive o bit de sinal de `-0`
e o menor subnormal positivo, que uma conversão decimal JSON não podia garantir.
Provider e adapter recusam campos extras, encoding, tamanho, cardinalidade ou
valor não finito divergente; os recibos ligam o protocolo inteiro por SHA-256.
Evidência da E4B real está em
[`docs/validation/gemma4-e4b-lossless-runtime-tensor-transport-2026-07-20.md`](docs/validation/gemma4-e4b-lossless-runtime-tensor-transport-2026-07-20.md).

O schema v57 compromete também os envelopes exatos que atravessam essa
fronteira. `executionProtocol.transcriptCommitment` fixa JSON compacto UTF-8
sem whitespace ou bytes finais e SHA-256 sobre os bytes do request file e do
stdout. Cada recibo schema 5 registra tamanho e digest dos dois envelopes; o
executor reconstrói independentemente o request completo — inclusive programa
de invocação, ambiente da torre e operandos Base64 — e a response completa
antes de aceitar a evidência. Alterar somente um parâmetro como
`attentionContextRight`, acrescentar newline ao stdout ou trocar qualquer hash
falha fechado. Os 100 recibos source-removed da E4B carregam compromissos
válidos sob o mesmo protocolo incorporado. Evidência em
[`docs/validation/gemma4-e4b-runtime-reduction-transcript-commitments-2026-07-20.md`](docs/validation/gemma4-e4b-runtime-reduction-transcript-commitments-2026-07-20.md).

O schema v58 fecha a ordem agregada desses recibos. O
`executionProtocol.replayCommitment` declara `provider-append-order`, a
serialização `JSON.stringify` de cada recibo completo, um `LF` após cada
registro e SHA-256 sobre o transcript concatenado. Cada relatório de replay
carrega a lista ordenada de `operationIds`, byte count, digest e os recibos que
reconstroem esse digest. A comparação composite deriva do programa incorporado
as 32 operações de imagem/vídeo ou 36 de áudio e exige igualdade posicional;
contagem correta com operação omitida, duplicada ou reordenada falha fechado.
O replay source-removed real cobriu as 100 BMM em ordem e manteve comparação de
prefill e geração a tolerância zero. Evidência em
[`docs/validation/gemma4-e4b-ordered-runtime-replay-commitment-2026-07-20.md`](docs/validation/gemma4-e4b-ordered-runtime-replay-commitment-2026-07-20.md).

A regeneração real e a verificação integral com a fonte indisponível estão em
[`docs/validation/gemma4-e4b-literal-generation-program-2026-07-17.md`](docs/validation/gemma4-e4b-literal-generation-program-2026-07-17.md).

O executor paginado não mantém uma segunda implementação implícita desse
loop. `src/gemma4-literal-generation.ts` interpreta as doze atribuições
serializadas, instancia nomes como `selected_token[0]` e
`forward_state[1]`, e registra o valor produzido por cada transição. O forward
prefill/incremental entra por uma interface estreita; argmax, append, posição,
cache, EOS e seleção terminal pertencem ao programa literal. O relatório de
`replay:gemma4-paged-text` inclui `generationProgramExecution`, usando valores
literais para escalares/tokens e shape mais hash para tensores grandes. A
execução real de todas as atribuições com o checkpoint ausente está em
[`docs/validation/gemma4-e4b-literal-generation-execution-2026-07-18.md`](docs/validation/gemma4-e4b-literal-generation-execution-2026-07-18.md).

#### Evidência de redução linear por feature de saída

`probe:gemma4-linear-reduction` mede hipóteses de acumulação contra hooks
passivos da runtime nativa. Além do resultado global por perfil, uma campanha
com entradas declaradas distintas grava `outputFeatureCoverage`: spans
compactos dos perfis que coincidiram em **todas** as linhas de cada feature de
saída e em todos os prompts capturados. Isso localiza possíveis fronteiras de
microkernel/tile de saída sem ocultá-las como uma política global. A cobertura
é somente evidência diagnóstica: nunca altera o adaptador nem autoriza uma
regra de redução por coordenada sem um contrato de runtime estabelecido.

Uma amostra serializada como F32 também não prova que o módulo nativo tenha
executado em F32. As capturas lineares Gemma 4 registram
`reference.operationDtypes` para a entrada, saída e parâmetro vistos pelo
módulo antes da serialização. O probe exige esse registro e o compara com a
política literal e o dtype do peso; portanto uma promoção de backend ou uma
captura sem fronteiras de dtype não pode ser usada para justificar uma agenda
de redução BF16.

O mesmo probe exige e preserva `reference.nativeKernelEnvironment`: SHA-256 da
configuração do build Torch, contagens intra/inter-op, modo determinístico e
estado disponível/habilitado de MKLDNN. Capturas com qualquer desses campos
distintos não podem ser combinadas numa campanha. Isso limita a evidência ao
mesmo ambiente nativo, mas não transforma o microkernel opaco numa semântica
literal: sem uma agenda escalar declarada que coincida exatamente, o replay
permanece aproximado.

Por padrão, o CLI explora toda a matriz de perfis declarados. Para repetir uma
hipótese já identificada sem transformar uma campanha multi-prompt em uma nova
varredura combinatória, `--profile-id` pode ocorrer mais de uma vez. Cada ID
precisa ser um dos IDs concretos gerados pelo próprio CLI; IDs repetidos ou
desconhecidos falham antes de abrir o artefato. Por exemplo, a confirmação
source-removed das duas hipóteses de 32 lanes é explícita, sem permitir que uma
opção de desempenho invente uma agenda matemática:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/capture-a.json --trace /tmp/capture-b.json \
  --operation-id layer_0_up_proj --output /tmp/reduction-report.json \
  --profile-id arm-neon-bf16-dot-fma-32-pairwise
```

Uma hipótese BFDOT não é sinônimo da agenda de lanes acima. O perfil explícito
`arm-neon-bf16-bfdot-fma-32-pairwise` agrupa pares BF16 adjacentes por lane e
mantém os quatro registradores ativos e os quatro registradores zero da árvore
fonte. Ele permite rejeitar uma instrução BFDOT compilada/dispatchada sem
inventar essa semântica a partir de `FEAT_BF16`; a revisão E4B atual está em
[`docs/validation/gemma4-e4b-down-proj-source-dispatch-2026-07-17.md`](docs/validation/gemma4-e4b-down-proj-source-dispatch-2026-07-17.md).

Além do dtype, a captura limitada registra `reference.operationLayouts` antes
de a serialização F32 tornar tensores contíguos. O probe exige input, output e
peso com shape, strides row-major, `storageOffset: 0` e contiguidade compatíveis
com a fórmula indexada do literal. Portanto um view nativo transposto, estriado
ou deslocado não pode ser confundido com os bytes row-major incorporados no
JSON e usado para justificar uma agenda de redução; o relatório source-removed
preserva o layout nativo aceito como parte da evidência.

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
node dist/src/cli.js \
  --source ./gemma-4-E4B \
  --output ./gemma4.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
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
`generateGemma4CompositeLiteralF32` são o replay síncrono da fixture F32: eles
decodificam apenas os payloads, mas falham fechados se o programa declarar
BF16, acumulador F64 ou uma agenda de lanes que esse executor não implementa.
A regressão remove o mapa de bytes/tensores de uma fixture multimodal registrada
antes do replay e compara logits e dois tokens greedy. O E4B BF16 usa o caminho
paginado compatível descrito abaixo; nunca tem sua política reduzida a F32.

Para o checkpoint obrigatório `google/gemma-4-E4B` BF16, o mesmo comando exige
identidade imutável explícita e
escreve o artefato completo em streaming: ranges Safetensors de 12 MiB são
codificados em base64 sem acumular o pacote ou uma string de vários GiB na
heap. O resultado local contém os 2,130 payloads originais
(15,992,314,836 bytes) e é auditado por `npm run audit:literal`. O hash e
tamanho de uma exportação anterior schema v25 estão registrados em
[`docs/validation/gemma4-e4b-explicit-operand-bindings-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-operand-bindings-2026-07-18.md); sua política numérica
declara fronteiras de resultado BF16, acumuladores F32/F64 e agendas de redução
por operação do texto, áudio e visão; toda exportação deve registrar seu próprio hash, pois o programa literal
inclui as políticas numéricas. As 54 projeções/normas K/V locais dos
consumidores compartilhados são incorporadas com proveniência explícita em
`unreachableConstants`; o grafo usa apenas os KV do produtor declarado. A
evidência reproduzível está em
[`docs/validation/gemma4-e4b-literal-export-2026-07-16.md`](docs/validation/gemma4-e4b-literal-export-2026-07-16.md).

O audit padrão confirma a cardinalidade, o total de bytes e a ausência do
caminho fonte; para estabelecer que nenhum payload de tamanho igual foi
trocado, use a verificação Gemma 4 antes de tornar a fonte indisponível:

```bash
npm run audit:literal -- --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense --output ./gemma4-e4b-payload-audit.json \
  --verify-gemma4-payloads
```

Ela compara cada tensor por nome, dtype, shape e todos os bytes de storage em
faixas limitadas de 12 MiB, e registra SHA-256 concatenados da fonte e do
artefato. Não substitui replay nem comparação de runtime: só prova a fronteira
lossless de exportação enquanto o Safetensors ainda está disponível.

Exportações streaming atuais também carregam um compromisso SHA-256 por
payload. Depois de remover a fonte, o leitor pode verificar todos os bytes
incorporados sem reabrir o Safetensors:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json --verify-payloads
```

Esse compromisso detecta corrupção posterior do artefato, mas não substitui a
comparação fonte-para-literal feita antes da remoção. O E4B atual foi
regenerado em 18 de julho a partir da fonte imutável e passou tanto a
comparação completa fonte-para-literal quanto a verificação interna com o
diretório fonte temporariamente indisponível. A evidência e os hashes exatos
estão em
[`docs/validation/gemma4-e4b-explicit-eager-vision-composite-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-eager-vision-composite-2026-07-18.md).

O leitor `gemma4-composite-literal-reader.ts` abre esse JSON em streaming: ele
indexa os offsets dos payloads base64 e valida a estrutura semântica sem
`JSON.parse` do artefato completo. Cada faixa de bytes é decodificada a partir
do próprio JSON, sem abrir shard Safetensors. Para auditar o índice ou uma
faixa específica (o comando não executa o modelo), ou navegar pelas operações
e fórmulas escalares substituídas:

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --tensor model.language_model.embed_tokens.weight \
  --offset 0 --byte-length 4096

npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-operations --output /tmp/gemma4-operation-index.json
```

Isso fecha a fronteira de leitura seletiva, substituição escalar em todas as
operações composite registradas e declaração autocontida do estado de geração;
ainda não é uma alegação de replay numérico BF16 ou de equivalência da geração
E4B real. A medição do artefato real e o limite
de memória do executor atual estão em
[`docs/validation/gemma4-e4b-literal-reader-2026-07-16.md`](docs/validation/gemma4-e4b-literal-reader-2026-07-16.md).
Isso prova materialização completa e ausência do caminho do checkpoint no
artefato, mas ainda não é replay E4B sem fonte nem comparação numérica com o
runtime autoritativo.

`paged-dense.ts` é a fronteira seguinte: aceita somente matrizes literais
row-major densas `F32`/`F16`/`BF16`, lê linhas por `readTensorBytesRange` e
executa embedding e linear na agenda escalar ou de lanes explicitamente
declarada pelo artefato. A leitura é limitada por uma janela explícita e não aceita
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
do JSON literal já indexado. O comando textual standalone não recebe entradas
multimodais. Para composição, o mesmo interpretador expõe fronteiras distintas
para os embeddings antes do scatter, a projeção PLE depois do scatter e as
camadas/epílogo a partir do prelude preparado; esses cálculos não são
duplicados num executor composite paralelo.

```bash
npm run replay:gemma4-paged-text -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --input-ids 2,106,3 --max-new-tokens 1 \
  --allow-unverified-fidelity --output ./paged-text-report.json
```

O relatório inclui hash dos logits, tokens greedy, produtores KV, janela de
leitura e RSS, incluindo o máximo residente (`maxRssKiB`) do processo. A
regressão atual prova equivalência byte-a-byte de prefill e
dois passos cached contra o executor eager apenas para a fixture Gemma 4
registrada após remover os tensores de origem. `--allow-unverified-fidelity`
é obrigatório para a E4B real atual: ele registra que o replay é diagnóstico
aproximado, não uma alegação de fidelidade estabelecida. Um prefill E4B real de
um token mais um decode cached foi executado somente do artefato literal. A
comparação multimodal autoritativa posterior é descrita abaixo; o marcador
continua proibido porque as torres não são exatas.

Há agora uma comparação nativa separada para esse ramo textual: o capturador
fixado em Transformers 5.5.0/PyTorch 2.12.1 executa
`Gemma4ForConditionalGeneration` BF16 e vincula a captura aos SHA-256 do
`config.json` e do Safetensors. `compare:gemma4-paged-text-trace` executa o
candidato somente a partir do artefato e pode exigir que a fonte esteja ausente
com `--assert-source-unavailable`. A primeira execução E4B reproduziu o token
greedy `184`, mas é explicitamente `approximate`: logits e os 24 caches KV
produtores divergem do eager BF16. Comandos, hashes e métricas estão em
[`docs/validation/gemma4-e4b-native-text-differential-2026-07-16.md`](docs/validation/gemma4-e4b-native-text-differential-2026-07-16.md).

Uma repetição independente contra a revisão imutável oficial
`411aa17b749aa952df1359d2dcea73917a544d9a` confirmou a mesma fronteira,
inclusive a ausência do checkpoint durante o replay candidato; a evidência
atualizada está em
[`docs/validation/gemma4-e4b-immutable-source-revalidation-2026-07-16.md`](docs/validation/gemma4-e4b-immutable-source-revalidation-2026-07-16.md).

A fronteira textual atual substitui os perfis históricos por dispatch de
classe baseado na fonte PyTorch fixada. Todas as 344 lineares BF16 compatíveis
declaram a árvore ARM de 32 lanes; todas as 302 RMSNorm declaram o
`cascade_sum` F32 de quatro lanes/quatro registradores; 84 GELU, 66 RoPE e o
softcap final declaram os kernels SLEEF e cada cast BF16 observável. As 42
atenções compatíveis também declaram um único contrato eager BF16: dot ARM
QK, score BF16, softmax F32/SLEEF com redução vetorial de quatro lanes,
probabilidade BF16 e dot ARM com V. Com o checkpoint removido, um forward real
em token `184`/posição `1` passou as 1.229 atribuições, logits e 24 caches KV
com erro zero; quatro passos cached geraram `184,3910,531,974` com logits de
seleção, 96 snapshots KV ao longo dos quatro passos, logits terminais e 24 caches terminais
também em erro zero. O loop seguinte inspecionou essa implementação e repetiu
independentemente a captura/reprodução dos quatro passos, aceitando a fronteira
textual cached. O scalar view expõe `SLEEF_SIN_F32`, `SLEEF_COS_F32`,
`SLEEF_TANH_F32`, `SLEEF_EXP_F32`, as árvores ARM e cada cast BF16 dentro das
fórmulas indexadas. O range reducer SLEEF `rempif` completo também está
transcrito: a tabela F32 little-endian de 1.664 bytes é incorporada ao contrato
de todas as 66 operações RoPE. O loop seguinte repetiu independentemente uma
comparação real na posição 125: as 1.229 atribuições, logits e caches passaram
em tolerância zero com a fonte removida, aceitando também a fronteira
full-range. O checkpoint continua proibido porque a execução diferencial
multimodal real permanece aberta.
Comandos, hash do artefato e métricas estão em
[`docs/validation/gemma4-e4b-exact-cached-attention-2026-07-18.md`](docs/validation/gemma4-e4b-exact-cached-attention-2026-07-18.md); a fronteira anterior está preservada em
[`docs/validation/gemma4-e4b-source-dispatched-text-math-2026-07-18.md`](docs/validation/gemma4-e4b-source-dispatched-text-math-2026-07-18.md). O contrato
full-range e sua validação estão em
[`docs/validation/gemma4-e4b-full-range-rope-2026-07-18.md`](docs/validation/gemma4-e4b-full-range-rope-2026-07-18.md).

O ramo de áudio real também é agora executável diretamente do artefato. O
materializador limitado por tensor decodifica `752` referências densas a partir
do próprio JSON e alimenta as `619` atribuições nomeadas da torre de 12 camadas,
sem catálogo ou fallback para Safetensors. O adaptador aplica a política BF16
por classe em todo o ramo: 134 lineares bias-free usam a árvore ARM de 32 lanes,
109 RMSNorm usam `cascade_sum`, Conv2d e depthwise BF16 usam o GEMM ILP4 do
`slow_conv2d`, LayerNorm por canal usa Welford vetorial e o linear com bias usa
o contrato `addmm`. Q/K, posição relativa, softcap/máscara e o cast BF16 do
contexto também são atribuições separadas; cada resultado de módulo BF16 é
estreitado antes do consumidor seguinte. O score antes opaco também foi aberto
por classe em AC, BD não deslocado, relative shift, soma F32, softcap SLEEF e
máscara causal. Somente os 12 AC, 12 BD e 12 value matmuls F32 permanecem
`runtime-defined`; a vista escalar falha fechada nessas 36 reduções e expõe as
demais etapas por coordenada.

A captura autoritativa de um frame agora compara 444 fronteiras. Com o
checkpoint ausente, 374 passam em tolerância zero; AC passa em 1/12 e BD em
0/12, localizando a divergência no `sgemm` nativo antes do relative shift. O
contexto F32 produzido pelo value matmul é capturado diretamente antes do cast
e passa em 12/12; um relatório source-anchored adicional prova em 12/12 o cast
BF16 a partir desses valores autoritativos. O relatório AC/BD também mantém
48/48 instâncias de relative shift, soma, softcap e máscara bitwise exatas.
Todos os consumidores posteriores e `audio_features [1,2560]` permanecem
exatos. A saída terminal tem erro absoluto e relativo zero, cosseno e top-k
overlap `1`, mas a classificação global permanece `approximate` porque uma
entrada exata não prova a agenda escalar interna de `sgemm`. Comandos, hashes,
políticas e métricas estão em
[`docs/validation/gemma4-e4b-explicit-audio-context-boundary-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-audio-context-boundary-2026-07-18.md); a decomposição anterior foi preservada em
[`docs/validation/gemma4-e4b-explicit-audio-score-pipeline-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-audio-score-pipeline-2026-07-18.md),
[`docs/validation/gemma4-e4b-explicit-audio-native-classes-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-audio-native-classes-2026-07-18.md) e
[`docs/validation/gemma4-e4b-source-removed-audio-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-audio-2026-07-18.md).

A fronteira source-removed cobre a torre vision compartilhada por imagem e
vídeo. A captura agora exige `attn_implementation="eager"`; a captura anterior
rotulada eager estava efetivamente em SDPA e foi substituída. O programa de
427 atribuições separa cada uma das 16 atenções em score matmul, masked softmax
e value matmul. Softmax declara soma de chaves ascendente e cast BF16; pooling
declara `ordered-fma` com `F32(1/F32(9))`; RMSNorm reproduz a fronteira
`torch.pow(x,-0.5)` para toda a classe. Score/value batched continuam
`runtime-defined` e falham fechados na vista escalar, pois sua agenda nativa
ainda não foi comprovada de forma geral.

Capturas eager reais de imagem `[1,9,768]` e vídeo `[1,2,9,768]` compararam
294 fronteiras cada. Imagem passou 294/294 com erro zero e saída
`image_features [1,2560]` exata. Vídeo passou 292/294 e terminou exato; somente
`vision_layer_8_attention` (`5.960464477539063e-8`) e
`vision_layer_9_attention_scores` (`0.0000152587890625`) preservam diferenças
internas de batched matmul. A evidência, políticas e navegação escalar estão em
[`docs/validation/gemma4-e4b-explicit-eager-vision-composite-2026-07-18.md`](docs/validation/gemma4-e4b-explicit-eager-vision-composite-2026-07-18.md).

O caminho literal agora conecta essas torres ao modelo composto inteiro.
`gemma4-literal-composite.ts` executa embedding PAD, torre e scatter, projeção
PLE pós-scatter, máscaras visual full/sliding e as 42 camadas textuais somente
dos payloads incorporados. O mesmo forward alimenta diretamente o programa
estruturado de geração incorporado no schema v29; multimodalidade ocorre no
prefill, e decode recebe somente token, posição e cache explicitamente ligados,
com os demais inputs enumerados como omitidos.

No schema v30, a decisão de executar ou omitir cada torre já não pertence aos
`if` do executor: `forwardControl` seleciona os três ramos, seus aliases de
ausência, máscaras, posições e estado de cache diretamente do JSON antes de o
mesmo forward alimentar o controle greedy.

No schema v31, os valores que selecionam esses ramos também atravessam
`inputContract` antes do replay. Assim, o leitor não decide por conta própria o
que significa um tensor de pixels/áudio, quais layers possuem cache, como a
máscara se relaciona ao cache ou quantos placeholders cada torre deve
produzir; o JSON contém e executa essas restrições.

No schema v32, o resultado também deixa de depender da interpretação do
leitor. `outputContract` valida os tensores públicos e o mapa de intermediários,
a substituição modal dos IDs, os 24 caches produtores e as relações entre
prefill, cada snapshot greedy e o estado terminal. O mesmo contrato é exposto
pela vista `--end-to-end-calculation`, junto dos inputs e controles já
serializados.

No schema v33, cada estado incremental deixa um snapshot completo de logits e
cache. O output contract verifica `selection_logits[step+1] ==
step_forward_logits[step]`, o token contra o argmax de seu próprio snapshot e
o par terminal contra o último forward executado; adulterar somente um lado da
cadeia falha fechado.

No schema v34, o caminho inverso também é dado: cada input ordenado de uma
atribuição aponta para o produtor e lista as coordenadas exatas que seu
programa escalar lê. Vistas concretas materializam esses templates sem apagar
os bounds da redução; janelas diagnósticas são rotuladas `windowed`, nunca
confundidas com a redução completa. Dependências que afetam somente dimensão
ou seleção ficam marcadas `shape-or-control-only` em vez de receberem uma
leitura tensorial inventada.

O trace composite despacha `image`, `video` e `audio` por um contrato de
modalidade único, que fixa token type, input, feature output e scatter e rejeita
misturas antes de abrir a fonte. `compare:gemma4-literal-composite-suite`
exige agora as três modalidades exatamente uma vez, sob a mesma identidade e
runtime autoritativo, com tolerância absoluta/relativa zero. Cada modalidade
precisa passar as cinco fronteiras de prefill, token/posição greedy, logits
terminais e todos os caches; um resultado aproximado, trace trocado, modalidade
ausente ou contagem incompleta de BMM invalida a suite.

As capturas atuais usam
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`.
Com a fonte fisicamente ausente, imagem, vídeo e áudio passaram
`lossless-within-dtype`: token `184` na posição `2`, erros absoluto/relativo
zero e 24/24 caches por modalidade. A suite também exige e preserva as 100 BMM
`runtime-defined` (32 imagem, 32 vídeo, 36 áudio); portanto esta cobertura ampla
não inventa a agenda escalar Apple Accelerate nem declara o checkpoint
concluído. O contrato, comandos, hashes e métricas estão em
[`docs/validation/gemma4-e4b-source-removed-modality-suite-2026-07-18.md`](docs/validation/gemma4-e4b-source-removed-modality-suite-2026-07-18.md).

Uma captura posterior de todas as 1.229 atribuições declaradas do Gemma4Text
torna a divergência localizável sem reabrir o checkpoint durante o candidato.
Ela corrige o cast BF16 observável do embedding (`sqrt(2560)` para `50.5` e o
produto BF16), e mantém a acumulação escalar ordenada F64 para os
linear/RMSNorm sem evidência de kernel mais específica. Campanhas posteriores
com a fonte removida generalizaram a árvore ARM BF16 para toda a classe linear
compatível por dtype, layout, transpose, ausência de bias e dispatch
autoritativo; nenhuma atribuição é selecionada pelo ID ou somente pelo shape.
Cada instância declara 32 lanes, oito registradores, a árvore de registradores,
fold horizontal pareado e tails. A instrumentação é comparada a um forward nativo sem hooks e falha se
logits ou cache KV mudarem. A comparação completa ainda permanece
`approximate`, documentada em
[`docs/validation/gemma4-e4b-native-operation-checkpoints-2026-07-16.md`](docs/validation/gemma4-e4b-native-operation-checkpoints-2026-07-16.md).

Para evitar transformar um palpite de SIMD em semântica do artefato, há também
`probe:gemma4-linear-reduction`. Ele vincula uma projeção linear ao produtor e
resultado nomeados de **duas capturas nativas independentes** com fingerprint
de IR e identidade de fonte iguais, lê somente ranges do JSON literal e exige
igualdade bitwise dos dois traces antes de comparar todos os elementos após o
cast declarado. No E4B, duas capturas completas de 1.229 atribuições foram
idênticas; o probe confirma as 32 lanes de `layer_0_gate_proj`, mas nenhuma das
agendas escalar/F32/F64 ou 2..256 lanes testadas resolve
`layer_0_up_proj`. O probe agora também varia a árvore de fold horizontal
(ascendente, descendente e pareada balanceada) e o limite explícito de FMA;
para os operandos BF16 E4B, FMA não altera os candidatos observados. Essas
primeiras campanhas de um único prompt não foram promovidas: somente as
campanhas posteriores de três entradas, com a árvore de registradores completa,
podem vincular uma agenda emitida. Os resultados históricos estão em
[`docs/validation/gemma4-e4b-linear-reduction-probe-2026-07-16.md`](docs/validation/gemma4-e4b-linear-reduction-probe-2026-07-16.md).
Quando houver capturas de prompts distintos, `--min-distinct-inputs N` exige
dois captures bitwise-idênticos para cada prompt/posição e só mantém perfis
exatos em todos os grupos; identidade de fonte/runtime/IR/dtype não pode variar
entre eles. Assim uma coincidência em uma ativação não pode virar contrato de
kernel do artefato. Todo relatório agora declara `candidateSelection` como
`unique`, `ambiguous` ou `none`; somente `unique` contém o único `profileId`
que uma alteração posterior do adaptador pode considerar. Um conjunto de
`exactProfileIds` com mais de um elemento é explicitamente `ambiguous`, não uma
permissão para escolher uma agenda equivalente por shape.

Uma correção posterior completou a terceira etapa da árvore de oito registros
ARM que a primeira implementação tinha omitido. Com o programa literal
regenerado, a agenda explícita `arm-neon-bf16-dot-fma` de 32 lanes e fold
pareado para `layer_0_up_proj` coincidiu em todas as coordenadas BF16 de seis
capturas independentes, cobrindo `[2]`, `[17]` e `[2,17]`, com o checkpoint
removido durante o probe. O fold ascendente falhou nessa mesma campanha, por
isso ele não é emitido. A mesma campanha para `layer_0_gate_proj` também
selecionou unicamente essa árvore completa. Essas são reivindicações candidatas
somente para as duas atribuições e o ambiente fixado, não uma conclusão do
checkpoint Gemma 4; os comandos, checksums e limites estão em
[`docs/validation/gemma4-e4b-arm-32-gate-up-reduction-candidate-2026-07-17.md`](docs/validation/gemma4-e4b-arm-32-gate-up-reduction-candidate-2026-07-17.md).

Uma revisão independente subsequente repetiu o probe source-removed do
`layer_0_down_proj` e uma captura completa atualizada de `[2]`/`[0]`. Os seis
perfis down continuam `ambiguous`, enquanto `layer_0_gate_proj` e
`layer_0_up_proj` passam bitwise; por isso o primeiro desvio completo é
`layer_0_down_proj` (`0.001953125` máximo absoluto), seguido por
`layer_0_post_ffn_norm`. O prefill mais um decode ainda escolhe `184`, mas os
logits de seleção e terminais permanecem `approximate` (`0.1875` máximo
absoluto). A fonte ficou indisponível durante ambos os replays candidatos;
nenhuma agenda down foi instalada. Veja
[`docs/validation/gemma4-e4b-current-text-boundary-2026-07-17.md`](docs/validation/gemma4-e4b-current-text-boundary-2026-07-17.md).

Uma campanha posterior ampliou o `layer_0_down_proj` para oito linhas de uma
sequência diversa e também introduziu entradas diagnósticas explicitamente
marcadas: escala BF16 por potência de dois e por um escalar BF16 serializado
pelos bits IEEE. O helper primeiro prova que hooks passivos preservam o
forward/KV original; só então chama o mesmo `Linear` nativo sobre a ativação
transformada. Esses traces são segregados de `model-forward` pelo contrato e
não contam como fidelidade do modelo. Mesmo com a escala BF16 `1.5`
(`factorBf16Bits=16320`), as seis agendas down continuam exatamente
compatíveis após a fonte ser removida; nenhuma política foi instalada. Os
comandos, regras de separação e resultado estão em
[`docs/validation/gemma4-e4b-down-proj-transformed-activation-2026-07-17.md`](docs/validation/gemma4-e4b-down-proj-transformed-activation-2026-07-17.md).

Para repetir essa investigação sem serializar novamente o trace completo de
1.229 atribuições, `capture:gemma4-linear-reduction` captura somente uma
projeção MLP declarada (`layer_<n>_(gate|up|down)_proj`) e sua atribuição
produtora nomeada. O helper aceita somente esses módulos registrados, observa
o `Linear` nativo por hooks passivos e rejeita a captura se o segundo forward
alterar logits ou qualquer KV cache do forward sem hooks. Cada pequeno trace
continua vinculado aos checksums de toda a fonte e ao fingerprint do programa;
`probe:gemma4-linear-reduction` continua exigindo dois traces independentes e
é executável com o diretório da fonte removido. A repetição limitada do
`layer_0_up_proj` E4B está em
[`docs/validation/gemma4-e4b-bounded-linear-reduction-2026-07-17.md`](docs/validation/gemma4-e4b-bounded-linear-reduction-2026-07-17.md).
Toda captura nativa Gemma 4 agora exige `--device cpu` ou `--device mps` e
persiste o valor em `reference.executionDevice`. O probe recusa traces sem
esse campo e também recusa misturar dispositivos: CPU e MPS podem ter
reduções BF16 distintas. As campanhas E4B históricas foram CPU — o helper
anterior não movia modelo nem tokens para MPS — portanto elas não são
evidência de uma agenda MPS; a nova opção torna essa fronteira reproduzível.
Uma recaptura CPU real e o probe source-removed com esse contrato estão em
[`docs/validation/gemma4-e4b-device-bound-capture-2026-07-17.md`](docs/validation/gemma4-e4b-device-bound-capture-2026-07-17.md).
O vocabulário do probe também representa tiles com grupos contíguos de termos
por lane (`--tiled-lane-counts` e `--tiled-terms-per-lane`), em vez de supor
que toda SIMD use `i mod lanes`. Uma campanha source-removed de 122 agendas
E4B ainda não encontrou perfil exato; a evidência está em
[`docs/validation/gemma4-e4b-tiled-linear-reduction-2026-07-17.md`](docs/validation/gemma4-e4b-tiled-linear-reduction-2026-07-17.md).
O mesmo probe declara também FMA escalar ordenado e parciais de termos
contíguos (`--blocked-terms-per-block`): cada parcial e sua entrada no
acumulador são fronteiras F32 explícitas, nunca uma suposição implícita sobre
uma instrução BF16. Duas novas capturas E4B e uma execução com a fonte removida
rejeitaram todos os 16 candidatos bloqueados de 2, 4, 8 e 16 termos; a evidência
permanece separada da semântica instalada em
[`docs/validation/gemma4-e4b-blocked-linear-reduction-2026-07-17.md`](docs/validation/gemma4-e4b-blocked-linear-reduction-2026-07-17.md).
Por fim, `--blocked-tiled-lane-counts` e
`--blocked-tiled-terms-per-lane` testam uma fronteira diferente: lanes F32
reiniciam em cada tile contíguo, o tile é dobrado na ordem declarada e somente
seu parcial entra no acumulador F32 dos tiles. A campanha source-removed de 81
perfis E4B também não encontrou agenda exata; nenhuma aproximação foi instalada
no adaptador. A evidência e o comando reprodutível estão em
[`docs/validation/gemma4-e4b-blocked-tiled-linear-reduction-2026-07-17.md`](docs/validation/gemma4-e4b-blocked-tiled-linear-reduction-2026-07-17.md).
O vocabulário literal também contém a árvore finita de registradores do
dot-product BF16 ARM de PyTorch: oito registradores F32, FMA por lane, árvore
`0+4`/`0+2` e fold horizontal declarado. Ele suporta as larguras 32 e 64 da
abstração vetorial sem chamar um kernel durante o replay. Duas capturas E4B
novas e uma campanha source-removed de 231 perfis rejeitaram as quatro variantes
ARM (10.218 ou 10.229 coordenadas divergentes), portanto nenhuma é instalada
como semântica Gemma 4; a melhor aproximação continua 32 lanes balanceadas com
um erro BF16. A evidência e a distinção entre fonte possível e dispatch provado
estão em
[`docs/validation/gemma4-e4b-arm-vector-reduction-2026-07-17.md`](docs/validation/gemma4-e4b-arm-vector-reduction-2026-07-17.md).

### Comparação com captura autoritativa

`npm run compare:trace -- --source <checkpoint> --trace <captura.json> --report <relatorio.json>` executa a fronteira completa de validação declarada pela captura (`F32` ou `F64`): reabre o contêiner, confere SHA-256 de `config.json` e de cada shard/arquivo que participa do checkpoint, reconstrói o IR, materializa os pesos por range, executa o interpretador correspondente e compara cada operação e cache KV com a captura. O relatório só é escrito depois de todas essas verificações.

A captura é JSON `schemaVersion: 1`, `kind: "execution"`, e exige:

- `source.files`: lista exata de caminhos relativos seguros e checksums SHA-256 do checkpoint;
- `irFingerprint`: SHA-256 do IR serializado antes da política declarada;
- `candidatePolicy`: `dtype: "F32"` ou `"F64"` e o identificador do executor candidato;
- `reference`: identidade imutável do runtime/modelo/revisão, `executionDevice` quando a captura define dispositivo, tokens de entrada e cada operação por `operationId`, além do cache KV pós-RoPE BHSD;
- cada tensor com o mesmo `dtype` da política (`"F32"` ou `"F64"`), `shape` e `valuesBase64` com bytes IEEE-754 little-endian — não arrays decimais sujeitos a arredondamento JSON nem mistura silenciosa de precisão.

O comando recusa arquivo ausente/extra, checksum divergente, fingerprint de IR diferente, dtype implícito, operação duplicada, shape/payload inválido e evidência incompleta. A captura ainda precisa ser produzida por hooks verificados no runtime autoritativo; esse mecanismo não transforma o executor escalar em uma referência de Transformers, MLX ou llama.cpp.

## Objetivo de engenharia Gemma 4

O foco do repositório é produzir um programa JSON matemático, navegável e
autossuficiente para o Gemma 4 denso e sem quantização. Pesos e constantes
aprendidos devem estar embutidos sem perda, com decodificação exata e vistas
escalares capazes de substituí-los por literais numéricos. Entradas, operações,
intermediários, transições de cache, logits e geração greedy devem formar um
grafo ordenado que possa ser reproduzido sem reabrir o checkpoint.

A arquitetura deve permanecer coesa, extensível e fail-closed, com contratos
explícitos, Clean Code, SOLID, testes independentes e documentação compatível
com o comportamento. Evidência de conclusão exige identidade imutável da fonte,
integridade do artefato, replay com a fonte fisicamente ausente e comparação
diferencial com um runtime Gemma autoritativo.

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
ser um limite separado da montagem texto+imagem+áudio+vídeo: o replay literal
BF16 do próprio tower e seu diferencial real estão descritos abaixo, enquanto
o scatter composto, cache e geração multimodal ainda não têm comparação
autoritativa.

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

## Programa global real simplificado (schema 61)

O schema 61 preserva o programa IEEE anterior como baseline e acrescenta
`realSimplifiedProgram`, com semântica `gemma4-exact-real-simplified-v1`.
Nesse programa, cada valor BF16/F16/F32 aprendido é interpretado como o número
racional diádico exato codificado pelos bits incorporados. Casts e
arredondamentos IEEE intermediários são removidos; `exp`, `sin`, `cos`, `tanh`,
`log1p` e `sqrt` denotam as funções matemáticas reais, e a quantização ocorre
somente na fronteira pública declarada.

O JSON contém uma função para cada uma das 2.560 dimensões de
`final_hidden_states` e para cada um dos 262.144 logits. As funções compõem um
DAG de operações content-addressed: isso é matematicamente equivalente à
substituição integral, mas evita duplicar fisicamente a mesma subexpressão até
esgotar memória. Os únicos inputs livres da closure textual são
`hidden_states_0`, `ple_inputs`, posições, máscaras e cache; pesos continuam no
artefato e são lidos como racionais, não como variáveis.

As 100 reduções antes marcadas `runtime-defined` também possuem definições
autônomas como somas finitas reais de produtos sobre seus operandos (32 de
imagem, 32 de vídeo e 36 de áudio). Assim, a representação simplificada não
contém uma fronteira `unpublished-provider-boundary`; a ordem do kernel nativo
continua somente no baseline IEEE. A nova representação é deliberadamente uma
candidata numérica e não promete paridade bitwise sem validação diferencial.

O diferencial vetorizado pode ser executado com:

```bash
./venv/bin/python scripts/gemma4-real-differential.py \
  --source ./gemma-4-E4B-dense \
  --input-ids 2,818,5279,529,7001,563 \
  --output /tmp/gemma4-real-differential.json
```

Ele compara o eager BF16 com operações F64 sem casts F32/BF16 intermediários,
partindo das mesmas fronteiras de embedding/PLE e aplicando BF16 RNE somente
nos logits terminais.

O programa pode ser projetado para um arquivo SSA navegável, com uma atribuição
final por dimensão e subexpressões compartilhadas executadas uma única vez:

```bash
npm run export:gemma4-real-ssa
```

O resultado padrão é
`artifacts/gemma4-e4b-dense.real-simplified.ssa.json`. Ele contém 542.655
statements, 1.321 funções e 264.704 outputs `calc_*`. A memória de constantes
é o artefato literal autenticado declarado em `constantMemory`; portanto o
checkpoint Safetensors original não é consultado.

O lowering vetorizado correspondente é compilado para um plano compacto e
persistente antes de iniciar o runtime:

```bash
npm run compile:gemma4-vector-lowering -- \
  --artifact artifacts/gemma4-compiled-global-runtime-bundle/constants.literal.json \
  --bind-bundle artifacts/gemma4-compiled-global-runtime-bundle
```

Isso grava `vectorized-real-lowering.json`, adiciona seu SHA-256 ao manifesto
v2 do bundle e vincula as 1.221 closures globais às nove famílias de kernels
MLX/Metal. O worker no modo `real` não aceita um certificado calculado apenas
em memória: ele carrega o plano, verifica seu SHA-256 e exige que contrato,
bindings, raiz do artefato e SHA da seção `realSimplifiedProgram` coincidam com
o certificado produzido após a validação integral do JSON. O SSA standalone de
430 MB continua sendo a representação navegável; ele não é reparsed no hot
path de geração.

Ao criar um bundle novo, passe o mesmo plano com
`bundle:gemma4-compiled -- --real-lowering-plan <arquivo> ...`; bundles sem o
plano persistido não são reconhecidos como prontos pela interface.

Além da validação de inicialização, cada entrada no hot path real deve cobrir
exatamente as 1.221 operações da closure global, na ordem dos bindings e até o
softcap dos logits. O relatório direto publica
`vectorizedRealExecution`: quantidade autorizada, primeiro/último ordinal,
hash dos bindings e SHA-256 da sequência efetivamente despachada. Omissão,
reordenação ou substituição de qualquer operação falha antes do kernel Metal.

O plano vetorial schema 3 fecha também a ligação que antes terminava nas
operações intermediárias. As 264.704 funções públicas são validadas uma a uma:
cada raiz precisa ser uma chamada da operação realmente presente na closure,
as dimensões de cada família precisam ser contíguas desde zero e os parâmetros
e a quantização final não podem variar. O plano atual cobre 2.560 dimensões de
`final_hidden_dimension` e 262.144 `terminal_logit`, comprometidas por
`outputBindingsSha256`
`221e0d09ee3fe663254223e06f1e6c8b965ef28c12a8188b1f5c3bb1c6b4f0d8`.

Além do compromisso semântico, `standaloneSsaOutputsSha256`
`8545e86dbfd50fad57471015c39ac80b4b1ec49e6d6ac5c25f8a3cd679dde9f0`
é calculado sobre o array `outputs` físico de `global-formulas.ssa.json`. A
montagem ou atualização do bundle relê esse trecho em streaming e recusa SSA
alterado, truncado ou vindo de outro plano. O recibo de cada geração publica
`completeOutputFunctions`, a contagem e o mesmo hash. Isso prova que o caminho
vetorial cobre todas as fórmulas finais persistidas sem carregar 430 MiB no hot
path. O schema 3 compila essas raízes em uma unidade paramétrica compartilhada
`gemma4-text-real-final-vectors`: o plano ordena as 1.221 operações da closure,
compromete a projeção `{id, op, output}` por
`orderedDispatchSha256` e declara um único despacho lógico por forward. No
startup, o worker resolve as operações do artefato na ordem do plano, valida a
unidade integral uma vez e conserva a mesma identidade no hot path residente.
Por isso o manifesto agora declara honestamente
`directlyExecutesGlobalFormula: true`: não porque interprete os 430 MiB de SSA,
mas porque a DAG paramétrica compilada — equivalente às 264.704 raízes — passa
a dirigir a execução, em vez de servir apenas como certificado de uma lista
reconstruída pelo runtime.

A geração greedy diferencial possui uma interface local funcional:

```bash
npm run compare:gemma4-real-ui
```

Abra `http://127.0.0.1:8787`, informe o prompt e a quantidade de tokens. A
opção padrão detecta `artifacts/gemma4-compiled-global-runtime-bundle` e usa o
próprio diretório como fonte do Transformers, constant pool, pesos do runtime e
tokenizer. Assim, a interface não reabre `gemma-4-E4B-dense`; o baseline lê a
cópia byte a byte idêntica de `model.safetensors` empacotada no bundle e o
runtime direto resolve todas as referências relativas ao mesmo diretório. Use
`--compiled-bundle <diretório>` para selecionar outro pacote autocontido, ou
`--source` mais `--literal-artifact`/`--binary-pool` para uma comparação
explícita. Se um `--compiled-bundle` não contiver constantes, pesos, tokenizer
e config, a inicialização falha fechada.

Bundles novos também incluem `constants.runtime-index.json`, uma projeção de
execução autenticada schema 2 de aproximadamente 2,71 MiB. Ela contém somente
programa textual, metadados/offsets das constantes, geração, manifesto de
integridade e certificado do lowering. O JSON auditável de 20 GiB continua no
bundle, mas não é aberto no hot path. Para criar ou atualizar a projeção:

```bash
npm run index:gemma4-compiled -- \
  --bundle ./artifacts/gemma4-compiled-global-runtime-bundle
```

O manifesto schema 3 vincula o índice ao SHA-256 do constant pool, ao próprio
SHA-256 do índice e à raiz de integridade estrutural. Na compilação, o gerador
abre e valida integralmente o artefato; no runtime, cada seção materializada é
rehashada e comparada ao compromisso correspondente. Assim o worker não confia
em cache local, não repete validações sobre 270 MiB de fórmulas e não precisa
varrer novamente `constants.literal.json`.

A interface executa Transformers eager BF16, a recomposição F32/F64 e o runtime
compilado MLX/Metal, escolhe os tokens independentemente e compara os
resultados e tempos. O endpoint `POST /api/compare-stream` entrega NDJSON por
token enquanto a geração direta ainda está em curso. A interface mantém uma
sessão entre envios, possui cancelamento que drena os workers sem avançar a
conversa e oferece **Nova conversa** para descartar o prefixo atual. Os três
executores mantêm seus próprios KV caches e só reutilizam um prefixo quando os
IDs anteriores coincidem exatamente; as métricas mostram quantos tokens foram
reaproveitados e quantos precisaram de novo prefill. O resumo visual separa
paridade token a token, tempo até o primeiro token do executor, aceleração
contra o Transformers e percentual de prefill evitado. O mesmo contrato é
publicado como `directExecutionMetrics`, com `firstTokenForwardSeconds`,
`firstTokenWallSeconds`, tempo dos decodes seguintes, reutilização KV e
`baselineSpeedup`, para que testes e clientes não precisem extrair números do
texto técnico da interface.

O painel distingue explicitamente os dois contratos de paralelismo. Na
referência PyTorch, a contagem de threads configura o pool CPU. No runtime
direto MLX, essa contagem não controla os GEMMs: o paralelismo ocorre dentro dos
kernels vetorizados Metal sobre as dimensões dos tensores, atualmente em uma
command stream. O relatório publica `parallelExecutionBackend`,
`configuredHostThreads`, `hostThreadSettingApplied`, `metalCommandStreams` e
`parallelTerminalOutputDimensions`; assim `--direct-threads` não é apresentado
como aceleração efetiva do backend MLX.

Uma validação real em dois turnos no modo chat gerou 8/8 tokens iguais ao
Transformers. No segundo turno, 17 tokens do prefixo foram reutilizados, 18
foram calculados e 48,6% do domínio de prefill foi evitado. O painel do
navegador mostrou primeiro token em 130,7 ms e 7,62x contra o baseline naquela
execução isolada. Esses tempos descrevem o ensaio, não uma garantia universal;
a paridade e as métricas são recalculadas para cada prompt enviado.

O agendamento padrão da interface é **Isolada (fiel)**: primeiro executa e
entrega o texto do runtime compilado sem cálculo concorrente e só então roda o
original e a recomposição de compatibilidade. O relatório separa a latência da
fase compilada, a fase de referência e o tempo de parede total. **Paralela
(stress)** continua disponível para medir deliberadamente a contenção de CPU,
GPU e memória unificada; seus tempos não devem ser tratados como a latência
isolada de inferência. A API aceita `measurementSchedule: "isolated" |
"parallel"`, usando `isolated` quando o campo é omitido.

A prontidão inicial carrega somente o runtime compilado e um worker dedicado de
tokenização; o processo Transformers de aproximadamente 9 GiB permanece em
estado `unloaded`. No modo isolado, a stream publica `direct-complete`, inicia a
referência (`reference-loading`), conclui a comparação e encerra novamente o
processo de referência, preservando o próximo turno compilado sem pressão desse
modelo residente. O verificador PyTorch da política de margem também é lazy
até o primeiro fallback; depois disso seu índice autenticado permanece pronto,
enquanto o kernel pesado é reciclado após cada uso. O relatório distingue
`referenceStartupSeconds`, `referenceComputeSeconds`, `referenceColdStart` e
`referenceReleased`. Como a passagem da referência pode expulsar páginas do
constant pool compilado do cache do sistema, o servidor repagina um prefill e
um decode curtos depois de encerrar a referência; `directRecoverySeconds` mede
essa recuperação, executada depois que o texto compilado já foi entregue.

Na geração residente, o `argmax` e o top-K do vetor final de 262.144 logits são
calculados no MLX/Metal. Em passos não terminais, apenas o token e os candidatos
top-K atravessam para a CPU; o vetor F32 completo (1 MiB) é materializado uma
única vez, no passo terminal, para preservar o SHA-256 auditável. A interface
publica quantos passos foram ranqueados na GPU e quantas transferências de vetor
completo foram evitadas.

O head de vocabulário usa por padrão o peso BF16 autenticado sem quantização,
preservando a política promovida pela calibração atual. A opção Q8 affine,
`group_size=64`, é calculada deterministicamente quando selecionada e oferece
uma representação alternativa da matriz `262144 × 2560`;
`--direct-mlx-head-quantization off|q8|q8-shortlist|q4`
permite comparar os modos, e a UI e o relatório registram
`mlxHeadQuantization`. Q4 permanece apenas experimental: divergiu em uma
decisão sensível. Na calibração Q8 de oito prompts
e oito tokens, o caminho direto reproduziu 64/64 tokens do Transformers, sem
divergência raiz, com 96,56% de sobreposição top-K média e 21,13 tok/s contra
1,91 tok/s do baseline (11,08×). Essa matriz é evidência amostral, não prova de
paridade universal; prompts arbitrários continuam sendo comparados na interface.

O modo opt-in `q8-shortlist` reduz a projeção principal da cabeça a um único
matmul Q8 e recalcula com os pesos BF16 exatos os 16 IDs de maior logit
aproximado antes do argmax e do top-K. Ele é identificado como
`single-stage-affine-q8-shortlist-refined-v1` no relatório e como experimental
na interface: o refinamento é exato dentro da shortlist, mas não constitui um
limite global que prove que nenhum dos outros 262.128 IDs poderia entrar nela.
Por isso o padrão continua `off` e mantém o vetor terminal BF16 integral.

No controle pareado de 8 prompts × 4 tokens desta implementação,
`q8-shortlist` preservou 32/32 decisões e elevou o throughput direto de
`22,9385` para `24,1024 tok/s` (`+5,07%`). A calibração ampliada persistida em
`artifacts/gemma4-three-way-calibration-32x4-q8-shortlist.json` preservou
32/32 prompts, 128/128 tokens e zero divergências raiz. O fast path somou
`5,3305 s`, alcançou `24,0126 tok/s` e foi `20,2494×` mais rápido que o
Transformers medido no mesmo processo. Essa evidência finita não promove a
shortlist a uma garantia para prompts arbitrários.

O prefill compilado consolida também as verificações de finitude. Em vez de
reter uma redução `isfinite/all` depois de cada uma das 42 camadas, ele valida
uma vez o hidden state terminal e todos os caches K/V que formam o estado
residente exportado. O decode incremental conserva a verificação por passo e
examina somente as extensões recém-adicionadas aos caches. No relatório
`artifacts/gemma4-three-way-calibration-32x4-prefill-validity.json`, essa
mudança preservou 32/32 prompts, 128/128 tokens, zero divergências raiz e os 32
SHA-256 terminais bit a bit. O prefill agregado caiu de `2,0980 s` para
`2,0676 s` (`-1,45%`), enquanto o tempo direto total caiu de `5,3305 s` para
`5,3185 s`; o ganho total pequeno é reportado sem tratá-lo como mudança de
ordem de grandeza.

As alternativas aproximadas maiores foram rejeitadas. Q8 também na projeção
FFN `down` atingiu `26,0676 tok/s`, mas preservou somente 127/128 decisões. Q8
na atenção das camadas 21–41 preservou 125/128; mesmo o bloco 37–41 preservou
somente 124/128 no corpus completo. Por isso `down` e atenção permanecem BF16
no caminho publicado, apesar do throughput experimental maior.

Para reproduzir o modo na interface:

```bash
npm run compare:gemma4-real-ui -- \
  --direct-mlx-head-quantization q8-shortlist \
  --direct-verification-margin off
```

O compositor vetorizado também faz eliminação de subexpressões comuns por passo
de geração: fatores seno/cosseno de RoPE e máscaras causais/sliding-window com
a mesma configuração são construídos uma vez e compartilhados entre as camadas
que os consomem. As contagens construídas e evitadas aparecem na UI e nos
relatórios de calibração, permitindo distinguir redução algébrica real de uma
simples alegação de otimização.

Depois do prelude de embedding/PLE, o hot path incremental é capturado como uma
única função `mx.compile(..., shapeless=True)`: executa as 42 camadas, norma
final, projeção das 262.144 dimensões de vocabulário e softcap, e devolve logits
mais o próximo KV cache. Batch e comprimento da query permanecem `1 × 1`,
enquanto o comprimento do cache K/V é dinâmico; a atenção GQA usa a forma agrupada
`[kv_heads, groups, query, key]`, sem materializar `repeat` dos heads K/V. O
campo `compiledIncrementalDecoderSteps` do relatório e da UI conta somente os
passos `decoder → logits` realmente atravessados por esse grafo (o primeiro
token ainda vem do prefill).

A closure compilada permanece residente no worker entre prompts. O campo
`incrementalCompilerCacheHit` prova se a requisição reutilizou essa closure. A
interface executa dois warm-ups: o primeiro compila a função e o segundo força
a especialização estrutural do KV dinâmico, impedindo que a primeira mensagem
do usuário absorva esse custo único.

As tabelas completas de embedding não são capturadas pela closure: fazê-lo
melhora o benchmark isolado, mas aumenta a pressão sobre memória unificada e
degrada o executor direto quando a interface roda Transformers e compatibilidade
em paralelo. O prelude continua fora do grafo até existir um lowering paginado
que não mantenha as tabelas completas como constantes compiladas.

A forma agrupada preserva a função matemática, mas pode alterar a árvore de
redução em ponto flutuante. Por isso a validação de qualidade separa a primeira
decisão divergente das divergências em cascata e reporta empates/margens do
top-2; igualdade do hash terminal não é prometida no modo sem arredondamentos
intermediários. O custo único de compilação pertence ao warm-up e não deve ser
misturado ao throughput aquecido.

O caminho residente mantém ainda um invariante de validação incremental: cada
novo segmento K/V é verificado antes de entrar no cache, portanto os passos
seguintes não percorrem novamente o prefixo já comprovado. Os logits continuam
falhando para `NaN`/`Inf`, mas a mesma varredura não é repetida antes e durante
o ranking. A UI publica ambas as quantidades de validações redundantes evitadas.

O modo padrão, **Conclusão bruta (fiel)**, envia o texto literalmente porque o
`gemma-4-E4B-dense/tokenizer_config.json` possui `chat_template: null`. O modo
**Chat IT (contrato oficial Gemma 4)** aplica, igualmente aos três executores,
`<|turn>user\n...<turn|>\n<|turn>model\n`, para a geração no token EOT 106 e
reutiliza esse terminador sem duplicá-lo ao montar o turno seguinte. Os
delimitadores de controle são rejeitados no texto do usuário para que ele não
altere a estrutura da conversa. A formatação permite testar a mecânica de chat,
mas não transforma os pesos densos base em pesos instruction-tuned; a interface
exibe esse limite em vez de atribuir eventual qualidade textual à compilação.
Trocar o modo inicia uma nova sessão para impedir mistura de contratos de
tokenização.

Além de igualdade do argmax, cada passo agora compara os top logits do runtime
direto com os do Transformers: erro absoluto do logit escolhido, variação da
margem entre primeiro e segundo lugares, erro máximo entre IDs comuns e
sobreposição top-K. A comparação marca até qual passo os contextos ainda eram
iguais; depois da primeira escolha diferente, os próximos tokens são
classificados como cascata e não como novas divergências numéricas.

O hot path MLX quantiza seletivamente em Q8 as matrizes `gate_proj` e
`up_proj` das 42 FFNs (`--direct-mlx-decoder-quantization q8-ffn-gate-up`, o
padrão). Pesos de atenção, `down_proj` e a cabeça final permanecem BF16 exatos
por padrão (`--direct-mlx-head-quantization off`). O modo opcional Q8 usa a
decomposição hierárquica certificada e recalcula localmente a projeção BF16
completa quando o limite não prova o argmax. A seleção é materializada uma vez
ao carregar os pesos residentes e
`mx.quantized_matmul` participa da mesma closure incremental compilada; o
relatório e a UI publicam separadamente `mlxDecoderQuantization`,
`mlxDecoderQuantizationStrategy`, `mlxHeadQuantization` e
`mlxHeadQuantizationStrategy`. O modo padrão declara
`single-stage-affine-q8-calibrated-v1`: ele é uma política selecionada por
calibração diferencial, não uma certificação matemática do decoder. Os modos
mais agressivos
(`q8-ffn`, `q8-ffn-down`, `q8-attention`, `q8-all`) continuam disponíveis para
experimentos, sem alegação de paridade.

As projeções Q8 `gate` e `up` compartilham a mesma entrada e são materializadas
residentemente como uma única matriz de códigos, escalas e biases concatenada
por linhas. Cada FFN executa um `quantized_matmul` e separa matematicamente as
duas metades antes da GELU gated; como os grupos de quantização não cruzam
linhas, cada coeficiente e parâmetro afim permanece inalterado. Relatórios e a
UI identificam esse lowering como `concatenated-affine-q8-v1` em
`mlxDecoderGateUpProjectionStrategy`.

Na calibração ampliada de 32 prompts × 4 tokens, essa projeção preservou
`32/32` prompts, `128/128` tokens e zero divergências raiz. O erro absoluto do
logit escolhido permaneceu em `0.0595703125` na média e `0.25` no máximo. Sob
o mesmo corpus e configuração, o tempo direto agregado caiu de `7.0893 s`
para `6.1974 s` (`-12.58%`) e o throughput subiu de `18.0553` para
`20.6537 tok/s` (`+14.39%`). O decode estacionário continuou próximo de
`36 ms/token`; o ganho concentrou-se no prefill e no primeiro passo, sem mudar
a política Q8 calibrada das 42 camadas.

No decode incremental compilado, as duas metades dessa projeção concatenada
agora são recuperadas por índices estáticos com `mx.take`, em vez de criar dois
seletores F32, multiplicar o tensor inteiro por ambos e reduzir o eixo de duas
entradas. Esse lowering é publicado como `static-index-take-v1` em
`mlxDecoderGateUpSplitStrategy`; ele não altera coeficientes nem executa
aritmética sobre os valores separados. Em três repetições quentes de 16 tokens,
a média estacionária caiu de `35.7471 ms/token` para `35.2534 ms/token`
(`-1.38%`), preservando o hash terminal
`b768fc66f5cae39a4e48100052a94ab99b4525b3f8a27d9365253744e1866a6b`.
A calibração completa posterior preservou novamente `32/32` prompts,
`128/128` tokens e zero divergências raiz; nessa execução, o runtime direto
somou `7.2801 s` (`17.5821 tok/s`) e foi `19.4390×` mais rápido que a
referência original medida no mesmo processo.

Para localizar quais camadas participam dessa compensação numérica, o mesmo
modo aceita `--direct-mlx-decoder-quantization-layers` com índices e faixas
inclusivas, por exemplo `0-19,21-41`. A especificação é normalizada, validada
contra as 42 camadas e encaminhada até o worker Metal; o relatório e a UI
publicam `mlxDecoderQuantizationLayers`. Uma máscara explícita recebe a
estratégia `experimental-affine-q8` até passar pelo corpus completo, enquanto
o padrão sem máscara continua significando todas as camadas e mantém
`single-stage-affine-q8-calibrated-v1`.

A ablação de sensibilidade confirmou que a compensação atravessa a pilha:
usar somente `0-20` ou somente `21-41` divergiu em pelo menos um dos dois
prompts-limite; retirar individualmente as camadas 0, 1, 21 ou 41 também
divergiu. Retirar apenas a camada 20 preservou esses dois casos, mas falhou no
corpus 32×4: `31/32` prompts, `124/128` tokens e uma divergência raiz no passo
zero de `The first month of the year is` (token de referência `1024`, token da
máscara `496`). Apesar de elevar o throughput medido de `18.06` para
`21.75 tok/s`, a máscara não foi promovida porque velocidade sem paridade não
atende ao contrato. O padrão permanece nas 42 FFNs.

Toda tentativa de otimização pode ser promovida pelo gate reprodutível
`npm run promote:gemma4-calibration --`. Ele recebe dois relatórios completos,
por exemplo `--baseline baseline.json --candidate candidate.json
--min-throughput-gain-percent 2 --output promotion.json`, e falha com código
não zero se qualquer condição não for satisfeita. O gate exige o mesmo corpus,
zero divergências raiz, paridade de todos os tokens contra a referência,
identidade dos tokens compilados e do SHA-256 do vetor terminal de logits por
prompt, além do ganho mínimo configurado. Relatórios de calibração agora
preservam `terminalLogitsSha256`; artefatos antigos ou incompletos são
rejeitados em vez de serem promovidos sem identidade numérica suficiente.

O perfil da geração residente separa `prefillSeconds`,
`incrementalDecoderSeconds`, `tokenSelectionSeconds` e
`terminalLogitTransferSeconds`. Esses quatro tempos atravessam o protocolo
binário, o relatório JSON, a calibração e a interface, permitindo distinguir o
custo do grafo decoder do ranking e da transferência final. Na medição sensível
de 3 prompts × 4 tokens que validou o contrato, o caminho direto manteve 12/12
tokens, zero divergências raiz e nenhum fallback; o decoder incremental somou
aproximadamente `0.107 s` por prompt, a seleção `0.0022–0.0026 s` e a
transferência terminal menos de `0.000003 s`. Assim, novas otimizações devem
atacar o forward incremental, não a transferência ou o ranking top-k.

O worker MLX também aceita o modo experimental
`--direct-mlx-decoder-quantization q4-ffn-gate-up`, opcionalmente limitado por
`--direct-mlx-decoder-quantization-layers`. As camadas selecionadas usam Q4 em
`gate/up`; as demais continuam Q8, e o relatório identifica o caminho como
`experimental-affine-q4`/`concatenated-affine-q4-v1`. Ele não é o padrão: Q4
nas 42 camadas elevou o caminho sensível de `22,73` para `24,04 tok/s`, mas
causou uma divergência raiz em 12 decisões. No corpus de 8 prompts × 4 tokens,
as faixas `0–20` e `0–9` causaram, respectivamente, 4 e 3 divergências raiz.
O fallback PyTorch recuperou 12/12 tokens no corpus sensível, porém reduziu o
throughput híbrido a `4,96 tok/s`. O runtime promovido permanece Q8 até existir
uma seleção ou certificação que preserve a paridade em calibração ampliada.

O prelude textual incremental preserva a redução CPU BF16 declarada e, por
isso, não é reassociado artificialmente dentro do grafo Metal. Como suas duas
saídas dependem apenas do token e dos pesos imutáveis, o worker mantém um LRU
exato de 1.024 tokens com o embedding oculto e as entradas PLE já calculadas.
O prefill alimenta esse cache e o decode reutiliza a entrada sem novo produto
matricial ou RMSNorm. O protocolo publica `tokenPreludeCacheHits`,
`tokenPreludeCacheMisses`, `tokenPreludeSeconds` e
`compiledDecoderGraphSeconds`, exigindo que hits mais misses coincidam com os
passos incrementais compilados. Em duas execuções consecutivas do mesmo prompt
com 4 tokens, a primeira já reutilizou um token do prefill (`1 hit / 2 misses`)
e consumiu `0.00420 s` de prelude; a segunda registrou `3 hits / 0 misses` e
`0.0000066 s`. O tempo incremental total caiu de `0.10668 s` para `0.10014 s`,
mantendo 8/8 tokens e zero divergências raiz.

Na calibração oficial de 8 prompts × 8 tokens, head BF16 mais `gate+up` Q8
produziu 64/64 tokens iguais ao Transformers, zero divergências raiz, erro
máximo de `0.25` no logit escolhido e sobreposição top-5 média de `95.625%`.
O executor direto alcançou `24.09 tok/s` contra `1.83 tok/s` da referência,
razão de throughput `13.20×`, com o verificador seletivo desabilitado para medir
somente o fast path. Esse corpus é evidência finita, não uma promessa de
igualdade para todo prompt.

O runtime MLX também aplica uma seleção híbrida independente da resposta
original. Por padrão, `--direct-verification-margin 0` identifica qualquer
empate no top-2 do próprio Metal. O verificador PyTorch sempre recompõe o
decoder e o cache na mesma ordem incremental, mas só calcula a cabeça de
262.144 logits nos passos sensíveis. Passos com margem segura reutilizam o
token Metal; depois da primeira divergência confirmada, todas as cabeças
seguintes são recalculadas porque o contexto mudou. Margens ou hashes ausentes
mantêm o fallback integral, sem aceitar uma verificação parcial. Tokens Metal
emitidos durante streaming são marcados como provisórios; o relatório final
declara `selectedBackend`, `fallbackTriggered`, a margem observada e os tempos
separados de Metal, verificação e execução híbrida. Margens ausentes ou
malformadas falham de modo fechado e também acionam a verificação. O limiar é
configurável e `--direct-verification-margin off` desabilita o segundo backend.
Essa política reduz empates dependentes da árvore de redução, mas continua
sendo medida contra o Transformers: um segundo backend BF16 pode preservar o
mesmo empate e, portanto, não é apresentado como prova geral de paridade.

Depois do primeiro fallback, o servidor mantém o índice autenticado do
artefato literal já aberto, mas reinicia o kernel linear PyTorch após cada
verificação. Isso evita reler e reindexar os 20 GiB de `constants.literal.json`
nos empates seguintes e, ao mesmo tempo, libera as páginas e buffers pesados
do cálculo anterior. `/api/status` identifica esse ciclo como
`retained-index-restarted-kernel-v1`; o primeiro empate ainda inclui a
inicialização fria, enquanto `verificationSeconds` mede explicitamente o custo
recorrente dos próximos empates.

O corpus ampliado e reproduzível pode ser executado com:

```bash
npm run calibrate:gemma4-real -- \
  --output ./artifacts/gemma4-q8-residual-head-calibration.json \
  --prompts-json ./artifacts/gemma4-calibration-prompts-32.json \
  --tokens 4 --request-threads 1 \
  --direct-mlx-head-quantization q8 \
  --direct-mlx-decoder-quantization q8-ffn-gate-up \
  --direct-verification-margin off \
  --precision f32 --rounding-policy none
```

Ele cobre 32 prompts em inglês, português e espanhol, completions factuais,
matemática, código, Unicode e repetição, totalizando 128 tokens gerados.
No relatório seletivo atual, 3/32 prompts acionaram a verificação. O resultado
selecionado coincidiu com o baseline em 128/128 decisões e 32/32 prompts, sem
divergências raiz. Os fallbacks calcularam 5 cabeças de vocabulário em vez de
12: 1 no caso sem divergência, 3 após uma divergência no passo 1 e 1 numa
divergência terminal. O throughput agregado selecionado foi 2,8396 tokens/s,
2,8698× o baseline de 0,9895 token/s na mesma execução. Essa calibração é
evidência sobre o corpus versionado, não garantia universal para qualquer
prompt.

O relatório versionado
`artifacts/gemma4-three-way-calibration-32x4-self-contained-bundle.json`
confirma que `source`, `literalArtifact` e `binaryPool` apontaram para o bundle
compilado durante 32 prompts × 4 tokens. O runtime selecionado preservou
`32/32` prompts e `128/128` tokens, sem divergência raiz. O
`model.safetensors` empacotado e a fonte original possuem o mesmo SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.

O `lm_head` Q8 possui um caminho autocontido de decisão certificada. Na carga,
o worker decompõe cada peso BF16 como `W = Q8₀(W) + Q8₁(W-Q8₀(W)) + ε`.
O forward soma as duas projeções Q8 e, para cada token, limita apenas o produto
escalar do resíduo restante `ε` pelo menor dos limites de Hölder
`||x_g||₁ ||ε_g||∞` e Cauchy-Schwarz
`||x_g||₂ ||ε_g||₂`, somados entre os grupos. Como `tanh`, o softcap e o
arredondamento BF16 são monotônicos, o limite superior é propagado até o logit
final. O argmax hierárquico só é aceito quando o logit BF16 exato do candidato é
estritamente maior que todos esses limites superiores; caso contrário, o
próprio worker calcula o `lm_head` BF16 completo, sem consultar o Transformers
ou outro modelo-oráculo. A interface expõe as duas contagens como
`quantizedHeadCertifiedSteps` e `quantizedHeadExactFallbackSteps`, além de
declarar `two-stage-residual-affine-certified-v1` como estratégia.

Na calibração local de 21 de julho de 2026 com os 32 prompts versionados × 4
tokens, Q8 em `gate+up` e no `lm_head`, 125/128 decisões foram certificadas e
3/128 usaram o fallback BF16 local. O resultado preservou 32/32 prompts e
128/128 tokens, com zero divergências raiz. O caminho direto somou 7,0893 s,
ou 18,0553 tokens/s, contra 1,0362 tokens/s do original na mesma execução
(17,4242×). Contra a calibração imediatamente anterior da cabeça Q8 em um
único estágio, os fallbacks caíram de 30 para 3 e o throughput direto subiu de
17,3645 para 18,0553 tokens/s. Isso é evidência para esse corpus, não prova
universal para prompts arbitrários; a interface continua executando e
comparando ambos para cada entrada solicitada.

A calibração seguinte manteve Q8 apenas em `gate+up` e restaurou o `lm_head`
BF16 exato. Ela preservou novamente `32/32` prompts, `128/128` tokens e zero
divergências raiz, elevou a sobreposição top-K média de `96,5625%` para
`96,71875%` e eliminou toda a lógica de certificado/fallback da cabeça. O tempo
direto agregado caiu de `7,2801 s` (`17,5821 tok/s`) para `5,6213 s`
(`22,7704 tok/s`) nas calibrações consecutivas, e o microbenchmark quente de
16 tokens caiu de `35,3246 ms/token` para `34,6816 ms/token` (`-1,82%`). Por
ser simultaneamente mais exato, mais simples e mais rápido nessa máquina, o
head BF16 passou a ser o padrão da interface e da calibração; Q8 e Q4 continuam
disponíveis apenas por flag para experimentos reproduzíveis.

A materialização final agora grava também `final-formulas.json` com exatamente
262.144 propriedades no formato solicitado:

```json
{
  "calc_final_0": "BF16_RNE(EVAL_EXACT_DAG(\"sha256:...\", x))",
  "calc_final_1": "BF16_RNE(EVAL_EXACT_DAG(\"sha256:...\", x))"
}
```

Cada raiz referencia a DAG matemática autenticada no próprio bundle; `x` é a
única variável livre. O arquivo é produzido em streaming, vinculado ao SHA-256
de `global-formulas.ssa.json` e incluído no manifesto com contagem, tamanho e
SHA-256 próprios. O manifesto preserva também `orderedRootsSha256`, calculado
sobre cada par ordenado `dimensão:raiz`. No modo compilado real, o servidor
entrega esses compromissos ao worker, que valida as 262.144 chaves, a forma de
cada `F_n(x)` e a ordem das raízes antes do warm-up. O worker publica
`finalFormulaProgram.execution = vectorized-shared-dag-output-program` no
status e em cada geração; a interface não fica pronta se esse vínculo divergir.
Para adicionar ou regenerar o mapa em um bundle existente:

```bash
npm run generate:gemma4-final-formula-map -- \
  --bundle artifacts/gemma4-compiled-global-runtime-bundle
```

A interface publica uma função final concreta do bundle no formato
`calc_final_n(x) = BF16_RNE(EVAL_EXACT_DAG(root_n, x))`. O inspetor aceita
qualquer dimensão `n` de `0` a `262143`, lê somente seu binding no
`global-formulas.ssa.json`, valida família, quantização e raiz, e decodifica o
mesmo `n` como token do vocabulário. Assim é possível navegar de uma dimensão
matemática até o texto que ela seleciona sem carregar os 411 MiB do SSA na
memória ou no navegador. A dimensão `0` continua exibida imediatamente com
quantidade de nós, comprimento de `x`, arquivo `formula.graph.json` e seu
SHA-256 vindos do manifesto validado no startup. O grafo fechado possui somente
`x[i]` como entrada variável; pesos e constantes são números fixos do
`constants.literal.json`. A forma DAG é a representação executável da função
objetiva. A expansão textual plana da mesma função é evitada na UI porque
duplicaria subexpressões compartilhadas em escala astronômica, sem reduzir o
cálculo que o Metal efetivamente executa.

A escolha Q8 do decoder também foi comparada diretamente com `gate/up` BF16 e
com uma decomposição experimental de dois estágios
`W = Q8₀(W) + Q8₁(W-Q8₀(W))`. No mesmo corpus de 32 prompts × 4 tokens, o
decoder BF16 preservou apenas 30/32 prompts e 124/128 tokens, com divergências
raiz nos passos 1 de `Translate to Portuguese: Good morning` e 3 de
`Traduza para inglês: boa noite`; seu throughput agregado foi 17,9001
tokens/s. O Q8 simples preservou 32/32 e 128/128 a 18,0553 tokens/s. O modo
residual repetiu as duas divergências BF16 nos prompts sensíveis e foi mais
lento no smoke (`1,5611 s` contra `1,3470 s` do Q8 simples), portanto foi
removido em vez de ser publicado como uma opção aparentemente mais fiel. Isso
mostra que, no contrato real sem arredondamentos BF16 intermediários, aproximar
isoladamente cada matriz do peso BF16 não garante maior proximidade do argmax
Transformers; a política precisa continuar sendo validada ponta a ponta.

### Gerador de fórmulas fisicamente planas

`generateGemma4FlatFormulaObject` substitui recursivamente funções de operação,
desenrola reduções fixas, resolve pesos alcançados como literais e aceita
somente `x[...]` como variável livre. O resultado possui exatamente o formato:

```json
{
  "calc_final_0": "(((-1/4) * x[0]) + (-1.125 * x[1]))"
}
```

O CLI aplica o mesmo contrato ao artefato real:

```bash
npm run generate:gemma4-flat-formulas -- \
  --family terminal_logit --dimension 0 \
  --batch 0 --sequence 5 \
  --max-characters 1000000000 \
  --output /tmp/gemma4-flat-0.json
```

Ele falha antes de escrever JSON parcial quando a expansão sintática direta não
cabe no limite. A estimativa original de `1,160623467963383e247` caracteres
para `terminal_logit[0]` não considera composição algébrica incremental e não é
um limite inferior da fórmula já simplificada. O programa também rejeita
qualquer closure que ainda deixe PLE, máscara, posição, cache, peso ou nome de
camada como variável livre.

### Composição reversa com simplificação incremental

O compositor paramétrico parte de um output fixo, expande a função mais próxima
da saída e reconstrói o grafo pelo simplificador canônico antes de avançar:

```bash
npm run compose:gemma4-reverse -- \
  --family terminal_logit --dimension 0 \
  --batch 0 --sequence 5 --steps 8 \
  --output /tmp/gemma4-reverse-logit-0-step8.json
```

O simplificador conhece racionalização, combinação de termos e identidades
exatas como `sin(x)/cos(x)=tan(x)`, `sin(x)^2+cos(x)^2=1`, paridade de
seno/cosseno/tangente/tanh e `sqrt(x^2)=abs(x)`. Regras dependentes de domínio
não são aplicadas sem prova; `x/x` não é cancelado se `x` puder ser zero.

# Gemma 4 E4B: pipeline escalar explícito dos scores de áudio — 2026-07-18

## Classificação

Esta sessão aceitou independentemente a evidência candidata anterior de imagem
(`294/294` em tolerância zero) e repetiu a fronteira de áudio anterior
(`361/372`, com `audio_features` exato). Em seguida, removeu uma operação opaca
do produto: cada uma das 12 atenções de áudio agora declara separadamente AC,
BD não deslocado, relative shift, soma de logits, softcap, máscara, softmax,
redução de valores e cast BF16 do contexto.

O checkpoint continua proibido. Os dois matmuls F32 de score e o matmul F32 de
valores permanecem `runtime-defined`; nenhuma agenda escalar foi inventada. A
comparação completa continua `approximate`, embora a saída terminal registrada
seja bitwise exata.

## Autoridade e limite nativo

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- pacote: Safetensors BF16 denso, sem quantização;
- runtime: `transformers-5.5.0/torch-2.12.1`, CPU eager;
- build PyTorch: `7269437d655783a26cba32aa88195b741ff496aa`;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`.

A revisão fixada do PyTorch direciona BMMs maiores que o limiar escalar para
`addmm_impl_cpu_`; no build macOS sem MKL, F32 chega ao BLAS `sgemm`. O PyTorch
declara essa chamada, mas não a árvore de arredondamento interna do Apple
Accelerate. Portanto a evidência autoritativa sustenta um dispatch geral por
classe para `pytorch-cpu-f32-matmul/runtime-defined`, não uma soma ascendente ou
FMA fictícia. Referências fixadas:

- `aten/src/ATen/native/LinearAlgebra.cpp`, `bmm_out_or_baddbmm_` e
  `addmm_impl_cpu_`:
  <https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/LinearAlgebra.cpp>;
- `aten/src/ATen/native/CPUBlas.cpp`, dispatch F32 para `sgemm`:
  <https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/CPUBlas.cpp>.

## Programa de cálculo

O programa de áudio cresceu de 559 para 619 atribuições. Em cada camada, a
dependência é agora literal:

```text
q_scaled,k_scaled -> attention_ac
q_scaled,relative_keys -> attention_bd_unshifted
attention_bd_unshifted -> attention_bd
attention_ac,attention_bd -> attention_logits
attention_logits -> attention_softcapped
attention_softcapped,audio_output_mask -> attention_scores
attention_scores -> attention_weights
attention_weights,v -> attention_context -> attention_context_bf16
```

Somente `chunked-attention-content-matmul`,
`relative-attention-position-matmul` e
`chunked-relative-attention-values` falham fechados na vista escalar. Relative
shift declara o pad/flatten/slice/reshape por coordenada; soma, softcap e máscara
declaram F32, SLEEF, índices de query/key, horizonte causal e o literal
`-1000000000`. O binding da navegação usa substituição simultânea, evitando que
nomes sobrepostos como `attention_bd` e `attention_bd_unshifted` dupliquem o
prefixo da instância composite.

## Artefato e payload

Regeneração:

```bash
npm run build
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop52-final-audit.json \
  --verify-gemma4-payloads
```

- bytes do artefato: `21,328,161,289`;
- SHA-256: `be4b66971fd584f66803ae57cfedc737293220071dbec27be65f7a17af657e34`;
- constantes: `2,130`;
- payload comparado: `15,992,314,836` bytes;
- SHA-256 agregado da origem e do literal:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- caminho de origem serializado: `false`.

Com `gemma-4-E4B-dense` renomeado, `--verify-payloads
--assert-source-unavailable` revalidou os mesmos 2.130 payloads e o mesmo digest
somente a partir do JSON. O relatório tem SHA-256
`e20b3229dcdfbb2188c8c213a8f838760fbee1be1964c2a9b96baa53519e15ed`.

## Diferencial de 432 fronteiras

A captura instrumentada executa um forward não instrumentado e rejeita qualquer
alteração bit a bit em `pooler_output` ou `attention_mask`. A entrada foi
`input_features [1,1,128]`, de `-0.25` a `0.25`, com máscara verdadeira.

```bash
npm run capture:gemma4-audio-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop52-audio-explicit-score-stages.json \
  --frames 1 --feature-start=-0.25 --feature-end=0.25 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

O trace tem SHA-256
`8f7b7400bfd7d9bc0cbb38dfed6c467169ea327b5c4814092bfc60dfa62feb04`.
Com a fonte indisponível, o replay produziu:

- `432` operações comparadas, `362` exatas em tolerância zero;
- primeira divergência: `audio_layer_0_attention_position_scores`;
- AC: `1/12` exato, erro absoluto máximo `7.62939453125e-6`;
- BD não deslocado: `0/12` exato, erro absoluto máximo
  `4.57763671875e-5`;
- relative shift, soma, softcap e máscara propagam essas diferenças no replay
  end-to-end; não são classificados artificialmente como exatos;
- todos os softmax, contextos BF16, consumidores posteriores e
  `audio_features [1,2560]` permanecem exatos;
- saída terminal: erro absoluto `0`, relativo `0`, cosseno `1`, top-k overlap
  `1` e argmax agreement verdadeiro;
- relatório SHA-256:
  `ce80c4d75d1347f052341b787a3f5b758b8d1f6851bad6c70712bf8d897bcff1`.

Para separar propagação de defeito local, o mesmo comparador ancora somente os
dois resultados nativos AC/BD do trace e executa o programa literal a partir
deles. As `48/48` fronteiras de relative shift, soma, softcap e máscara passam
bitwise, com erros absoluto e relativo zero. Esse modo não substitui os
matmuls no artefato nem melhora a classificação global; ele prova apenas as
quatro classes downstream agora explicitadas.

A regressão image com a fonte ausente repetiu `294/294` fronteiras em tolerância
zero (`lossless-within-dtype`), relatório SHA-256
`db409a3a5963a808959060473e88cedb26d570146255eb3545514785d44d0451`.

## Limite preservado

- 36 atribuições de áudio continuam `runtime-defined`: 12 AC, 12 BD e 12
  reduções de valores;
- a captura de um frame observa diretamente AC/BD, mas o contexto F32 é
  observado apenas no cast BF16 consumidor;
- vídeo ainda contém duas divergências internas de batched matmul;
- a evidência cobre uma entrada determinística de áudio, não geração composite
  ampla com áudio/vídeo;
- `.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

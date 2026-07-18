# Gemma 4 E4B: programas unários incorporados — 2026-07-18

## Fronteira fechada

A revisão independente do artefato schema v18 confirmou os quatro programas
SLEEF incorporados, mas encontrou uma dependência de runtime ainda ampla:
atribuições armazenadas e vistas escalares chamavam `log1p`, `sqrt`, `rsqrt`,
`exp` ou `tanh` como funções host. O schema v19 elimina essa classe em todas as
atribuições compatíveis de texto, visão e áudio, sem despachar por layer id.

O artefato agora contém sete programas F32 finitos:

- `SLEEF_EXP_F32`, `SLEEF_SIN_F32`, `SLEEF_COS_F32` e `SLEEF_TANH_F32`;
- `SLEEF_LOG1P_F32`, incluindo o ramo `SLEEF_LOG_F32` para entradas grandes;
- `ARM_SQRT_F32`, especificado por busca de bits binary32 e desempate ties-to-even;
- `PYTORCH_POW_NEGATIVE_HALF_F32`, com sqrt seguido de divisão F32 explícita.

O dispatch cobre softmax, GELU/softcap, RoPE, softplus aprendido, Welford e
RMSNorm. A linguagem de fórmulas subiu para schema 5 e tanto o programa quanto
as vistas escalares rejeitam fail-closed qualquer chamada crua a
`exp`, `tanh`, `log`, `log1p`, `sqrt`, `rsqrt`, `sin` ou `cos`.

## Autoridade e comparação numérica

A implementação foi transcrita do PyTorch 2.12.1 commit
`7269437d655783a26cba32aa88195b741ff496aa` e do SLEEF incorporado commit
`5a1d179df9cf652951b59010a2d2075372d67f68`. Uma comparação determinística
contra o runtime instalado executou 2.048 inputs F32 para cada classe:

```text
bitwise comparisons: log1p=2048, sqrt=2048, pow(-0.5)=2048, mismatches=0
```

Os testes também exercitam valores representativos, os bits de todas as
constantes incorporadas, alteração do programa serializado, intrínseco SLEEF
desconhecido e chamada host opaca.

## Artefato real

Fonte imutável: `google/gemma-4-E4B`, revisão
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
quantizado. Comando de geração:

```bash
npm run build
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Resultado:

- schema: `19`;
- bytes: `21,384,531,002`;
- SHA-256: `2d33744294ebb7d2318ed7e3760c3652e2dc910bdc1e70b9b7a6d1654b1580db`;
- constantes/decoders: `2,130/2,130`;
- bytes aprendidos incorporados: `15,992,314,836`;
- atribuições forward instanciadas: `2,709`.

## Evidência source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente para fora do caminho
declarado sob um restore trap durante cada inspeção e replay. A verificação de
payloads cobriu as 2.130 constantes e todos os 15.992.314.836 bytes, registrou
`sourceCheckpointAccessed=false` e preservou o SHA-256 concatenado do storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
Relatório: `/private/tmp/llm-inner-loop75-v19-payloads.json`, SHA-256
`2776ab7c723135e120604eec4514e6bdbcb155ab4bdc4d153a4ed6cb7ad1c8fd`.

Três vistas escalares source-removed demonstram as classes novas e seus
operandos reais:

- RMS de visão usa `ARM_SQRT_F32`; relatório
  `/private/tmp/llm-inner-loop75-v19-vision-rms.json`, SHA-256
  `0f5498ee2f411defa5a07251f52c2ead1769178d623464605a368b6430d75fe6`;
- Q-scale de áudio substitui o peso BF16 real `-2.0625` dentro de
  `SLEEF_LOG1P_F32(SLEEF_EXP_F32(-2.0625))`; relatório
  `/private/tmp/llm-inner-loop75-v19-audio-softplus.json`, SHA-256
  `8b2b664e49adcd4287046a7bf137604a21f96414133b65cd5c1951677e4393ea`;
- softmax de visão usa `SLEEF_EXP_F32`; relatório
  `/private/tmp/llm-inner-loop75-v19-vision-softmax.json`, SHA-256
  `e6c5bfb6171f9a87d0e82c69d7f6f8db28ac936a893640b41f06dafb439bc22e`.

A vista end-to-end source-removed contém as 2.709 atribuições forward, 12
atribuições greedy para um token, 2.076 constantes alcançáveis e 54 constantes
KV locais declaradas runtime-unreachable. Nas fórmulas forward aparecem 147
usos de exp, 131 de sin, 131 de cos, 129 de tanh, 12 de log1p, 2 de sqrt e 637
de pow(-0.5), todos pelos sete programas incorporados; chamadas unárias host:
zero. Relatório `/private/tmp/llm-inner-loop75-v19-end-to-end.json`, SHA-256
`e503e90ea21bdeaede177aef12569077e09dedde98d88df4678160caaefdc26a`.

O replay source-removed do trace composite de áudio autoritativo, com tolerância
absoluta e relativa zero, permaneceu `lossless-within-dtype` no prefill e na
geração. O token 184 na posição 2, logits e 24 caches KV passaram sem primeira
divergência. Relatório `/private/tmp/llm-inner-loop75-v19-composite-audio.json`,
SHA-256 `7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Validação e limite preservado

`npm run typecheck` passou. `npm test` passou 234 testes, zero falhas.

Esta é uma alegação candidata de fechamento da classe de matemática unária,
não do checkpoint Gemma 4. A vista end-to-end permanece
`fail-closed-runtime-reduction`: 64 BMMs de visão e 36 BMMs de áudio ainda não
possuem árvore escalar publicada pelo Apple Accelerate. O diretório
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

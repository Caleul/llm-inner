# Gemma 4 E4B: programas explícitos de indexação e RoPE — 2026-07-18

## Fronteira fechada

A revisão independente do schema v20 confirmou os estágios explícitos de
softmax. A inspeção de todas as 2.297 fórmulas forward, porém, encontrou uma
classe distinta de semântica escondida: as 66 atribuições RoPE de texto usavam
`theta_power` e `rotate_rotate_half`, enquanto máscaras e substituições
multimodais usavam `contiguous_group_id` e `is_declared_modal_token`.

O schema v21 resolve a classe inteira por operação e contrato, sem dispatch por
layer id:

- as 66 fórmulas RoPE escrevem explicitamente o par e a feature pareada, o
  limite de pares ativos, o expoente theta, o fator proporcional, os casts
  BF16/F32 e os ramos de soma/subtração de `rotate_half`;
- as 11 instâncias proporcionais incorporam
  `partial_rotary_factor=0.25`; as outras 55 usam o domínio rotativo completo;
- `formulaLanguage.indexing.programs.contiguousVisionGroupId` incorpora o
  algoritmo ordenado que numera runs vision `{1,2}` e retorna `-1` fora delas;
- a máscara sliding compara diretamente os IDs não negativos de query e key,
  sem o predicado simbólico `same_nonnegative_vision_block`;
- posição 2D explicita a sentinela `[-1,-1]`, os dois índices da tabela e o
  zero tipado; substituição multimodal compara diretamente os IDs declarados;
- o fallback F32 de LayerNorm de áudio explicita as reduções ordenadas de média
  e variância;
- o leitor valida todas as fórmulas forward e de geração contra um vocabulário
  finito de funções. Helper não registrado, programa de indexação alterado ou
  coordenada fora do domínio falha fechado.

Essa fronteira avançou a navegabilidade do produto sem repetir probes dos 100
BMMs nativos. A árvore escalar desses BMMs continua indisponível como uma única
classe Apple Accelerate e não foi aproximada por atribuição ou layer.

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

- schema do artefato: `21`; schema da linguagem de fórmulas: `7`;
- bytes: `21,384,800,786`;
- SHA-256: `1392e35da990710e65dde3117a000e40d0bc1dd101cd4815424ad89909234952`;
- constantes/decoders: `2,130/2,130`;
- bytes aprendidos incorporados: `15,992,314,836`;
- atribuições forward instanciadas: `2,709`;
- definições escalares: `2,297`, com zero helper opaco.

A auditoria source-present verificou os 2.130 payloads contra os seis arquivos
da fonte. O SHA-256 concatenado de storage nos dois lados foi
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
Relatório `/private/tmp/llm-inner-loop77-v21-source-audit.json`, SHA-256
`73c2073a853ac2122ecc728a8c7fb879edd4ca4f34dfa2480545876a1caf9c5a`.

## Evidência source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente para fora do caminho
declarado sob restore trap durante cada inspeção e replay. A vista end-to-end
verificou todos os payloads e bytes, registrou
`sourceCheckpointAccessed=false`, preservou o SHA-256 de storage acima e
materializou 2.709 operações forward e 12 operações de geração. Relatório
`/private/tmp/llm-inner-loop77-v21-source-removed.json`, SHA-256
`8af5b60a8badbbd5cd6abb98e105145c3c6c31dffd32e5e6bbb7997d73326040`.

A abertura source-removed do programa confirmou schema `21`, linguagem `7`,
2.297 definições forward e 12 de geração, zero helper opaco e a cobertura
integral das 66 instâncias RoPE (`11` proporcionais e `55` completas). Essa
estrutura está incorporada no mesmo relatório end-to-end acima.

O replay do trace composite de áudio autoritativo, ainda sem a fonte, usou
tolerâncias absoluta e relativa zero. Prefill e geração permaneceram
`lossless-within-dtype`, sem primeira divergência; token 184 na posição 2,
logits e os 24 caches KV passaram, com erro absoluto e relativo máximos zero,
similaridade cosseno e top-k iguais a 1 e argmax concordante. Relatório
`/private/tmp/llm-inner-loop77-v21-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Validação e limite preservado

`npm run typecheck` passou. `npm test` passou 234 testes, zero falhas. O teste
de contrato executa o programa de grupos sobre runs alternadas, recusa o
programa serializado alterado, recusa coordenadas inválidas e injeta um helper
opaco para provar a fronteira fail-closed.

Esta é uma alegação candidata de fechamento dos helpers de indexação e RoPE,
não do checkpoint Gemma 4. A vista end-to-end permanece
`fail-closed-runtime-reduction`: 64 BMMs de visão e 36 BMMs de áudio ainda não
possuem árvore escalar publicada pelo Apple Accelerate. O diretório
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

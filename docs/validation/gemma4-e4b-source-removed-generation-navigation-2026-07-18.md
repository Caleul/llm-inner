# Gemma 4 E4B: navegação concreta da geração sem fonte — 2026-07-18

## Resultado

Esta sessão revisou independentemente a evidência composite do commit
`050f5bd`. Com `gemma-4-E4B-dense` temporariamente ausente, os três relatórios
foram regenerados pelo artefato literal de 21,328,161,289 bytes. Imagem e vídeo
continuam `lossless-within-dtype` em prefill e geração; áudio continua
`approximate`. Os relatórios reproduziram byte a byte os SHA-256 anteriores:

- imagem: `b784ecd3bfdac1e18a8f921b41b342a02936eceef5ac4fcb673996b7ffc71c4e`;
- vídeo: `584f3ddee2aa004e0f4cf3383dbaa9f0ccb01d29a4dac255e2f473bf907ca48d`;
- áudio: `7d7f5027f7f5db49128a0183bacbdeddab91f9d24eb7f72036eff601ceae21e4`.

Essa revisão aceita a evidência candidata de imagem/vídeo e confirma que a
divergência de áudio não era estado obsoleto. Ela não aceita o checkpoint: os
36 matmuls F32 nativos de áudio e os batched matmuls internos de visão/vídeo
continuam sem agenda escalar geral comprovada.

## Fronteira de produto fechada

O programa já serializava doze templates da máquina de estados greedy, mas o
leitor não os tornava navegáveis para um `max_new_tokens` concreto. A nova
fronteira `src/gemma4-literal-generation-navigation.ts`:

- materializa as oito operações de controle para cada passo solicitado, além
  de prefill, posição inicial e dois estados terminais;
- substitui todas as referências `step`, `step+1`, `step-1` e ranges de tokens
  por nomes concretos;
- resolve produtores, consumidores, ordem anterior/seguinte e a condição de
  execução de cada passo após EOS;
- liga prefill e cada forward incremental a uma única expansão compartilhada
  das operações do programa composite, evitando uma invocação de decoder
  opaca e evitando duplicar essa expansão em cada passo; e
- renderiza fórmulas exatas para seleção de logits, argmax ascendente com
  desempate pelo menor ID, append, posição, entradas incrementais, snapshot KV,
  EOS e seleção do estado terminal.

No E4B real, uma navegação de dois passos contém 20 operações de controle e
uma expansão compartilhada de 2,709 operações, iniciando em
`composite_placeholder_masks` e terminando em `final_logit_softcap`. O argmax
expõe `candidate[v]` para `v=0..262143`, recusa valores não finitos e usa a
recorrência ascendente cujo empate mantém o token anterior, portanto o menor
ID.

## Reprodução sem checkpoint

Com `gemma-4-E4B-dense` renomeado e verificado como inexistente:

```bash
npm run build

node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-generation-operations --generation-max-new-tokens 2 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop55-generation-navigation.json

node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --generation-operation 'generation_argmax[0]' \
  --generation-max-new-tokens 2 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop55-generation-argmax.json
```

Os dois resultados declaram `sourceCheckpointAccessed=false`. Seus SHA-256
são, respectivamente,
`196871411052be5ff66ef77e0ed633077b90647e1d0a32976368999221a65bae`
e `516c7786a83fee2f02b8f506195f705d3e8f097d770beefac606663a2e465a51`.

## Validação local

```text
npm run typecheck  -> exit 0
npm test           -> 227 pass, 0 fail
git diff --check   -> exit 0
```

O limite restante é numérico, não de navegação: a vista agora chega a cada
transição de geração e a cada operação do forward declarado, mas preserva a
falha fechada nas classes de matmul nativo sem agenda literal comprovada.

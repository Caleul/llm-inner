# Gemma 4 E4B: clausura explícita de operandos — 2026-07-18

## Fronteira candidata

A revisão independente começou no commit
`d1155fdfbc9002fb1391dacd3d1fc704e6addc0b`, inspecionou o diff e o handoff do
schema v24 e repetiu `npm run typecheck && npm test`: os 234 testes passaram.
Isso aceita os programas KV do loop anterior dentro do seu escopo; não aceita
o checkpoint Gemma 4.

A inspeção do artefato v24 encontrou fórmulas canônicas que ainda exigiam uma
inferência externa. RMSNorm e ativações usavam `input`/`x`; lineares usavam
`...`; BMM vision usava `q`, `k`, `probability` e `value`; Conv2d usava
`padded_input`; e as 32 definições RoPE 2-D vision ainda continham `+/-` e
"sign/order follows rotate_half". As vistas concretas já resolviam parte desses
nomes, mas o programa de definições que governa o grafo não era operand-closed.

O schema v25 e a linguagem de fórmulas schema 11 aplicam um único contrato por
classe de operação:

- cada leitura nomeia um `orderedInput` e uma coordenada completa;
- lineares e clipped-linears substituem a elipse pelo prefixo de saída seguido
  de `input_feature`;
- os 525 RMSNorms de definição nomeiam input, eixo de redução, largura,
  epsilon, transformação do peso e coordenada de gamma;
- ativações, elementwise e tensor-scale textuais indexam o predecessor real;
- Conv2d declara o predicado de padding e lê o tensor fonte só quando a
  coordenada está dentro dos bounds; LayerNorm, GLU e depthwise nomeiam seus
  inputs, canais e kernel;
- score/value vision vinculam os dois operandos reais e o mapeamento
  `hidden -> [head,head_feature]`, preservando a redução nativa como fail-closed;
- todo RoPE vision deriva `axis`, `local_feature`, `pair`, `paired_feature`,
  `angle`, seno/cosseno BF16 e escolhe explicitamente subtração ou soma.

A validação percorre a classe completa e recusa operandos ordenados ausentes,
aliases livres e prosa de branch. Não há dispatch por assignment ID.

## Artefato real e bytes aprendidos

Fonte: `google/gemma-4-E4B`, revisão imutável
`411aa17b749aa952df1359d2dcea73917a544d9a`, Safetensors BF16 denso não
quantizado. A regeneração produziu:

- schema do artefato `25` e linguagem de fórmulas `11`;
- bytes: `21.385.294.917`;
- SHA-256:
  `5e7624410333c8bf38d05e54744c463bc5ec9bfab0e93f63b997874d11f39962`;
- constantes/decoders: `2.130/2.130`;
- bytes aprendidos incorporados: `15.992.314.836`;
- definições forward: `2.297`, das quais `1.574` pertencem à fronteira
  operand-closed;
- bindings ordenados ausentes, shorthands proibidos e RMS operands livres: `0`;
- RoPE vision explícito: `32/32`.

`npm run audit:literal -- --verify-gemma4-payloads` comparou cada payload com
os seis arquivos fonte. Os dois lados produziram SHA-256 de storage
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
O relatório é `/private/tmp/llm-inner-loop81-v25-source-audit.json`, SHA-256
`7b167601bd5da2567d339715f14d90926b4c729c392af6a3fb8d3dbf91b07eab`.

## Prova source-removed

O diretório `./gemma-4-E4B-dense` foi movido fisicamente sob restore trap. Com
o caminho ausente, o leitor verificou todos os 2.130 compromissos e
15.992.314.836 bytes, montou 2.709 operações forward, 20 operações de geração
e 42 transições KV, sempre com `sourceCheckpointAccessed=false`. O cálculo
end-to-end conserva 2.609 operações literais e exatamente 100 instâncias
fail-closed. Relatório:
`/private/tmp/llm-inner-loop81-v25-source-removed.json`, SHA-256
`6994df3b3db8b6b8e5b469f14f695814e6c9c4c2610b5c58c78f69ec8025f8d8`.

Ainda sem a fonte, foram renderizadas views concretas para RoPE vision,
RMSNorm, GELU, residual elementwise e Conv2d. Elas exibem, respectivamente, a
feature pareada `16`, gamma real `7.6875`, predecessor indexado completo,
dois predecessores residuais e pesos Conv2d BF16 decodificados. Os hashes dos
relatórios são:

- RoPE: `c079feb9851a6f1c1324e373cee9c8f1f54a20ad621b78b872027ad0f2b8c068`;
- RMSNorm: `8bbad63298e38f943a8bf1c0561ff7e4de6d2c3dd82cb6ed1232820215e69a9c`;
- ativação: `a6f0f8a1f9ea17fc7ca3a3849f978acc82e1446e04c01682eeb523a770a2b7ab`;
- residual: `db27571d812281d2d921d22707603524561c415248dd42c375b823386bd6d62b`;
- Conv2d: `e9d5232323c2d0915ffee2a3950aa559f373be2caaecb876846d2b3e12d51e89`.

Uma tentativa de renderizar
`composite_image_features/vision_layer_0_attention_scores` foi corretamente
recusada com: `vista escalar falha fechada porque a redução
pytorch-native-batched-matmul ainda não possui agenda literal comprovada`.

Por fim, a captura autoritativa
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`
foi comparada em tolerância absoluta e relativa zero, ainda com a fonte
ausente. Prefill e geração foram `lossless-within-dtype`; o token `184` na
posição `2`, logits de seleção e todos os 24 caches produtores passaram sem
primeira divergência. Relatório:
`/private/tmp/llm-inner-loop81-v25-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Limite preservado

As 100 instâncias Apple Accelerate BMM — 64 de visão e 36 de áudio — agora
nomeiam os operandos e coordenadas corretos, mas continuam com agenda escalar
`unpublished-fail-closed`. O schema v25 não promove a fórmula matemática sem
uma árvore de acumulação autoritativa. Por isso esta mudança é evidência
candidata de clausura de operandos, não certificação do checkpoint, e
`.agent-loop/checkpoints/gemma4-dense-lossless/` não foi criado.

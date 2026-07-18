# Gemma 4 E4B: cálculo navegável end-to-end sem source — 2026-07-18

## Resultado candidato

O leitor do artefato literal agora produz uma única vista finita que liga os
inputs declarados ao forward completo, aos logits, às atribuições greedy
concretas, às transições KV e aos outputs terminais. O forward é armazenado uma
vez e referenciado pelas invocações de prefill/decode; não é duplicado por
passo e não é substituído por um decoder opaco.

A mesma construção valida fail-closed a cobertura de storage. Cada uma das
constantes incorporadas precisa pertencer exatamente a uma destas classes:

- operando aprendido alcançável pelo fecho de produtores dos logits; ou
- projeção K/V ou K-norm local de uma camada consumidora que o programa
  declara runtime-unreachable porque reutiliza o cache de sua camada produtora.

Remover uma declaração da segunda classe, declarar uma constante alcançável
como unreachable, omitir seu decoder ou deixar storage órfão interrompe a
construção.

## Evidência real source-removed

- modelo: `google/gemma-4-E4B`;
- revisão: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- source denso BF16 `model.safetensors`: `15.992.595.884` bytes, SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- `config.json`: SHA-256
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`;
- artefato schema v8: `21.339.397.767` bytes, SHA-256
  `d25f9e2c7cbdd05d50de65c93911537057498b139ae0d7a5fe7a85480244b9b3`.

O diretório `gemma-4-E4B-dense` foi movido para fora do caminho durante toda a
inspeção e restaurado por `trap` ao final. O comando executado foi:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --end-to-end-calculation \
  --generation-max-new-tokens 2 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop64-e2e.HldZPl/end-to-end.json
```

O relatório declara `sourceCheckpointAccessed=false`, possui `13.907.629`
bytes e SHA-256
`9aec2bd84bfa45b2a7c322149425a3285c3401a9304addc5436d91b9eca28bca`.
Sua vista contém:

- 13 inputs declarados, incluindo `input_ids`, modais, posições,
  `max_new_tokens` e `eos_token_id`;
- 2.709 atribuições forward até `final_logit_softcap`;
- 10 inputs externos realmente alcançados pelo forward;
- 2.076 constantes aprendidas alcançáveis, todas com decoder;
- 20 atribuições greedy concretas para dois passos;
- 42 transições KV, com ownership produtor/reuso;
- 70 literais numéricos com bits F64/F32/BF16; e
- 54 projeções K/V e K-norm locais explicitamente runtime-unreachable.

Assim, `2.076 + 54 = 2.130`: toda constante incorporada está classificada e
endereçável sem o checkpoint.

## Revisão independente anterior

Antes da mudança, o commit
`9598fd2d11d031c62a437cfeec603b3d4da39f66` foi inspecionado. `npm run
typecheck` e os 227 testes passaram, e uma nova slice source-removed de
`final_logit_softcap` confirmou as 2.709 atribuições, 2.076 constantes
alcançáveis, 68 literais forward e as mesmas 100 instâncias BMM fail-closed.
Portanto a alegação candidata de navegação do loop 63 é aceita sem alterar a
classificação de fidelidade.

## Limites preservados

- As 64 instâncias vision score/value e as 36 instâncias audio AC/BD/value
  continuam `fail-closed-runtime-reduction`.
- Áudio continua numericamente aproximado desde
  `composite_audio_features`, inclusive logits e caches.
- Esta vista fecha composição e cobertura navegável; não estabelece uma nova
  agenda numérica para kernels nativos.
- `.agent-loop/checkpoints/gemma4-dense-lossless/` permanece ausente.

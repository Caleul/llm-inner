# Gemma 4 E4B: slice navegável até logits com source removido — 2026-07-18

## Resultado candidato

O leitor do artefato literal agora calcula o fecho transitivo de produtores de
qualquer atribuição instanciada. A slice preserva a ordem canônica, os domínios,
fórmulas escalares, casts, reduções, arestas, constantes aprendidas, decoders e
bits dos literais numéricos já serializados; somente ramos que não podem afetar
o alvo são removidos. Nenhuma semântica é reconstruída pelo nome da operação.

Na E4B densa real, a slice de `final_logit_softcap` contém as 2.709 atribuições
do forward desde `composite_placeholder_masks` até `softcapped_logits`. Todos
os produtores referenciados estão presentes antes do consumidor. Ela declara:

- 10 inputs externos usados pelo forward;
- 2.076 constantes aprendidas alcançáveis, cada uma com decoder e consumidores;
- 68 tokens numéricos forward com bits F64/F32/BF16;
- 2.609 operações literais; e
- 100 instâncias fail-closed nas cinco classes BMM nativas já conhecidas.

Esse resultado torna o limite numérico visível na própria navegação: não
promove os BMMs de Apple Accelerate para uma ordem escalar inventada. Por isso
continua sendo evidência candidata de navegação, não o checkpoint dense-lossless.

## Evidência source-removed

O diretório `gemma-4-E4B-dense` foi movido para fora do caminho durante toda a
inspeção e restaurado por `trap` ao final. O comando foi:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --calculation-slice final_logit_softcap \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-loop63-final-logits-calculation-slice.json
```

O relatório de 12 MiB tem SHA-256
`f6f7552ca1a3f1f5b974219b4921fc0fd6009d7f5dec8b3b3a22c1e3b42387af`,
declara `sourceCheckpointAccessed=false` e referencia o artefato schema v8 de
21.339.397.767 bytes. O SHA-256 do artefato foi recalculado nesta execução:
`d25f9e2c7cbdd05d50de65c93911537057498b139ae0d7a5fe7a85480244b9b3`.

## Revisão independente do schema v8

Antes da mudança, o commit `255f0a077f2d37548176d29442a93b6b0df7eb4e`
foi inspecionado e a suíte original passou com 227 testes. A abertura do
artefato real nesta execução também reconstruiu e validou fail-closed a tabela
`numericLiterals`; portanto a reivindicação candidata do loop anterior sobre
os bits numéricos é aceita. Essa aceitação não altera os limites de fidelidade
de áudio/BMM nem autoriza o marcador de checkpoint.

## Limites preservados

- 64 instâncias vision score/value e 36 instâncias audio AC/BD/value continuam
  `fail-closed-runtime-reduction`.
- A slice é uma navegação fiel do programa; não é uma nova execução
  diferencial de áudio nem resolve a divergência já medida nessa modalidade.
- `.agent-loop/checkpoints/gemma4-dense-lossless/` permanece ausente.

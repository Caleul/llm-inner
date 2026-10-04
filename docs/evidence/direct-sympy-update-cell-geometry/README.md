# Divisão do domínio orientada pelas células dos updates

O controlador agora oferece `--seed-update-cells`. Ele lê os pesos necessários em streaming e usa os limites existentes dos updates de atenção e MLP para escolher cortes da entrada. Para cada coordenada, procura a primeira faixa Half cujo raio da célula de arredondamento, considerando o menor vizinho, supera estritamente o maior update limitado entre as layers. Esses cortes são oportunidades de simplificação; não são resultados numéricos adotados. Cada região executa novamente a compilação e suas provas.

Neste checkpoint, os limites de atenção são 0,006778717041015625 e 0,0023746490478515625. Os limites de MLP são menores. O processo encontrou magnitude 32 para X1 e 16 para X2. Esses valores não são constantes da arquitetura nem foram escolhidos por respostas do modelo.

Cada eixo selecionado mantém três faixas: negativa externa, central e positiva externa. O limite `--seed-max-regions` controla o crescimento da geometria; eixos adicionais permanecem intactos quando esse orçamento não admite outra divisão. Limites indisponíveis mantêm a geometria original. A faixa central sempre preserva ambos os zeros.

Os cortes personalizados são ranks Half exatos, armazenados nos nós da árvore. A auditoria reproduz os cortes e verifica a cobertura disjunta de todo o domínio. Cortes inválidos ou geometrias corrompidas são recusados. A importação de geometria preserva os novos cortes, mas descarta todas as expressões e provas de outra identidade.

A ordem `update-cells` visita primeiro as combinações externas e então continua a faixa central em ordem lexical. Essa escolha afeta somente o trabalho pendente; não reassocia operações numéricas, não omite entradas e não cria provas de equivalência. Execuções sem a nova opção mantêm a ordem lexical padrão.

## Execução real e arquivos efetivos

Foi criado um estado novo, com nove folhas iniciais e nenhum resultado numérico reaproveitado:

```sh
python helpers/direct_sympy_partition_run.py \
  docs/evidence/direct-sympy-test-checkpoint \
  artifacts/direct-sympy-input-partitions/update-cell-state \
  --seed-update-cells --seed-max-regions 9 \
  --max-attempts 48 --region-seconds 5 --max-paths 128
```

As quatro primeiras tentativas emitiram todas as regiões externas, cobrindo 553.648.128 padrões (13,735691987513007% do domínio). O controlador continuou a faixa central e terminou o lote assim:

| Evidência | Resultado |
| --- | ---: |
| Tentativas | 48 |
| Regiões emitidas | 23 |
| Corpos distintos preservados | 5 |
| Padrões abrangidos | 614.428.672 |
| Padrões pendentes | 3.416.297.472 |
| Cobertura geométrica | 15,243622366024974% |
| Comparações nativas das 23 regiões | 5.980 |
| Comparações adicionais das quatro regiões externas | 32.784 |
| Divergências observadas em ambos os ensaios | 0 |

Os cinco arquivos de expressões estão preservados neste diretório; `artifact-map.json` os relaciona aos domínios e hashes. O estado completo e sua identidade estão em `frontier.json`. O diretório retomável é `artifacts/direct-sympy-input-partitions/update-cell-state`. Todos os hashes das fontes atuais coincidem com a identidade salva. Limites de tempo e tamanho resultaram em subdivisões, sem publicar resultados incompletos como finais.

Esta cobertura pertence a uma árvore nova e auditada. Não foi somada às evidências anteriores de domínios sobrepostos. O tempo do lote está registrado no estado, mas houve outras verificações simultâneas; não foi executado benchmark isolado de tempo ou memória e não se afirma um fator de aceleração.

## Sondagem da faixa central

Também foram compilados, separadamente, domínios com X1 entre -65.504 e -32 e X2 centrado em zero. Até magnitude 2^-7, o processo de substituição e simplificação provou a saída constante e emitiu uma expressão de 11 caracteres. Os dois domínios testados receberam 16.400 comparações nativas, incluindo zeros mistos, sem divergências. O domínio mais amplo, até 2^-6, ultrapassou o limite do artefato e foi recusado.

`central-probes.json` conserva os resultados completos e a recusa; `central-probes-parity.json` conserva a paridade dos arquivos emitidos. Esses domínios se sobrepõem entre si e à árvore; não foram acrescentados à cobertura. O resultado constante é uma eliminação estrutural de dependências provada pelo compilador, sem lookup de respostas. A próxima melhoria deve aproveitar esses domínios centrais amplos preservando a auditoria de interseções e a identidade das provas.

## Testes e limites

Build concluído e 24 integrações passaram, sem falhas ou testes ignorados. Os oito testes de partição verificam cobertura de todos os 63.488 padrões Half, cortes personalizados, ambos os zeros, limite de crescimento geométrico, fallback de limites indisponíveis, prioridade das quatro regiões externas, retomada e ausência de artefato final quando faltam regiões. Os dois testes de importação verificam também a preservação dos cortes personalizados e o descarte de provas anteriores. O mapa global mantém as entradas anteriores e acrescenta esta execução.

A paridade dos arquivos do checkpoint é amostral, bit a bit, sem tolerância. Cobertura geométrica não é execução exaustiva de todos os padrões. O escopo permanece posição 0, dimensão 2, um token; a coordenada completa, comprimento variável, múltiplos tokens, o vetor inteiro e a emissão Rust continuam pendentes. As expressões seguem a instrução humana mais recente de strings matemáticas; JSON armazena estado e evidências.

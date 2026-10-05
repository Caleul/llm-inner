# Redução estrutural e vetor do próximo token no Colab

Esta rodada mantém o checkpoint e o forward original CPU arm64. A compilação
e os testes pesados foram executados no Colab, sessão
`llm-inner-final-logits-20261005`, com A100-SXM4-40GB, 12 CPUs e cerca de
83,5 GiB de RAM. A máquina local fez edição, transporte e build TypeScript.
Os relatórios de compilação em `remote/` não são o JSON final do modelo.

## Correções verificadas

- A expansão elementar da raiz usa uma fração parcial com cinco termos,
  preservando a ordem F64 certificada, a quantização F32 e a escala binária.
  As ocorrências do argumento caíram de 13 para 7. A inversa continua
  `R32(1 / R32(sqrt(x)))`; não foi trocada por um rsqrt arredondado separadamente.
- Conversões para Half reconhecem valores já representáveis em Half após
  somas com zero e operações com fatores unitários. As operações e seus
  zeros com sinal permanecem; somente a conversão redundante desaparece.
  A prova é reavaliada depois que os produtores estabilizam, inclusive no
  caminho antecipado de soma. A probabilidade constante da atenção de um
  token agora participa dessa prova antes de converter o produto.
- O emissor principal e sua CLI produzem, por padrão, somente o vetor do
  próximo token. Cada dimensão contém a seleção por comprimento `N`, com
  os cálculos da última posição de cada comprimento válido. O cabeçalho
  declara `outputSelection: "last-position"`; `position: 0` identifica a única
  linha lógica do vetor, não a posição zero da entrada. Coordenadas isoladas
  e a opção explícita `all-positions` são diagnósticos.
- A CLI aceita `--target-snapshot PATH`. O compilador valida checkpoint,
  configuração, identidade das fontes e descoberta antes da reutilização;
  a referência remota não é redescoberta como um forward CUDA/x64.

O teste do seletor emite e relê um vetor literal de quatro expressões,
verifica seus bits para comprimentos 1..8 e não fornece entradas de tokens
ausentes. Também verifica rejeição atômica de matriz e vetor incompleto.
Esse teste usa contribuições causais pequenas; não demonstra emissão do
modelo completo. A CLI do checkpoint tem um smoke separado, com rejeição
explícita pelo orçamento e sem publicação de arquivo parcial.

## Paridade das expansões

O teste C++ compila a própria árvore JSON, com contração desativada. Cobriu
16.777.216 entradas normalizadas para raiz e inversa (33.554.432 resultados),
mais 50.331.648 fronteiras inferior/empate/superior da quantização. Nenhuma
comparação divergiu. Testes adicionais cobrem valores Half positivos,
fronteiras de expoente e casos que distinguem inversa de rsqrt.

Na A100, o avaliador CUDA executou as operações elementares do mesmo JSON
em kernels separados, sem fusão. Foram 33.554.432 resultados, zero
divergências, aproximadamente 0,834 s e 48 MiB de pico de memória alocada
na GPU. É uma validação de primitiva, não a paridade de um artefato final.
As constantes dessa primitiva pertencem ao compilador; não são um lookup
de respostas do modelo.

## Crescimento e ensaios descartados

A instrumentação anterior encontrou amplificação de cinco vezes entre a
projeção V e o contexto da atenção de um token: uma conversão Half cujo
produtor já era Half permanecia viva. A reavaliação após substituição
remove essa conversão. Com a raiz menor, a dimensão zero de um token
passou de 79.655.384.760.357.754.817 para
2.679.354.850.605.462.323 bytes previstos, cerca de 29,7 vezes menor.
São tamanhos calculados da sintaxe literal, **não arquivos produzidos**.
Continuam acima de 512 MiB; não demonstram conclusão da meta.

Os relatórios preservam a investigação:

- `residual`: a regra por célula de arredondamento preservou paridade,
  mas não reduziu a expressão global; não virou configuração padrão.
- `cofactor`: o ponto fixo foi atingido em três rodadas, com duas promoções
  aceitas; reduziu um token para 56.393.457.973.004.369.801 bytes previstos,
  ainda sem permitir emissão.
- `rational-five`: 24 coordenadas e 288 comparações passaram antes da
  substituição do ensaio pela correção da conversão Half.
- `half-lattice` e `half-lattice-value`: versões intermediárias preservaram
  paridade, mas ainda não eliminaram a amplificação na atenção.
- `half-lattice-fixedpoint`: o novo teste de soma interna com zero falhou
  (cinco referências em vez de uma), bloqueando a compilação. A correção
  no caminho antecipado de soma gerou `half-lattice-fixedpoint-v2`.

Cada raiz tem seu snapshot e resultados próprios; estados numéricos de uma
versão não foram reaproveitados como validação da seguinte. A cópia remota
teve apenas as identidades das alterações auditadas atualizadas. Manifestos
`superseded.json` distinguem ensaios substituídos de falhas ou conclusão.

## Recursos, regressão e limite da entrega

O lote corrigido terminou as 32 coordenadas (quatro logits para cada
comprimento 1..8), com 384 comparações bit a bit e nenhuma divergência.
Todas ficaram sem primitivas pendentes no grafo de compilação. Nenhuma
expressão literal do checkpoint pôde ser emitida dentro do orçamento.

| Tokens | Coordenadas validadas | Segundos por coordenada | Máximo RSS por trabalhador (MiB) |
| --- | --- | --- | --- |
| 1 | 4 | 6,5–6,9 | 149,9 |
| 2 | 4 | 23,6–24,0 | 189,4 |
| 3 | 4 | 40,5–41,1 | 240,1 |
| 4 | 4 | 59,2–60,3 | 265,4 |
| 5 | 4 | 81,1–81,7 | 316,8 |
| 6 | 4 | 110,1–112,4 | 382,3 |
| 7 | 4 | 148,2–148,9 | 600,3 |
| 8 | 4 | 198,5–204,8 | 642,7 |

O lote anterior observou 202,6–215,5 s e 646,5 MiB no comprimento oito.
O ensaio atual observou tempos menores, mas a diferença não é uma medida
isolada de aceleração. O benefício comprovado é a redução estrutural e a
preservação da paridade; o pico de RAM não caiu de maneira uniforme.

Quatro trabalhadores compilaram coordenadas independentes, com contexto
próprio. Cada processo foi monitorado e limitado a 6 GiB e 300 s.
RAM/RSS e memória CUDA são medidas separadas do orçamento de emissão.
O emissor do vetor aplica 512 MiB ao arquivo acumulado, incluindo condições
e expressões; ele não corta caminhos nem publica vetores incompletos.
O smoke da CLI usa deliberadamente 1 MiB para exercitar essa rejeição.

O emissor também exige `cofactor.converged`: limite de rodadas/candidatos
não equivale a estabilização. Os dois testes de admissão do checkpoint
passaram no Colab usando `LLM_INNER_DIRECT_JSON_SNAPSHOT`: o caso com uma
rodada é recusado antes da emissão; o caso estabilizado chega à verificação
de bytes e continua sem publicar artefato acima do orçamento. Os logs em
`remote/llm-inner-final-logits-convergence-gate/` preservam ambos os estados.
O teste via CLI ultrapassou o prazo de resposta do transporte; o resultado
persistido foi relido e confirmou os dois gates, sem repetir a execução.

As regressões, tempos, picos de RAM e progresso efetivo estão discriminados
em `validation.json` e nos logs integrais. A seleção JSON teve 103 passes,
zero falhas e cinco skips de integrações dependentes de ambiente. O teste
do emissor inicialmente não recebeu o perfil SiLU no transporte; o perfil
foi restaurado e o teste reexecutado, preservando o log da falha inicial.
Os resultados anteriores do mapa de testes permanecem separados.

Os ensaios compartilharam recursos com testes em parte. Comparar seus tempos
não demonstra um fator de aceleração controlado. O avanço exigido para a
conclusão continua sendo emitir o JSON literal do checkpoint, relê-lo e
demonstrar sua paridade. Esta rodada reduz um erro concreto de crescimento
e prepara o recorte correto do emissor; não entrega esse artefato final.

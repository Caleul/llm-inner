# Restrições conjuntas antes da distribuição dos caminhos

O compilador tratava como independentes três decisões sobre projeções do
mesmo vetor normalizado. Na região central com coordenadas de mesmo sinal,
isso produzia oito caminhos, embora quatro fossem impossíveis. A nova prova
propaga essa relação durante a enumeração, antes de expandir esses caminhos.

Na comparação da coordenada 2, posição 0, com um token, sobre
`X1=[0.0625,2]` e `X2=[0.0625,1]`:

| Evidência | Anterior | Atual |
|---|---:|---:|
| Produtores incorporados | 17 | 17 |
| Caminhos emitidos | 8 | 4 |
| Caracteres do arquivo efetivo | 11.064.357 | 4.100.744 |
| Comparações do corpus com o checkpoint | 8.196 | 8.196 |
| Divergências | 0 | 0 |

O arquivo efetivo ficou 62,94% menor. A emissão medida levou 145,52 s e
48,68 s, respectivamente, mas essas medições ocorreram com outras tarefas
em execução; não são um benchmark isolado de tempo ou memória. As medições
anteriores de paralelismo continuam registradas no mapa de testes.

`central-positive.expr` contém somente entradas originais, constantes,
operações elementares e condições. Não contém aliases do compilador,
chamadas a layers, leitura do checkpoint ou conversões pendentes. A expressão
é uma string matemática compatível com a sintaxe do compilador SymPy.
Os arquivos JSON desta pasta são registros de validação e retomada.

## Prova de exclusão

Para um vetor Half normalizado com pesos de normalização exatamente `+1`
ou `-1`, o compilador certifica um limite inferior do quadrado da norma.
Ele considera epsilon, a redução original em F32, raiz, recíproco,
multiplicação e armazenamento Half. Entradas que admitem o vetor nulo não
recebem um limite positivo e não podem usar essa exclusão.

Cada projeção é descrita somente para a prova como uma forma linear do
mesmo vetor, acrescida de um erro que engloba cada operação F64, fronteira
F32 e fronteira Half original. Coeficientes, inversão da matriz pequena e
limites são calculados com frações exatas. A ordem aritmética da expressão
emitida permanece intacta.

Se projeções independentes fossem simultaneamente pequenas, a inversão
desses coeficientes limitaria a norma abaixo do piso certificado. Esse
contexto é impossível e pode ser descartado. Matrizes singulares, operações
não suportadas, risco de overflow, limites nulos e esgotamento do orçamento
da prova mantêm o caminho alcançável. O limite de oito dimensões e 32
sistemas restringe o trabalho da prova, nunca o domínio do modelo.

## Validação do arquivo efetivo e limites de domínio

`exhaustive.py` compila o arquivo efetivo com Clang, sem contração de
operações, e transmite entradas Half em lotes de 4.096. Compara a saída
binária com o forward CPU do checkpoint. Ele percorreu todos os
20.980.737 pares da região positiva, sem divergências.

A região simétrica negativa foi recompilada independentemente e produziu
exatamente os mesmos bytes da expressão. Outros 20.980.737 pares foram
comparados exaustivamente, também sem divergências. Os dois domínios somam
41.961.474 pares distintos; compartilhar o arquivo não compartilha respostas
nem preserva intermediários em runtime.

Uma tentativa de aplicar esse arquivo fora desses domínios, nas duas
regiões de sinais opostos, foi rejeitada: cada região apresentou 92.413
divergências em 20.980.737 comparações. Por exemplo,
`X1=-0.0625, X2=0.1690673828125` produz `0.00006076693534851074`
nesse candidato, enquanto o checkpoint retorna `0.00006079673767089844`.
O primeiro valor sequer pertence à grade Half subnormal. Isso mostra que
certificados de magnitude usados na simplificação não podem ser transferidos
para contextos que admitem novos cancelamentos. Esses probes não validam
as regiões nem alteram a cobertura salva.

A compilação própria dessas regiões atingiu o orçamento de 8 MiB antes
da emissão completa, com 22.862.412 caracteres lógicos. Continuam pendentes.
O próximo trabalho deve investigar a expansão da conversão final quando a
projeção admite valores subnormais e cancelamentos, preservando esses casos.

## Retomada e mapa de testes

A mudança numérica tornou o estado anterior incompatível. A retomada
importou somente sua geometria, resetou os 241 resultados completos e
recompilou os resultados. O novo estado registra 80 admissões frescas,
655.726 comparações de corpus sem divergência e todas as identidades de
fontes, backend e checkpoint. Arquivos anteriores são referenciados somente
quando os bytes da nova compilação coincidem, com SHA-256 verificado.

A cobertura regional aumentou de 2.274.846.724 para 2.316.808.198 pares
Half, ou 57,48% do domínio. Restam 1.713.917.946 pares. Cobertura compilada
não significa que todos esses pares foram comparados numericamente.

`npm run build` passou. Os 30 testes de integração SymPy passaram sem skips,
incluindo a nova prova exaustiva das restrições e testes de limites de
memória/tamanho, paridade, zeros com sinal, isolamento de condições e
incompatibilidade dos estados. O histórico existente em
`docs/direct-string-validation.json` foi preservado e acrescido desta etapa.

A suíte geral `npm test` executou 624 testes: 570 passaram, 15 falharam e
39 foram ignorados por seus requisitos opcionais, incluindo os testes SymPy
que foram executados separadamente com o ambiente explícito. As mesmas
15 falhas foram reproduzidas nos quatro arquivos afetados, reconstruídos
em um arquivo Git isolado do commit anterior `abf5d72`: 16 passaram e
15 falharam nessa reprodução. As causas são dados locais ausentes, hashes
antigos de evidências Gemma e um contrato de ambiente fixado em versões
anteriores de Python/macOS. Nenhuma dessas falhas foi introduzida por esta
mudança; a suíte geral permanece sem aprovação completa. Os logs preservam
os nomes e as causas, sem alterar contratos ou fabricar evidências ausentes.

Não existe ainda um arquivo da coordenada inteira, nem paridade para
múltiplos tokens ou vetor completo. `frontier.json`, `artifact-map.json`,
`validation.json` e os relatórios exaustivos documentam o resultado parcial
e impedem tratá-lo como conclusão do objetivo.

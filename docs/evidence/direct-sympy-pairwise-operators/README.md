# Composição vetorial em pares: implementação e limites

O AGENTS e o plano da meta agora permitem qualquer representação adequada e
exigem composição ordenada de operadores: preparar 1|2 e 3|4 em paralelo,
estabilizar cada substituição e depois compor seus resultados. O novo backend
usa expressões matemáticas; JSON nos descritores é transporte de compilação.
Interfaces Xn internas são disjuntas e não recebem certificados da entrada.
Nenhum identificador de interface sobrevive à composição.

A integração atual cobre as projeções do adaptador Llama. A sequência completa
normalização/atenção/residual/MLP ainda não é planejada como blocos independentes.
A publicação de produtores e de arquivos de blocos verifica o teto acumulado;
as reservas incluem condições presentes nas expressões. RAM é medida à parte.
O novo módulo participa da identidade dos savepoints: fontes incompatíveis são
rejeitadas antes de restaurar produtores. A mudança local de bitcasts
condicionais que estava sem validação foi retirada.

## Validação

- Sete testes de operadores/reduções: composição vetorial de cinco operadores,
  árvore em três níveis, ordem de arredondamento, ramos, zeros com sinal,
  interfaces, memória e orçamento acumulado. A coordenada de referência
  posição zero/dimensão dois passou em 60 casos, com primitivas numéricas
  conservadas. Isso não é paridade de um artefato final completamente expandido.
- Nove testes de savepoints; 30.722 casos nativos de arredondamento retomado sem
  divergência. Uma alteração simulada no backend de operadores rejeita o estado
  sem modificar os produtores do compilador.
- Regressão portátil: 624 testes, 575 passaram, 49 skips, zero falhas.
- A100 via Access Broker: 12 CPUs, 89.629.196.288 bytes de RAM,
  42.405.855.232 bytes de VRAM, CUDA 13.0. As projeções numéricas V/O usam os
  pesos reais do checkpoint e a redução original em quatro lanes: 63.488 pares,
  126.976 resultados, zero divergências. Tempo CPU 28,08 ms, GPU residente
  1,53 ms. Não inclui transferências nem demonstra aceleração do compilador.
  SymPy permanece na CPU. A sessão foi encerrada e o inventário confirmou zero
  runtimes ativos.

## Comparação do checkpoint

Execuções locais frescas, em sequência, sem retomar estados antigos, teto de
512 MiB e orçamento nominal de 45 segundos. Ambas concluíram 14 produtores e
pararam em `model.layers.0.post:1`. Seus prefixos e tamanhos estabilizados são
idênticos. Sequencial: 45,80 s, pico observado agregado de RAM 2.018.344.960
bytes. Pares: 46,14 s, 1.970.241.536 bytes, quatro composições de projeções.

Não houve ganho de avanço nem redução da expressão. A normalização posterior
continua produzindo crescimento acentuado. Nenhum artefato final foi emitido;
as demais coordenadas não foram despachadas. Os prefixos incompletos foram
mantidos fora do repositório, com tamanho e hash registrados neste diretório.

`compare.py` reproduz os dois ensaios com o Python numérico configurado. O
monitor inicial encontrou uma corrida de leitura de RAM de um filho encerrado;
foi corrigido para tolerar `psutil.Error` e a comparação foi refeita. Leituras
RSS são amostradas, não uma garantia do pico absoluto. Após o ensaio foi
acrescentado o registro de substituições executadas nos filhos; essa mudança
é apenas de instrumentação e não alterou as regras numéricas.

A meta permanece incompleta: falta integrar a composição da arquitetura inteira,
reduzir o crescimento na normalização e entregar a primeira coordenada sem
primitivas pendentes, com paridade do artefato efetivo e diferentes comprimentos.

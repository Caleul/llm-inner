# Provas locais no fechamento de conversões — 2026-10-03

O fechamento de `Piecewise` agora processa cada conversão sob os limites
daquele braço, incluindo as negações dos predicados anteriores. O corpo
passa por simplificação aritmética e `factor()`/`simplify()` antes do fechamento
numérico e novamente antes de retornar à expressão externa.

As tabelas de provas são isoladas por contexto. Novas provas não entram no
cache global de produtores e não passam aos irmãos. A restauração ocorre
também após exceções. As strings imutáveis e as assinaturas estruturais
continuam compartilhadas durante a compilação; não há novos intermediários
no artefato de execução.

A propagação reconhece comparações com constantes negativas, anteriormente
ignoradas porque o sinal é um nó unário na gramática. Também reconhece
limites de magnitude sobre uma fonte finita certificada. Fronteiras estritas
usam o vizinho F64 correto. Um complemento com dois intervalos de sinais
opostos permanece conservador; não se substitui essa união por um intervalo
que eliminaria valores válidos. A comparação não fornece prova de dtype.

Dois testes nativos cobrem as mudanças: 30.722 valores Half em `[-1,1]`,
avaliados em duas expressões de divisão/conversão (61.444 comparações), e
32.770 valores em `[-2,2]` para uma atualização residual. Zero diferenças
de bits, incluindo zeros com sinal e fronteiras. No primeiro caso desaparece
uma classificação normal/subnormal impossível no braço pequeno. No segundo,
a atualização desaparece somente no braço em que é menor que a célula Half;
o outro braço conserva o cálculo.

Ramos elementares já fechados percorrem o caminho existente. A primeira
versão também os envolvia em novos contextos e repetia CAS desnecessariamente;
essa passagem foi removida. Os ensaios locais aconteceram simultaneamente
com testes e não fundamentam uma comparação de desempenho.

## Verificação do checkpoint real

A execução atual criou um estado novo em
`artifacts/direct-sympy-branch-context-current/state`, para a coordenada 2,
posição 0. As fontes do manifest coincidem com as fontes atuais. O estado
anterior possui identidade diferente e não foi reutilizado.

Foram produzidas as mesmas 13 dependências completas. Os 13 arquivos foram
relidos em blocos de 1 MiB: tamanho e SHA-256 coincidem com os registros e
com os produtores anteriores, totalizando 1.702.196.630 bytes. A próxima
ativação continua com 6.913.579.521 caracteres e foi recusada antes de sua
alocação, pelo limite de 2 GiB. Não há nova coordenada completa, paridade
final do modelo ou redução de tamanho no checkpoint nesta mudança.

Esta correção fecha conversões que já estão dentro de braços explícitos.
Ainda falta expor os braços dos produtores fechados à composição seguinte
e levar suas provas de dtype/intervalo para esses contextos selecionados.
A prova de um braço não pode ser transferida para o produtor inteiro.
Essa etapa continua pendente; os testes locais não a substituem.

## Mapa de validação

- 93 testes Python, sem skips.
- Oito integrações Node, sem skips; build TypeScript aprovado.
- Regressão: 599 testes, 581 aprovados, três skips e as mesmas 15 falhas
  do mapa anterior; nenhum nome de falha novo.
- Logs e auditoria dos arquivos estão nesta pasta; o mapa acumulado está
  em `docs/direct-string-validation.json`.

Os arquivos JSON desta pasta contêm evidências e metadados. As expressões
continuam sendo strings matemáticas, conforme a instrução mais recente.

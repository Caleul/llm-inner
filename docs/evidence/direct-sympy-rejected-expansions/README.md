# Experimentos rejeitados na expansão — 2026-10-03

Nenhuma das duas alternativas foi incorporada ao compilador. A representação
das expressões continua sendo string matemática e a coordenada completa
continua pendente. Os arquivos JSON nesta pasta contêm somente evidência.

As regras adicionais de máscaras passaram nos testes, mas os dez produtores
emitidos na execução limitada mantiveram os mesmos tamanhos e SHA-256 da
versão `7b537dc`. Não houve avanço adicional. A tentativa levou 6,510 s e
parou antes de alocar a próxima expressão de 164.608.933 caracteres, acima
do limite de 134.217.728. O patch experimental foi preservado e revertido;
seus estados não devem ser reutilizados pela versão restaurada.

Uma semente racional quadrática/quadrática para a raiz usaria menos
ocorrências da entrada que a semente cúbica/quadrática atual. Contudo, a
enumeração de todos os 16.777.216 casos de mantissa/paridade F32 encontrou
39 resultados diferentes da referência e um ponto médio de arredondamento.
Isso também invalida a precondição usada para eliminar a duplicação de
paridade na conversão F32. O programa C++ e sua saída estão preservados;
seu código de saída 1 é a rejeição esperada, não um teste aprovado.

A avaliação da raiz mantém o produto e a correção de Newton na mesma ordem,
com contração FMA desabilitada. A falha foi observada no domínio normalizado;
não se tentou admitir essa alternativa para os demais expoentes ou
subnormais. Os coeficientes de produção permanecem intactos.

Após restaurar o compilador, os 86 testes Python passaram, incluindo o teste
da raiz emitida com 25.167.601 casos sem divergências. O build TypeScript
também passou. Isso verifica a restauração; não prova a coordenada final.

O próximo trabalho deve reduzir a expressão realmente restaurada durante o
fechamento numérico. Não há evidência para declarar que todas as repetições
observadas são artificiais: algumas preservam fronteiras e condições
numéricas. Qualquer nova fatoração deve ser medida e certificada antes de
substituir essa implementação.

# Limites exatos para operações com zero

## Causa e correção

A análise de intervalos adicionava uma margem de um ULP em operações cujo resultado era comprovadamente zero, ou cujo módulo era preservado por soma/subtração com zero. Essa margem podia fazer uma dependência nula parecer não nula e impedir provas posteriores.

`ConversionSession.bounds()` agora preserva o intervalo do operando finito em soma/subtração com zero, inverte os extremos em zero menos operando e certifica módulo zero em produtos e divisões com numerador zero e denominador comprovadamente não nulo. A informação do sinal continua separada: nenhum zero negativo é descartado por essa mudança. Operandos desconhecidos/não finitos e divisores com intervalo incluindo zero permanecem barreiras.

## Evidência

- `zero-tests.log`: quatro testes, 656.300 comparações anteriores e 319.488 comparações adicionais das expressões emitidas. Nenhuma divergência. Inclui ambos os sinais de zero, isolamento dos contextos e barreiras de divisão.
- `integration-tests.log`: build e 20 integrações passaram, sem falhas ou skips. Depois de adicionar a asserção do novo corpus nativo, `targeted-integration.log` confirma também a integração correspondente.
- `region-tests-final.log`: cinco testes com checkpoint, sem skips; 576 comparações nativas da expressão regional sem divergência. A primeira chamada isolada em `region-tests.log` não tinha o checkpoint no ambiente e ignorou três casos; ela não é usada como prova de paridade.
- `coordinate-region.expr`: artefato real, 21.922 caracteres, exclusivamente X1/X2 da entrada; domínio positivo entre 32 e 65.504, dimensão 2, posição 0, um token. `coordinate-region-parity.json` confirma o arquivo por SHA e 1.028 comparações adicionais sem divergência. O corpo é byte a byte idêntico ao artefato regional anterior.
- `old-state-rejection.log`: o estado da versão anterior é incompatível com o novo código e foi rejeitado. Não reutilizamos as provas antigas.

## Resultado completo ainda pendente

`coordinate-run.json` registra uma execução nova sobre o domínio inteiro: 25 dependências compostas, 4.476.543.086.072 caracteres lógicos e limite de emissão excedido. **A correção não diminuiu essa contagem e não produziu uma coordenada completa.** Os tempos desta execução não são comparação controlada de desempenho.

As expressões continuam em string matemática; JSON é metadado de teste/estado. O mapa está em `validation.json` e `docs/direct-string-validation.json`. O artefato completo, sua paridade, o último token com comprimento variável e o vetor de todas as saídas continuam pendentes. O próximo trabalho precisa eliminar crescimento efetivo nas normalizações e nas condições, além destas provas locais.

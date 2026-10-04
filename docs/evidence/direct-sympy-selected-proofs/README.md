# Fechamento numérico após selecionar uma dependência

O compilador conserva a expressão matemática anterior à expansão numérica de cada produtor, além de suas provas de tipo, intervalo e fronteiras. Isso é estado exclusivo da compilação: os arquivos finais continuam a substituir todos os aliases e contêm apenas a entrada e operações elementares.

Ao selecionar uma dependência, o caminho recebe um contexto numérico novo. As provas globais de tipo e intervalo permanecem válidas naquele subconjunto, e a fronteira selecionada pode acrescentar a magnitude mínima. Identidades de palavras fechadas da expressão original não são transferidas. Depois da seleção dos operandos, sua expressão anterior é novamente fechada e estabilizada com factor/simplify. O candidato só é admitido quando está completamente baixado, todas suas decisões já estão determinadas no contexto e seu custo totalmente substituído é menor. A condição de dispatch conserva o contexto anterior à própria decisão.

As expressões guardadas referenciam apenas dependências anteriores. Dependências descartadas pelo caminho não são reabertas para tentar uma otimização. Uma tentativa que ultrapassa o orçamento conserva a forma já compilada e equivalente; não omite entradas ou caminhos. Neste diagnóstico não houve parada por orçamento nas tentativas de fechamento numérico.

## Evidência

- Build e 21 integrações passaram, sem falhas/skips. Nove testes de caminhos passaram, incluindo 30.722 entradas Half com duas comparações nativas por entrada, zeros com sinal e isolamento entre ramificações. O teste verifica que o fechamento da expressão anterior foi efetivamente admitido e que os certificados originais não foram alterados. Os sete testes de streaming verificam também captura e ordenação das expressões.
- A composição global continua com 25 produtores e 3.248.163.585.496 caracteres lógicos antes da seleção dos caminhos. Esse número não é tamanho de um artefato emitido.
- O primeiro caminho medido caiu de 8.648.306.495 para 8.615.711.603 caracteres de corpo (0,3769%); suas condições caíram de 4.992.480.549 para 4.973.664.345. Foram 108 tentativas e 46 admissões em contextos distintos, não 46 produtores únicos. O arquivo completo continua recusado pelo limite.
- Oito regiões foram recompiladas com os hashes novos, preservando somente a geometria auditada do estado anterior. Os arquivos emitidos passaram em 2.080 comparações contra referência CPU carregada novamente. Três expressões distintas estão preservadas em `artifact-map.json`.
- Uma região negativa caiu de 19.132 para 15.298 caracteres (20,04%). A região positiva continua com 19.532 caracteres e passou em mais 1.028 comparações após leitura do arquivo salvo. Não há ganho de velocidade estabelecido.

Uma versão inicial da nova fixture usava uma potência constante ainda sem certificado de intervalo, e a auditoria recusou o R16 residual. A fixture foi corrigida para usar a constante decimal exata, como os pesos efetivamente incorporados pelo adaptador. Os testes completos foram reexecutados; os logs anteriores estão preservados para distinguir as etapas.

## Limites

Os arquivos continuam parciais: um token, posição 0, dimensão 2. O estado novo possui oito regiões emitidas e as demais permanecem pendentes, sem reaproveitar expressões antigas. A cobertura desses arquivos é de 176.639.488 padrões; 3.854.086.656 padrões ainda não possuem artefato. A paridade regional é amostral, não exaustiva nessa cobertura. A divisão auditada mantém o domínio inteiro.

Ainda faltam coordenada inteira, último token com comprimento variável, múltiplos tokens, todas as dimensões e a comparação controlada atual no Colab. O resultado não autoriza avançar às outras coordenadas. A continuação deve examinar as próximas substituições em que provas de correlação poderiam reduzir cópias, mantendo a ordem numérica e a validação do artefato.

# Propagação da magnitude nas ramificações

Uma condição pode excluir os valores próximos de zero e ainda admitir ambos os sinais. O intervalo assinado anterior perdia essa informação. `FiniteSource` agora registra também a magnitude mínima comprovada, com propriedade exclusiva da ramificação. Isso permite excluir zero, calcular a menor célula de arredondamento e eliminar atualizações estritamente interiores à célula, sem presumir positividade ou mudar a ordem aritmética.

Os certificados passam por negação e atualizações por zero, interseções das fronteiras e estados salvos. A fronteira normal de uma conversão finita com exatamente dois braços recebe sua magnitude mínima; fronteiras com overflow ou sem certificado continuam conservadoras. A prova não é copiada para a ramificação irmã. O formato persistido inclui esse certificado e hashes novos; uma retomada do estado antigo foi recusada, como registra `rejected-resume.log`.

## Validação

Build e 21 integrações passaram, sem falhas ou testes ignorados. A regressão nova cobre ambos os sinais, zeros assinados, limites e isolamento de contexto: 122.888 comparações nativas, zero divergências. Sua expressão caiu de 226 para 160 caracteres. O teste de estados salvos comprova a restauração da magnitude não nula. Uma primeira execução encontrou uma expectativa de tamanho desatualizada no teste: a versão usada como anterior também recebia o novo certificado. A referência foi corrigida para representar a estratégia anterior, mantendo a comparação nativa; o conjunto completo foi reexecutado.

A região positiva da coordenada foi recompilada, produzindo 19.532 caracteres e 1.028 comparações contra referência CPU carregada novamente, sem divergência. As 43 regiões anteriores foram recompiladas com os novos hashes, importando somente a geometria auditada; 11.180 comparações dos arquivos emitidos não apresentaram divergências. Sete expressões distintas estão preservadas nesta pasta e identificadas em `artifact-map.json`.

## Resultado e limites

A correção não reduziu a coordenada inteira: permanecem 3.248.163.585.496 caracteres lógicos, com emissão completa recusada pelo limite. `coordinate-run.json` mede o crescimento em cada produtor e dependência do primeiro caminho. Há 183.650.944 padrões cobertos por regiões emitidas e 3.847.075.200 ainda sem artefato; as comparações de paridade são amostrais, não exaustivas nessa cobertura.

O resultado continua parcial: um token, posição 0, dimensão 2. Coordenada inteira, último token com comprimento variável, todas as dimensões e comparação atual de compilação sequencial/paralela no Colab permanecem pendentes. A tradução numérica experimental no Colab falhou ainda na comparação CPU (53 divergências); não há validação CUDA ou ganho de velocidade estabelecido para ela.

A próxima investigação deve verificar como os certificados numéricos chegam ao fechamento de cada dependência depois da seleção de caminhos. `CoherentPaths.literal` simplifica palavras e álgebra, mas não reexecuta o fechamento numérico com os certificados de cada produtor selecionado. As identidades de expressões originais não podem ser reutilizadas após seleção; qualquer nova propagação deve preservar apenas provas válidas no contexto e separar as condições de dispatch das condições da folha.

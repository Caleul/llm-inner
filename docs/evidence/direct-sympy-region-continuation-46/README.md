# Continuação: 46 tentativas acumuladas

A compilação foi retomada com a mesma identidade de fonte/backend/checkpoint. As 24 tentativas adicionais concluíram 12 regiões novas. O código de produção não mudou; a suíte de 21 integrações da revisão anterior continua sendo seu gate. Não houve novo benchmark controlado.

## Artefatos e paridade

- 30 regiões completas; 181.559.808 padrões cobertos de 4.030.726.144, aproximadamente 4,50%.
- Avanço adicional: 2.460.160 padrões. Ainda faltam 3.849.166.336.
- 7.800 comparações nativas dos arquivos salvos com checkpoint CPU carregado novamente; nenhuma divergência. São amostras regionais, não validação exaustiva de todos esses padrões.
- Seis expressões distintas, nove retângulos coalescidos, 634.272 caracteres de corpos sem repetição. Quatro expressões já estavam preservadas no registro anterior; as duas novas estão nesta pasta. Todos os caminhos duráveis e SHA estão em `coordinate-parity.json`.

A prova é de uma coordenada regional, dimensão 2, posição 0, um token. A coordenada inteira continua sem artefato e sem paridade final; comprimento variável/último token e todas as saídas também permanecem pendentes.

## Diagnóstico do crescimento

A região X1 de rank Half -17049 a -16988 e X2 de -31743 a -23808 excede o orçamento de emissão de 1 MiB. `diagnostic-parent-report.json` registra 17 produtores, três updates eliminados, cinco contextos divididos e três caminhos processados. O último caminho tem 1.023.657 caracteres de corpo e 533.823 de condições, com quatro decisões.

Um teste isolado elevou a emissão a 16 MiB e o orçamento CAS a 32 MiB, mantendo 64 caminhos. `larger-parent-report.json` registra o encerramento e coleta do worker no limite de 45 segundos; nenhum artefato foi emitido. Não se atribui esse tempo a um estágio específico sem perfil. Essa execução não altera o estado principal nem omite o domínio pendente.

Subdividir a mesma região permite gerar corpos idênticos menores, reunidos posteriormente por cobertura exata. O próximo passo é melhorar a propagação das condições e a simplificação nas normalizações, para evitar que essa descoberta dependa de tantas subdivisões. Aumentar apenas o orçamento não demonstrou resolver o caso testado.

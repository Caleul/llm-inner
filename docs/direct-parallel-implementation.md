# Compilação direta paralela

## Estado e integração

Implementação integrada sobre `133baf4`, incluindo os domínios disjuntos e as preimagens com sinal concluídos simultaneamente. A geração já em andamento não foi reiniciada. As alterações manuais de `AGENTS.md` e o plano anterior de savepoints foram preservados.

O contrato permanece: descoberta de Safetensors/config, substituição completa dos produtores e emissão de Rust direto. Não são compartilhadas ativações, cálculos, closures ou uma representação de execução do modelo. Metadados descobertos e uma prova de domínio numérico por camada são preparados uma única vez para o pool; os pesos continuam sendo lidos em páginas limitadas.

## Melhorias aplicadas

- Pool de workers com fila dinâmica e montagem por ordem canônica, independente da ordem de conclusão.
- Unidades por posição e pela variante de comprimento exigida pela redução da softmax. A posição 2 mantém a separação entre `n==3` e `n>=4`.
- Partições opcionais do domínio finito F16 de uma coordenada da entrada. As 63.487 representações numéricas ordenadas incluem ambos os padrões de zero na mesma partição; todos os 63.488 padrões finitos permanecem cobertos. Cada tarefa reconstrói a substituição inteira sob suas próprias condições. Particionar não elimina entradas nem fixa prompts/comprimentos.
- Cache de páginas dos bytes originais dos pesos, com limite e descarte por worker. Não retém matrizes decodificadas nem resultados de forward.
- Cache limitado de hashes lexicais puros e buffering limitado dos fragmentos usados pelo SHA256. Nenhum resultado de inferência por domínio é memorizado.
- Descoberta e provas preliminares executadas uma única vez antes das tarefas. Certificados contêm somente o limite de score por camada, não valores por coordenada.
- Limites de heap por worker, monitoramento de RSS global, orçamento atômico compartilhado de bytes de saída e verificação de espaço em disco para fragmentos e montagem.
- Verificação SHA256 dos fragmentos durante a montagem, cancelamento e descarte de temporários em falhas. Uma unidade parcial nunca é admitida como modelo completo.
- Exclusividade do caminho de publicação e verificação de imutabilidade do checkpoint antes e depois da geração.

A identidade do checkpoint usa SHA256 para config/index e identidade de stat para os payloads. Detecta substituições/modificações comuns; não é um checksum completo dos pesos nem um contrato de retomada. Encerramento abrupto pode deixar um lock, que deve ser inspecionado antes de remoção manual. Não foi implementada retomada por savepoints.

## Uso e recursos

`compile:direct-rust` agora usa o pool por padrão: até quatro workers, quatro partições, orçamento de 2.048 MiB para o processo, heap de 256 MiB por worker, páginas de até 16 MiB por worker e máximo de 8.192 MiB de fonte gerada. A quantidade efetiva de workers é reduzida quando o orçamento de memória não comporta o pedido.

```sh
npm run compile:direct-rust -- CHECKPOINT PYTHON DIMENSION OUTPUT.rs --validate
```

Para aumentar recursos explicitamente:

```sh
npm run compile:direct-rust -- CHECKPOINT PYTHON DIMENSION OUTPUT.rs --workers 8 --partitions 16 --memory-mib 8192 --weight-cache-mib 128 --max-output-mib 8192 --validate
```

`--heap-mib` configura o heap de cada worker; `--partition-coordinate` escolhe uma coordenada dentro da largura descoberta. `--partitions 1` preserva as fronteiras naturais; `--workers 1` executa a mesma partição serialmente. `--serial` mantém o percurso original e aceita os limites de páginas/saída, mas não opções do pool.

O limite de saída encerra a compilação com erro, sem publicar um resultado truncado. O monitor de RSS é periódico, não uma quota do sistema operacional. O espaço livre é verificado inicialmente para até duas vezes o orçamento de fonte; o consumo de outros processos pode mudar depois. O orçamento não se aplica aos processos externos de descoberta PyTorch nem ao `rustc` posterior. As reduções, arredondamentos e provas numéricas mantêm a associação original.

## Mapa de testes e evidências

O inventário estático está em `direct-parallel-test-inventory.json`; os resultados e falhas estão em `direct-parallel-validation.json`. Nomes declarados são inventário; aprovação exige uma execução registrada.

| Garantia | Cobertura |
| --- | --- |
| Reduções, arredondamento, raízes, exponenciais e SiLU | `direct-rust-stream`, `direct-round-preimage`, `direct-flat-substitution`, `fixed-f16-*` existentes |
| Condições herdadas, regiões disjuntas, sinais e isolamento de irmãos | `direct-disjoint-domains` e `direct-flat-substitution` existentes |
| Streaming, bounds, geometria e eliminação de pesos zero | `direct-stream-buffer`, `direct-source-bounds`, `direct-finite-model-proof`, `direct-flat-zero-mlp` existentes |
| Cobertura das partições e ordem de softmax | `direct-parallel`: partições com 1/2/3/4/7/16/63.487 unidades e fronteira de comprimento |
| Fidelidade sob partições | Rust executado sobre todos os 63.488 padrões F16 finitos, em 1/3/4/7 partições e comprimentos 1/2/4 |
| Leitura paginada | Todos os padrões F16 finitos, sinais de zero, não finitos rejeitados, eviction e separação de shards |
| Cache/digest | SHA256 idêntico, capacidade limitada, Unicode e pares substitutos divididos entre fragmentos |
| Pool e publicação | Conclusão fora de ordem, falha/saída inesperada de worker, cancelamento, RSS, corrupção, omissão, limite compartilhado e exclusividade |
| Integração com modelo | Checkpoint completo de saída zero: bytes serial/paralelo, um/quatro workers, Rust executável e paridade PyTorch em matrizes e comprimentos variados |
| Regressões gerais | Suíte completa antes/depois, com o mesmo corpus e o mesmo Python, incluindo referências legadas F16/IR |

O teste de empate Metal no servidor tinha uma pré-condição implícita: esperava zero espera sem aguardar a recuperação anterior terminar. A suíte sob carga revelou espera positiva; a mesma falha foi reproduzida no baseline atrasando somente as respostas de warmup do worker de teste. A correção mantém a exigência de zero espera, aguarda estado `idle` e usa respostas deliberadamente atrasadas para tornar o cenário determinístico. O código do servidor não foi alterado.

## Desempenho medido e limites

- Prefixo real com 450.015 bytes contabilizados: 23.440 ms antes e 13.243 ms depois, redução de 43,5%. Os 393.171 bytes persistidos comuns são idênticos; contadores de folhas, inspeções e eliminações também coincidem. É um experimento interrompido, não paridade do logit final.
- Quatro jobs independentes de prefixo real: 56.068 ms com um worker e 14.679 ms com quatro, ganho de 3,82×, com bytes idênticos. Esse experimento usa jobs iguais; não mede o balanceamento nem a amplificação de trabalho provocada por partições reais.
- No prefixo otimizado, foram 11 leituras de página e 72.896 acertos. O checkpoint diagnóstico é minúsculo; isso não estima o cache de um checkpoint grande.

Os experimentos foram executados com a geração anterior ainda ativa. Não se deve multiplicar os ganhos para prometer um tempo de compilação completo. Particionar pode duplicar provas locais e impedir fusões entre fronteiras; o custo varia com o checkpoint. Mais workers não resolve a explosão de expressões nem os limites de compilação de uma função gigante pelo `rustc`.

A integração de saída zero valida a montagem e o pipeline executável completo, mas não prova o logit final de um checkpoint com atenção/MLP não nulos. Essa validação continua pendente até uma geração completa viável. As reduções numéricas existentes e as comparações bit a bit exercitam as partes já alcançáveis.

## Resultado final da validação

- Baseline com corpus/Python equivalentes: 480 testes, 462 passaram, 15 falharam, 3 foram pulados.
- Implementação integrada: 497 testes, 479 passaram, as mesmas 15 falharam, os mesmos 3 foram pulados.
- Todos os 17 testes novos passaram. Nenhum teste anteriormente aprovado deixou de passar.
- Typecheck e verificação de whitespace passaram. Os 21 testes do servidor passaram após tornar explícita a pré-condição de recuperação.
- Falhas anteriores: 11 evidências Gemma4 ausentes/incompatíveis, 2 divergências da identidade Python/macOS fixada, 1 exemplo de documentação ausente e 1 prompt do controlador ausente. A suíte geral continua retornando código 1 por essas falhas; elas não foram mascaradas nem removidas.
- Commit na feature branch existente. Push depende de um remote: nenhum está configurado neste repositório.

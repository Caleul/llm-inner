# Fatoração exata com zeros de ambos os sinais

A fatoração anterior recusava todas as ilhas algébricas que poderiam produzir `-0`. Isso bloqueava identidades válidas como `3*A + 7*A → 10*A` quando A era um produtor fechado que admitia ambos os zeros.

A nova admissão continua exigindo um certificado independente de exatidão F64 para **cada operação original e proposta**. Quando o certificado anterior de ausência de zero negativo não basta, a regra adicional exige:

1. Um único produtor numérico fechado, com domínio e quantum conhecidos.
2. Polinômios iguais, homogêneos e de grau um, com coeficiente não zero.
3. Resultado zero com o mesmo sinal nas avaliações de A=`+0` e A=`-0`, seguindo a ordem original.
4. Menos ocorrências do produtor e menor custo expandido.

A igualdade linear e a exatidão excluem zero para todas as entradas não zero. Restam somente os dois casos de zero, verificados durante a compilação. Não há enumeração das respostas do modelo nem regra que autorize álgebra irrestrita. `3*A - 7*A`, por exemplo, continua recusado porque seu zero positivo não equivale ao zero negativo da forma `-4*A`. Cancelamentos totais, ilhas não lineares e variáveis independentes não recebem esta prova adicional.

O ensaio nativo compara quatro formas fatoradas em todas as entradas Half de [-1, 1], incluindo os dois sinais de zero: 122.888 comparações, nenhuma divergência. Os testes conservam barreiras de precisão, funções desconhecidas e isolamento de contextos.

## Efeito observado e artefatos reais

Esta regra **não reduziu o crescimento global do Llama atual**: a composição continua com 2.296.124.090.440 caracteres lógicos. O teste registra esse resultado, sem afirmar aceleração ou conclusão global.

Foi também testada uma geometria mais ampla que a continuação anterior: as quatro combinações de sinais em que ambas as entradas têm magnitude entre 32 e 65.504. Os updates são provados irrelevantes antes de sua expansão, e cada compilação alcança a mesma expressão sobre X1/X2, sem aliases de compilação ou checkpoint de runtime. O corpo de **17.552 caracteres**, com duas ramificações, é idêntico nos quatro domínios. Esse resultado provém das provas de eliminação de dependências já existentes, e não é atribuído à nova regra linear.

| Evidência atual | Resultado |
| --- | ---: |
| Padrões nas quatro regiões amplas disjuntas | 507.510.784 |
| Comparações nativas dessas regiões | 32.784 |
| Divergências observadas | 0 |
| Comparações da quinta região, compilada separadamente | 1.028 |
| Divergências observadas na quinta região | 0 |
| Padrões nas cinco regiões disjuntas | 507.756.800 |
| Cobertura geométrica desta evidência | 12,59715450417859% |
| Padrões ainda fora dessas cinco regiões | 3.522.969.344 |

`artifact-map.json` preserva os domínios e hashes das expressões efetivas. `large-regions-parity.json` e `coordinate-region-parity.json` registram a execução nativa contra o checkpoint recarregado. O arquivo `probe_regions.py` permite reproduzir as quatro regiões amplas. A quinta expressão permanece com 32.224 caracteres e o mesmo hash de uma expressão anterior, após nova compilação e validação com as fontes atuais.

As cinco regiões são disjuntas entre si, mas se sobrepõem a regiões das versões anteriores. Portanto, sua cobertura não foi somada à árvore anterior. Nenhum estado numérico salvo de outra identidade foi reutilizado. A mudança de fonte invalida a retomada numérica da árvore anterior; somente sua geometria pode ser importada em uma execução nova.

## Regressões e limites

Build concluído. Os seis testes de fatoração passaram; a suíte completa passou nas 24 integrações, sem falhas ou testes ignorados. Três cenários de sincronização usavam A+A para manter duas ocorrências de seletores. A nova fatoração eliminou essa duplicação antes da sincronização, invalidando a expectativa sobre qual passe faria a redução. Os cenários passaram a usar produtos para continuar exercitando a sincronização independentemente. A prova de transferência de dtype e de identidade da raiz permanece, e o cenário numérico compara 507.904 resultados nativos, sem divergências. As execuções inicialmente falhas estão preservadas nos logs, com a causa e a correção delimitadas.

O mapa global de testes conserva as entradas anteriores e acrescenta esta execução. Os hashes das fontes e do checkpoint estão no relatório de validação.

O escopo continua posição 0, dimensão 2, um token. A paridade dos artefatos do checkpoint é amostral e bit a bit, sem tolerância; os números de cobertura geométrica não são contagens de entradas executadas. A coordenada inteira, o comprimento variável, múltiplos tokens, o vetor completo e a emissão Rust continuam pendentes. As expressões são strings matemáticas, seguindo a instrução humana mais recente; JSON conserva evidências e domínios. Não foi feito benchmark de desempenho nem execução CUDA desta versão.

# Paralelismo SymPy e validação no Colab — 2026-10-03

O caminho atual usa strings matemáticas. JSON contém somente métricas,
identidades e manifestos de estados; não representa as expressões.

A sessão `llm-inner-sympy`, criada exclusivamente pelo Access Broker MCP
(`colab_cli`), forneceu 12 CPUs, aproximadamente 167 GiB de RAM e uma
NVIDIA A100 de 80 GB, CUDA 13.0 e PyTorch 2.11.0+cu130.

Os blocos executam substituições com parênteses, `factor()` e `simplify()`
até estabilizar. A combinação em pares substitui o contexto do bloco da
direita pelo resultado do bloco da esquerda. Não reagrupa somas F32:
`((16777216 + 1) + -16777216)` deve continuar dando zero, enquanto
`16777216 + (1 + -16777216)` dá um. Os quatro acumuladores e a combinação
final do adaptador permanecem na ordem original. Bifurcações continuam
recebendo os domínios e as condições de sua própria ramificação.

Os trabalhadores recebem caminhos de arquivos, não cópias das expressões
pela fila de processos. A admissão considera o tamanho dos operandos,
a memória do processo principal e uma reserva por tarefa. No Colab/Linux,
o controlador observa o RSS agregado e cancela trabalhos que excedem o
orçamento. RSS agregado pode contar novamente páginas compartilhadas
pelo fork: é uma estimativa conservadora, não uma medição de RAM física
exclusiva. No macOS, a observação disponível é o pico do processo principal;
a validação do monitor dos filhos foi feita no Linux.

O paralelismo implementado abrange os blocos das reduções. A construção
das dependências anteriores e o fechamento numérico dos produtores ainda
seguem o percurso atual; o grafo completo não é executado em paralelo.

Também foram removidas conversões F32 redundantes antes de substituir
produtos de dois Half finitos, limitou-se o cache de ponto fixo a 16 MiB e
corrigiu-se a gravação dos estados: produtores imutáveis já publicados
não são codificados e hasheados novamente. Novos arquivos são escritos
e hasheados em blocos de 1 MiB, com publicação atômica e `fsync`.

A primeira tentativa no Colab encontrou `clang++` ausente. A dependência
foi instalada e os testes nativos passaram antes da compilação. Uma
entrada inicial sem a otimização equivalente de varredura foi encerrada,
evidência preservada e comparação reiniciada com o mesmo backend nos
dois modos. Nenhum estado macOS foi reutilizado no Colab; a identidade
estrita inclui 14 fontes semânticas, checkpoint, domínio e ambiente de
referência.

| Medida no mesmo Colab, mesma coordenada e limite | Sequencial | Paralelo |
|---|---:|---:|
| Tempo total observado | 232,200 s | 238,424 s |
| Pico de RSS agregado amostrado | 29,244 GiB | 33,572 GiB |
| Dependências completas | 15 | 15 |
| Última dependência | `model.layers.0.up:0` | `model.layers.0.up:0` |
| Coordenadas finais admitidas | 0 | 0 |

O paralelo ficou 2,68% mais lento neste ensaio único. O checkpoint de
largura dois oferece poucos blocos independentes e o trabalho sobre
strings continua dominante. O modo sequencial permanece como padrão.
Não há estimativa defensável de tempo para concluir o modelo inteiro
com base nesse prefixo.

As duas execuções foram interrompidas pelo mesmo limite de expressão:
a composição de ativação e projeção up exigiria 10.534.977.939 caracteres,
além dos 9.663.676.416 permitidos. Isso preserva os produtores completos,
sem admitir uma coordenada incompleta. A ativação já tem 9.218.105.754
caracteres; esses resultados não demonstram resolução do crescimento.
A próxima tarefa continua sendo simplificar essa composição e fechar a
primeira coordenada, antes de avançar às outras.

A GPU executou os lotes numéricos compatíveis. Para 888.832 produtos Half,
a mediana de 15 amostras após aquecimento foi aproximadamente 2,04 ms na
CPU e 0,084 ms na GPU com dados residentes; incluindo transferências,
2,69 ms. Não houve divergências. SiLU também coincidiu em todos os 19.458
valores Half no domínio certificado `[-3/128, 3/128]`. Esses testes não
admitem CUDA para outros domínios, primitivas ou árvores de redução.
`factor()` e `simplify()` continuam na CPU.

Localmente: build aprovado, 85 testes Python aprovados, oito testes de
integração SymPy aprovados. A regressão geral teve 599 testes: 581
aprovados, 15 falhas preexistentes com os mesmos nomes e três pulados.
Os quatro testes legados `direct-json-model` mantêm a exclusão registrada
no baseline. O teste adicional de admissão reduziu oito trabalhadores
solicitados a um quando só um cabia no orçamento, e verificou encerramento
dos filhos após erro no trabalhador.

A referência Colab coincidiu com a referência local em 864 valores de
60 entradas. O teste da coordenada com primitivas preservadas coincidiu
em 60 casos, incluindo comprimentos 1, 2, 3, 4 e 8. Isso é diagnóstico;
não é paridade do artefato final completamente expandido.

A verificação dos arquivos emitidos terminou nos dois modos: 15 arquivos
por execução, 300 comparações em 60 entradas e nenhuma divergência. Os
15 produtores são idênticos byte a byte entre sequencial e paralelo. Os
contadores de passes CAS diferem, como esperado pelas substituições
adicionais dos blocos; o tamanho e o conteúdo das expressões coincidem.
Essa verificação reutiliza subárvores literais exatas somente no avaliador
de testes; não introduz cache ou intermediários no artefato de runtime.

A primeira coordenada completamente fechada permanece pendente. Não
foram compiladas coordenadas adicionais, nem declarado o modelo
compilado a partir desses prefixos.

Os estados e as expressões completas dos 15 produtores estão preservados
localmente em `artifacts/direct-sympy-colab`. Os novos manifestos são os
manifestos originais produzidos no Colab. O conteúdo literal foi
reconstruído por identidade SHA256 a partir de arquivos já existentes
localmente: 12.237.174.562 bytes verificados, sem duplicá-los em disco e
sem reutilizar um estado antigo durante a compilação. A identidade Linux
desses estados continua incompatível com retomada no macOS.

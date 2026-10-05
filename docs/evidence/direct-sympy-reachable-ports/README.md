# Composição SymPy: dependências efetivas e registro compacto

O percurso efetivo continua em strings matemáticas. JSON nos diretórios `remote/`
é apenas relatório/descritor. O teto acumulado de expressões e condições é
2.147.483.648 bytes; a RAM do ensaio é monitorada separadamente, com limite de
8 GiB por árvore de execução. Toda carga pesada foi executada via Colab CLI pelo
Access Broker. O runtime disponível tem duas CPUs, 13.605.830.656 bytes de RAM e
PyTorch 2.11.0+cpu, sem CUDA; nenhuma aceleração GPU é atribuída a estes resultados.

## Correções

- Composições ignoram portas ausentes da saída escalar. Interfaces disjuntas
  impedem que uma substituição posterior introduza outra porta do bloco direito.
  O teste inclui X2/X20 para não confundir prefixos de identificadores. Toda
  substituição efetiva continua passando por factor/simplify até estabilizar.
- O runner da arquitetura instala o backend de scans equivalentes já existente.
  Fatos e domínios não mudam; a verificação final dos matches continua completa.
- A prova de finitude da arquitetura autoriza retirar apenas arredondamentos
  F32 de produtos de dois Half finitos. São produtos exatos em F32. Somas,
  reduções, conversões de armazenamento e multiplicações sem essa prova mantêm
  suas fronteiras numéricas e sua ordem original.
- O registro de regiões não elementares analisa a gramática de seu envelope
  compacto. Ele precisa apenas reconhecer o tipo da raiz, sem reconstruir a AST
  de todas as cópias de produtores já certificados no mesmo contexto. A análise
  de payload/width de palavras continua usando a AST original. Esta otimização
  não altera expressões nem fornece fatos numéricos novos.
- O modo de rastreamento registra substituições e stacks tanto nos filhos como
  no processo coordenador. Serve para distinguir crescimento textual, CAS,
  registro de metadados e validação.

## Resultados já concluídos

Os ensaios `ports`, `scans` e `half` partiram de diretórios novos, sem reutilizar
savepoints incompatíveis. Para um token, há 12 preparações, 11 composições e
quatro expressões de trabalho: 48 comparações de logits originais, zero diferenças.
Foram ignoradas 56 portas nesse caso e 272 nas composições de dois tokens.

| Ensaio | Composição de 1 token | Caracteres das 4 expressões | RAM de pico da árvore |
| --- | ---: | ---: | ---: |
| Scans e portas alcançáveis | 2,673 s | 839.207 | 662.130.688 bytes |
| Produtos Half exatos | 2,507 s | 804.567 | 675.557.376 bytes |

A redução textual é de 4,13%. Estes são ensaios únicos; a diferença de tempo
não demonstra um ganho sustentado. Em ambos, a composição de dois tokens
concluiu 29 jobs, mas excedeu 8 GiB ao registrar/validar o resultado. Não houve
artefato completo validado para dois tokens nesses ensaios.

A regressão `half` passou 23 testes. A avaliação dos blocos alterados contra a
captura original CPU ARM64/PyTorch 2.12.1 passou 240 comparações, sem divergências,
para comprimentos 1, 2, 3, 4 e 8. Não é paridade da função Rust final, nem cobre
os comprimentos 5, 6 e 7 da referência original.

O ensaio original com teto de 2 GiB percorreu todos os comprimentos 1..8.
Somente 1 token emitiu quatro expressões de trabalho; as quatro expansões
numéricas atingiram seus deadlines. Dois tokens atingiram o teto de RAM;
3..8 atingiram deadlines. O sumário completo está em
`remote/baseline/architecture-2g/summary.json`.

A regressão após a correção de registro compacto passou 59 testes em 151,392 s,
incluindo a composição em pares para um token (20 comparações) e o forward
completo no runtime CPU PyTorch 2.11.0 do Colab: 96 casos de comprimentos 1..8,
1.728 logits de todas as posições e 384 logits finais, sem divergências.
Esses testes usam a referência nativa remota, separada da captura original.

O ensaio `safe` validou novamente os quatro resultados de um token: 48 bits,
zero diferenças. Sua composição levou 1,082 s, ainda sem benchmark repetido.
Dois tokens concluíram os 29 jobs e reservaram 156.684.102 bytes de strings,
com pico observado de 1.020.321.792 bytes na composição. A árvore inteira
atingiu 8.598.380.544 bytes depois disso e foi interrompida pelo monitor.
A análise do próximo ponto de alocação permanece aberta; esta redução de
metadados não resolveu por completo a execução de dois tokens.

O primeiro ensaio de rastreamento `registration-run` foi interrompido: o watchdog
iniciado no pai antes do fork interferiu no watchdog dos filhos. O código atual
captura a pilha do pai por SIGUSR1, sem iniciar esse thread antes do fork; o
ensaio `safe` confirmou novamente o funcionamento das composições. A regressão
59/59 rodou sem o rastreador defeituoso e permanece válida.

A captura diagnóstica agora identifica a máquina real. O smoke no Colab gerou
96 casos, comprimentos 1..8, com backend `cpu-x86_64-eager` e PyTorch 2.11.0+cpu.
Não declara mais ARM64 para um processo x86_64 e não substitui a referência
original CPU ARM64/PyTorch 2.12.1.

A captura de stack em `remote/parent-stack/` localizou o próximo pico na AST
do verificador, em `program()`/`syntax()`, depois da composição. O avaliador
`WorkingProgram` agora compila envelopes de literais certificados e os avalia
por chamadas preguiçosas, com cache exclusivo de cada entrada. Preserva ramos
inalcançáveis, ordem e zeros com sinal; não entra nos arquivos emitidos ou Rust.

O ensaio `working` passou os cinco testes iniciais e concluiu quatro strings
para cada comprimento 1 e 2. Cada vetor passou 48 comparações bit a bit contra
a captura original, sem divergências. Para dois tokens, foram 52,181 s de
execução, 43,726 s de composição e pico de 1.004.589.056 bytes na árvore. O
ensaio anterior terminava após 77,011 s com pico de 8.598.380.544 bytes e sem
paridade da expressão. A comparação demonstra remoção do bloqueio de memória,
mas não é um benchmark repetido de velocidade.

A regressão do verificador e runner passou depois seis testes em 23,738 s,
incluindo a rejeição de paridade quando não existe caso de referência para o
comprimento. Zero comparações nunca autoriza declarar paridade. O arquivo
literal continua completamente substituído; os aliases existem somente no
verificador. Primitivas numéricas ainda permanecem nas strings de trabalho.

## Conclusão exigida

Na fase registrada neste diretório, a função Rust integral ainda não havia sido emitida. As expressões de trabalho mantêm
primitivas numéricas que precisam ser expandidas, simplificadas e validadas.
A referência original também precisa cobrir todos os comprimentos antes de
admitir a paridade final. Nenhum resultado deste diretório substitui esses gates.

`remote-manifest.json` contém tamanho e SHA-256 dos arquivos recuperados do
Colab. O mapa geral de testes conserva os resultados anteriores e acrescenta
estes ensaios, distinguindo paridade de componentes e do artefato final.

O fechamento posterior está em `../direct-sympy-scalar-rust/README.md`, com
função Rust efetiva, todos os comprimentos e paridade contra a referência ARM64.

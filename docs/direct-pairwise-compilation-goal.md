# Meta corrigida: compilação por composição de operadores

Compilar Safetensors/config em expressões diretas especializadas: incorporar
pesos, compor operadores em pares e aplicar fatoração/simplificação até
estabilizar antes e depois de cada substituição, preservando tipos, ordem
numérica e contexto de cada ramificação. O JSON estrutural é a fonte de verdade; texto é ponte para o SymPy e diagnóstico.
Eliminar todas as interfaces de bloco e dependências do checkpoint no
resultado final. Usar paralelismo limitado pela memória, GPUs do Colab via
Access Broker para trabalho numérico compatível e CPU para SymPy quando
não houver backend CUDA equivalente. Entregar exclusivamente os logits da última posição, que preveem o próximo
token após toda a entrada, para todos os comprimentos descobertos. Preservar
todas as dependências causais alcançáveis de atenção. Preservar o mapa de testes, integrar alterações,
commitar e enviar para o remoto existente.

## Fluxo e contagem

Para F1 → F2 → F3 → F4, preparar em paralelo G12(X)=F2(F1(X)) e
G34(H)=F4(F3(H)), cada um com seus pesos incorporados e suas simplificações
estabilizadas. Em seguida compor G34(G12(X)) e estabilizar de novo. H só
existe como interface de compilação e desaparece nessa composição.

O planejamento parte das dependências escalares distintas alcançáveis
da coordenada. Reuso de um produtor não cria outro produtor. O inventário
original, composições de blocos, passes do CAS e ocorrências expandidas são
contagens distintas. Não inferir o número de branches da quantidade de pesos.

Uma composição balanceada preserva a ordem F1, F2, F3, F4; não transforma
reduções numéricas em somas balanceadas. Simplificações só são adotadas com
equivalência para o dtype e o contexto efetivos. Não publicar resultado com
interfaces, primitivas pendentes, cobertura parcial ou paridade não validada.

## Gates

- Provar composição sequencial versus pares com pesos, arredondamentos e
  condições, incluindo zeros com sinal e entradas de fronteira.
- Integrar a composição no percurso efetivo do checkpoint, com contagem
  de produtores e eventos de composição; um exemplo isolado não conclui a meta.
- Executar todas as coordenadas/logits da arquitetura completa, conforme
  autorização atual, e investigar divergências sem suprimir outros ensaios.
  A publicação final continua exigindo paridade de cada artefato.
- Emitir o artefato efetivo e validar seu resultado, sem trocar a entrega
  por uma estimativa de expansão ou por um executor de intermediários.
- Manter teto de 512 MiB para expressões/condições e medir RAM/GPU à parte.

## Estado na retomada

O vetor e uma coordenada completa ainda não foram entregues. Há provas
regionais e testes de componentes. Uma alteração local de cancelamento
de bitcasts condicionais ficou sem validação; será validada ou retirada
antes de integrar o novo percurso. A nova estratégia deve ser confrontada
com o checkpoint efetivo, não só com testes algébricos.

## Implementação iniciada

`helpers/direct_sympy_operators.py` prepara operadores vetoriais e os
compõe em pares ordenados. As projeções do adaptador Llama usam esse
percurso quando o paralelismo está habilitado. Interfaces não herdam
limites numéricos da entrada; elas desaparecem por substituição. Os
savepoints incorporam o hash desse backend e rejeitam fontes incompatíveis.

A integração ainda não compõe normalização, atenção, residual e MLP como
uma sequência completa de blocos independentes. Essa ampliação e a redução
do crescimento na normalização continuam pendentes, junto com o artefato
final. Não confundir a paridade da expressão com primitivas numéricas ou
das projeções CUDA com a conclusão da primeira coordenada.

## Ampliação autorizada em 2026-10-05

O planejador cobre pre/post/final RMS, Q/K/V, RoPE, escores causais, softmax,
contexto, projeção O, resíduos, gate/up, SiLU, produto e down. Cada camada
usa interfaces vetoriais próprias; a composição em pares elimina esses
intermediários. O lote percorre todos os comprimentos descobertos e os quatro
logits da última posição, com limites separados de expressões, RAM e tempo.
Expressões com primitivas numéricas conservadas continuam explicitamente
marcadas como trabalho intermediário, sem declaração de conclusão.

## Recorte e execução remota corrigidos

Selecione `[(length-1)*vocab, length*vocab)` antes da composição. A poda
retroativa elimina produtores exclusivos dos logits anteriores, sem remover
os tokens que fornecem chaves e valores à atenção da última posição.
Os índices publicados e os bits esperados são sempre os da última linha.

A compilação pesada e a expansão numérica executam no Colab via Access
Broker, incluindo quando a execução remota não acelera o processo. Localmente
ficam edição, empacotamento, coordenação e verificações leves.

O snapshot `direct-target-discovery-v1` incorpora a política do forward
original CPU arm64 e vincula configuração, Safetensors, descoberta e fontes
do compilador por SHA256. O host remoto x64 não altera essa política.
Estados incompatíveis são recusados e as provas de finitude são refeitas.
O snapshot não contém pesos de runtime, ativações ou respostas.

A validação CPU do recorte cobriu 96 entradas e 384 logits, com zero diferenças
de bits. A A100 também cobriu esses 384 logits. A poda reduziu produtores de
72 para 45 (2 tokens) e de 336 para 105 (8 tokens). A composição textual ainda
pode exceder RAM/expressões; paridade de templates ou da avaliação de um
JSON compartilhado durante compilação não equivale à entrega literal final.

Sessão em execução: `llm-inner-final-logits-20261005`; A100 40 GB, 12 CPUs,
89.6 GB de RAM, teto de expressões/condições 512 MiB. As coordenadas JSON
fechadas são processadas com quatro trabalhadores, sem executor no artefato
final. O lote textual compara sequencial versus pares no mesmo ambiente.

A meta continua aberta até emissão e paridade do produto final.

## Fechamento elementar remoto observado

A sessão de simplificação reutiliza provas estruturais somente sob os mesmos
fatos relevantes de ramificação, por coordenada. Isso remove a reanálise de
produtores concluídos a cada substituição; não preserva intermediários no produto.
No Colab foram fechadas as 32 coordenadas (quatro logits, comprimentos 1–8),
sem primitivas pendentes, com 384 comparações diagnósticas sem divergência.
O tempo por coordenada ficou entre 7,4 e 215,6 s, RSS observado até 646,5 MiB
por trabalhador. A emissão literal continua recusada pelo teto de 512 MiB:
nenhum artefato final foi publicado, e a paridade desse artefato não está provada.
As evidências estão em `docs/evidence/direct-sympy-whole-architecture/`.

# Meta corrigida: compilação por composição de operadores

Compilar Safetensors/config em expressões diretas especializadas: incorporar
pesos, compor operadores em pares e aplicar fatoração/simplificação até
estabilizar antes e depois de cada substituição, preservando tipos, ordem
numérica e contexto de cada ramificação. O formato intermediário é livre.
Eliminar todas as interfaces de bloco e dependências do checkpoint no
resultado final. Usar paralelismo limitado pela memória, GPUs do Colab via
Access Broker para trabalho numérico compatível e CPU para SymPy quando
não houver backend CUDA equivalente. Entregar uma coordenada completa com
paridade bit a bit e depois o vetor completo para uma e múltiplas entradas
e comprimentos de tokens. Preservar o mapa de testes, integrar alterações,
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
- Validar uma coordenada completa antes de despachar outras dimensões.
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

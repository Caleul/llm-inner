# Continuação da coordenada e comparação atual no Colab

A fonte numérica é o commit `4f2240977a970b3e25afdfc1d749ca44a287b08c`. Este registro não altera o forward nem o compilador. As expressões são strings matemáticas; JSON registra estados, medições e paridade. O estado local foi retomado somente após conferir a identidade exata dos módulos, checkpoint e backend.

## Avanço efetivo local

A execução acrescentou 48 tentativas, das quais 43 emitiram regiões completas e cinco exigiram divisão. O estado acumulado tem 56 tentativas, 51 regiões emitidas e seis regiões pendentes. A cobertura passou de 176.639.488 para 185.004.032 padrões, acréscimo de 8.364.544 padrões. Isso corresponde a 4,58984375% do domínio; 3.845.722.112 padrões ainda não possuem artefato.

Os arquivos emitidos foram relidos, compilados nativamente e comparados com o checkpoint de referência carregado novamente: 13.260 casos amostrais, zero divergências. Há sete expressões distintas, preservadas em `artifact-map.json`, com 644.346 caracteres de corpos únicos. A auditoria mantém o domínio inteiro no estado; regiões pendentes não contam como saída pronta.

## Sequencial versus dois trabalhadores no Colab

O Access Broker criou a sessão A100 e recebeu um arquivo de fontes auditado. A compilação usou três execuções por modo, em ordem 1, 2, 2, 1, 1, 2, com threads numéricas CPU limitadas a uma por processo. Factor/simplify permaneceram na CPU. A concorrência foi limitada a dois trabalhadores e orçamento de 8 GiB.

| Mediana | Sequencial | Dois trabalhadores |
| --- | ---: | ---: |
| Compilação | 2,385758 s | 2,459416 s |
| Processo incluindo imports | 4,655819 s | 4,758158 s |
| RSS agregado amostrado | 639.111.168 bytes | 1.220.562.944 bytes |
| Produtores concluídos por execução | 9 | 9 |
| Caminhos emitidos | 2 | 2 |
| Caracteres emitidos | 19.532 | 19.532 |

Os seis arquivos são idênticos, SHA256 `ec7554949ecba4e07dd1d226b3b24379ed2910ffedf9526e71107605f442df66`. Cada execução paralela processou quatro trabalhos, dois blocos e nenhuma combinação de pares: a largura dois deste checkpoint não exercita a fase de merge. A ordem dessa fase permanece coberta pelos testes existentes. O RSS somado pode contar páginas compartilhadas repetidamente e não mede memória física exclusiva.

O paralelo foi 3,09% mais lento na mediana de compilação desta região. Três repetições por modo não estabelecem uma conclusão geral para modelos maiores; este resultado não sustenta acelerar a compilação completa nem estimar quando ela terminaria. O caso medido é um token, posição 0, dimensão 2, X1/X2 entre 32 e 65.504.

A verificação inicial encontrou ausência de clang++, após as seis compilações terem terminado. O compilador nativo foi instalado pelo mesmo acesso autorizado, e somente a verificação foi retomada. O arquivo passou em 1.028 comparações contra referência CPU no Colab; depois do download, passou em outras 1.028 contra referência CPU local. O estado final da comparação é COMPLETED. Os logs conservam a falha inicial e seu reparo.

## Tarefa numérica CUDA

A tradução da expressão aritmética de raiz F32 foi executada em A100: 25.167.601 casos, zero diferenças em relação à mesma expressão CPU e à raiz F64 convertida para F32. A comparação inicial em 8.196 casos identificou que a raiz Float32 do PyTorch AVX512 divergia da referência escalar em 53 casos; a expressão compilada, a referência escalar, a raiz F64 convertida e CUDA coincidiram nessa amostra.

Essa diferença entre referências não foi ocultada nem incorporada como mudança do forward. A validação CUDA declara explicitamente suas referências e não afirma equivalência à raiz Float32 desse PyTorch ou ao modelo completo.

Para 262.272 valores, cinco medições deram medianas de 64,804 ms na CPU com uma thread, 1,156 ms com dados já na GPU e 2,914 ms incluindo transferências. O ganho observado nessa tarefa foi 22,24 vezes incluindo transferências. Ele não mede factor/simplify, compilação simbólica ou inferência do modelo. O orçamento numérico foi 512 MiB; a GPU atingiu 110 MiB alocados e 144 MiB reservados.

## Limites e continuação

O build e as 21 integrações do commit de origem continuam válidos para os mesmos bytes de produção; não foram repetidos para esta mudança apenas de evidência. Ainda não há artefato da coordenada inteira, paridade no domínio completo, último token com comprimento variável, múltiplos tokens ou vetor inteiro de saída. Esses testes e artefatos continuam requisitos da conclusão.

A continuação local mostrou que a ordem atual percorre várias folhas pequenas consecutivas, enquanto seis domínios pendentes concentram 95,41% das entradas. A próxima investigação deve comparar o custo e a seleção dessas regiões grandes e verificar paralelismo entre regiões independentes, sem alterar a ordem numérica dentro de cada expressão ou reutilizar estados incompatíveis.

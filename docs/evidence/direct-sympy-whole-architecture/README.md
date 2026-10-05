# Compilação remota dos quatro logits finais

A compilação efetiva e as validações numéricas foram executadas no Colab,
na sessão `llm-inner-final-logits-20261005`, em uma NVIDIA A100-SXM4-40GB,
12 CPUs, 89.629.196.288 bytes de RAM e 42.405.855.232 bytes de memória GPU.
CPU executou substituição e simplificação; CUDA executou as tarefas numéricas
compatíveis. O teto de expressões/condições foi 512 MiB, separado da RAM.
A descoberta original CPU arm64 foi preservada por snapshot vinculado ao
checkpoint, configuração e fontes. A execução CUDA não redefine a referência.

## Resultado efetivo e limite da evidência

Foram fechadas 32 coordenadas: quatro logits da última posição para cada
comprimento de 1 a 8. As expressões de compilação incorporam os pesos,
dependem somente das entradas e não têm primitivas numéricas pendentes.
Sua avaliação diagnóstica produziu 384 comparações bit a bit com o forward
original, sem divergências. A validação CUDA dos mesmos logits também teve
384 comparações sem divergências, em aproximadamente 1,64 s.

**Não foi emitido nenhum artefato final JSON literal.** A expansão da sintaxe
ainda ultrapassa o teto, mesmo com poucas centenas/milhares de nós distintos.
O avaliador com compartilhamento estrutural é uma ferramenta de compilação,
não a entrega final e não um executor admitido para o produto. Os arquivos
`remote/**/length-*.json` são relatórios; não são expressões finais.

## Avanço, tempo e memória

Quatro trabalhadores independentes processaram as coordenadas, mantendo
contextos próprios e ordem numérica original. RSS inclui a árvore do trabalhador;
o máximo por trabalhador não é o pico agregado de todos os quatro.

| Tokens | Coordenadas fechadas | Segundos por coordenada | Máximo RSS por trabalhador (MiB) | Nós distintos | Decisões distintas |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 | 7.4–7.6 | 153.5 | 715 | 23 |
| 2 | 4 | 25.1–26.0 | 206.2 | 1548 | 52 |
| 3 | 4 | 42.5–44.4 | 269.5 | 2017 | 68 |
| 4 | 4 | 68.7–69.7 | 282.2 | 2443 | 82 |
| 5 | 4 | 86.6–90.7 | 356.1 | 2891 | 97 |
| 6 | 4 | 116.5–117.8 | 454.2 | 3335 | 112 |
| 7 | 4 | 155.1–156.7 | 573.9 | 3782 | 127 |
| 8 | 4 | 202.6–215.5 | 646.5 | 4227 | 142 |

A poda antes da composição reduziu produtores de 72 para 45 em dois tokens
e de 336 para 105 em oito tokens. Foram preservadas as dependências causais
de atenção que contribuem à última posição. Contagem de produtores do
planejador vetorial e eventos de substituição do lowering são métricas distintas.

O lote JSON anterior, sem sessão de provas reutilizáveis, fechou somente
16 coordenadas: comprimentos 5–8 atingiram o prazo de 180 s. Com a sessão,
as 32 fecharam; o prazo foi 300 s. Em cinco tokens, o fechamento observado
ficou abaixo de 91 s. Os lotes compartilharam recursos em parte; essa comparação
não é um benchmark isolado nem demonstra um fator de aceleração exato.
O cache é exclusivo da compilação, condicionado aos fatos relevantes de cada
ramificação, e não aparece no artefato.

Na comparação textual SymPy de dois tokens, sequencial e quatro trabalhadores
atingiram o limite sem emitir uma coordenada: aproximadamente 150,8 s e 151,4 s.
Não há ganho nem paridade de artefato completo demonstrados nesse ensaio.
O backend textual ainda conserva operações numéricas pendentes nos arquivos
de trabalho de um token; eles não são a fonte estrutural nem o produto final.

## Regressão e proveniência

Os testes novos verificam o recorte, a última linha da referência, a manutenção
de dependências causais, rejeição de snapshots incompatíveis, cancelamento
seguro de bitcasts e isolamento de contexto na sessão de simplificação.
A regressão JSON remota inicial teve 107 testes: 99 passaram, 6 foram pulados
e 2 falharam por ferramentas nativas ausentes. A instalação de clang/Rust e
reexecução desses gates foram registradas nos logs nativos: raiz/inversa passou
com clang e a prova exaustiva de soma/subtração Half passou com Rust 1.85.1,
em 1.007.713.280 comparações de magnitudes e oito casos de zero com sinal.
Assim, os 101 testes executáveis da seleção JSON passaram em sua execução
inicial ou reexecução, mantendo seis skips explícitos. Nenhum teste foi
relaxado para ocultar falhas. O mapa mantém os resultados anteriores separados.

O manifesto `colab-bundle.json` identifica o transporte inicial. Os snapshots
em cada raiz identificam a versão efetivamente compilada; após alterações
na simplificação, apenas os hashes dos arquivos auditados foram atualizados
na cópia remota, sem reaproveitar resultados numéricos de outra versão.
`validation.json` resume os relatórios integrais em `remote/`.

A meta permanece aberta: reduzir o crescimento estrutural, emitir o JSON
literal efetivo e demonstrar a paridade do arquivo relido.

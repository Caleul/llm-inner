# Gemma 4 E4B: SSA simplificado e comparação interativa

## Arquivo aritmético

`npm run export:gemma4-real-ssa` projetou a seção real simplificada para
`artifacts/gemma4-e4b-dense.real-simplified.ssa.json` por escrita streaming.

- bytes: `430.678.350`;
- SHA-256: `ce2516604b8457ced0270a67242938f30664e85786431ed6336332654743b69e`;
- statements SSA: `542.655`;
- funções de operação: `1.321`;
- atribuições finais: `264.704`;
- `final_hidden_dimension`: `2.560`;
- `terminal_logit`: `262.144`;
- reduções não resolvidas: `0`;
- arredondamentos IEEE intermediários: `0`.

Cada output possui uma atribuição `calc_<família>_<dimensão>`. Pesos são
constantes racionais endereçadas na memória autenticada do artefato literal,
declarada no próprio arquivo em `constantMemory`; o checkpoint Safetensors não
é necessário para navegar ou executar essa representação.

O arquivo foi relido com `JSON.parse`; foram confirmados o primeiro output
`calc_final_hidden_dimension_0`, o último `calc_terminal_logit_262143` e as
contagens acima. Um recorte verificável está em
`docs/examples/gemma4-real-simplified-ssa.example.json`.

## Limite da expressão fisicamente expandida

A estimativa conservadora mantém cada soma finita como um único operador e não
desenrola suas iterações. Mesmo assim, `terminal_logit[0]` teria
`5,8031173398169e247` ocorrências de nós após substituir fisicamente todas as
chamadas. SSA preserva a mesma composição matemática sem repetir cálculos.

## Interface e execução real

`npm run compare:gemma4-real-ui` inicia `http://127.0.0.1:8787`. O endpoint
`POST /api/compare` recebe prompt e quantidade de novos tokens. Em cada passo:

1. Transformers eager executa BF16 com seus arredondamentos declarados;
2. o candidato executa as operações textuais em F64 sem casts intermediários;
3. ambos quantizam os logits terminais em BF16 RNE;
4. os dois argmax são anexados independentemente ao respectivo contexto;
5. os 262.144 logits são comparados e os textos são decodificados.

Para tornar a comparação completa executável em segundos, o candidato agrupa
as mesmas somas, produtos e transcendentais em operações tensoriais F64. Ele
não interpreta os 542.655 statements um por um; essa vetorização preserva a
semântica numérica declarada pelo SSA e evita adicionar arredondamentos. O
tempo exibido pela interface mede esse executor, não uma aceleração presumida.

No prompt `The capital of France is`, com dois tokens, os dois caminhos geraram
`[496, 3207]`, decodificados como ` a city`. Foram comparados 524.288 valores:

| passo | divergência BF16 | erro absoluto máximo | argmax |
|---:|---:|---:|---|
| 0 | 88,168716% | 0,5 | igual |
| 1 | 90,806580% | 0,4375 | igual |

O fluxo também foi exercitado em navegador real com Playwright: carregar a
página, definir dois tokens, clicar em `Gerar e comparar` e verificar os dois
textos, tokens, tabela e JSON. Após adicionar favicon vazio, a abertura ficou
sem erros de console.

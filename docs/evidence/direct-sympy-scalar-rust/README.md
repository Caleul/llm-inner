# Llama direto: quatro logits finais em Rust

O artefato efetivo está em `artifacts/llama-next-token.rs`: 393.073 bytes,
SHA-256 `25aac526deff02d6d60726e07925e7f32eb7cb34e43b706282da245cd8e7eecd`.
`predict_next_token(&[f64]) -> [f64; 4]` recebe os valores de entrada Half
alargados exatamente para f64, em ordem token/dimensão. Cada token tem duas
dimensões neste checkpoint; o comprimento varia de 1 a 8 tokens. É o mesmo
contrato `inputs_embeds` do forward capturado, incluindo zeros com sinal,
subnormais e todo o domínio de entradas Half finitas.

A função incorpora os pesos necessários e todas as dependências causais da
última posição. Não lê checkpoint, descritores, respostas, arquivos ou grafo
em runtime. Os temporários são escalares locais com produtores definidos;
não são interfaces pendentes. Os cinco corpos numéricos incluídos no arquivo
contêm somente aritmética, operações de palavras, bitcasts e decisões.
Não restam chamadas a exp/sqrt nativos ou abstrações R16/R32/SiLU/Exp32.
Os corpos especializados têm domínios provados pela arquitetura: raiz de F32
positivo, ativação Half até 1/16 e exponencial de diferenças não positivas de
scores Half. Não extrapolam esses certificados para outros checkpoints.

## Percurso efetivo

O comando público `helpers/direct_sympy_architecture_run.py` agora usa
`direct_sympy_scalar_run.py`. Descobre geometria/config/pesos, seleciona apenas
os quatro logits finais, poda retroativamente suas dependências e prepara
expressões matemáticas em strings. Compõe blocos em pares sucessivos usando o
mesmo `StringCompiler`: parênteses, substituição, factor/simplify até estabilizar
e propagação de condições. Um produtor reutilizado ganha uma definição escalar;
não se copia a arquitetura acumulada a cada ponto de uso. Os ramos continuam
nas expressões de seus produtores, com avaliação condicional preguiçosa.

JSON serve somente para relatórios e descritores de compilação persistidos.
Esses descritores são relidos e validados antes da emissão: produtores únicos,
ordem, entradas e dependências resolvidas. O comando recompõe cada comprimento;
não reutiliza estados antigos de outro backend. O percurso textual totalmente
expandido está disponível apenas com `--inline-diagnostic`.

A emissão Rust é streaming e atômica. O teto acumulado é 2 GiB
(2.147.483.648 bytes), separado da RAM. Nesta execução a RAM foi limitada a
6 GiB e a concorrência a dois processos, com admissão e observação do RSS da
árvore. O candidato só é promovido depois de compilar e executar a comparação
bit a bit dos quatro resultados para todos os comprimentos suportados.

## Evidência real

A captura original usa Torch 2.12.1, CPU ARM64 eager, 96 entradas e comprimentos
1–8. Capturas de referência são pequenas, com uma thread; compilação, testes
pesados, emissão e execução Rust ocorreram no Colab via Access Broker.

- Composição com dois processos: 384 comparações, zero divergências.
- Composição com um processo: os mesmos 384 resultados, zero divergências.
- Rust efetivo: 384 comparações contra a captura original, zero divergências.
- Comando público completo: exit code 0, gera o mesmo SHA-256 e repete a paridade.
- Regressão: 66 testes passaram em 36,227 s; gates posteriores cobrem emissão
  streaming, orçamento e a fronteira obrigatória R32(sqrt(...)).
- Raiz: todos os 16.777.216 pares mantissa/paridade normalizados, zero divergências.
- Arredondamento F32: 1.000.000 fontes F64 finitas, zero divergências.
- SiLU: todos os 22.530 Half no domínio certificado, zero divergências contra ARM64.
- Exp: todos os 474.369 pontos da grade que contém as diferenças de scores,
  zero divergências contra ARM64. A captura contém somente saídas de referência;
  não participa do artefato final.

O teste adicional com o exp do Torch 2.11/x86 do Colab teve 2.583 diferenças de
um ULP F32. Isso identificou uma referência de ISA/versão diferente, não autoriza
alterar a semântica original. Os contraexemplos foram conferidos no ARM64 e a
grade completa foi capturada; o Rust coincidiu com todos os bits originais.
Essa prova e o resultado diferencial estão preservados separadamente.

## Avanço, tempo e memória

Ensaio com dois processos; tempo total inclui preparar a arquitetura e verificar
as expressões. Os produtores contam cálculos distintos, não cópias de fórmulas.

| Tokens | Produtores escalares | Caracteres das expressões | Composição (s) | Total (s) | Comparações / divergências |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 25 | 2.204 | 0,659 | 1,803 | 48 / 0 |
| 2 | 45 | 4.419 | 1,916 | 4,935 | 48 / 0 |
| 3 | 55 | 6.525 | 0,858 | 4,294 | 48 / 0 |
| 4 | 65 | 10.149 | 1,695 | 7,416 | 48 / 0 |
| 5 | 75 | 16.479 | 1,881 | 8,483 | 48 / 0 |
| 6 | 85 | 26.954 | 4,421 | 13,935 | 48 / 0 |
| 7 | 95 | 43.360 | 7,115 | 19,344 | 48 / 0 |
| 8 | 105 | 67.802 | 8,862 | 25,341 | 48 / 0 |

A execução anterior de dois tokens levava 52,181 s e pico de 958 MiB;
a composição com escalares levou 4,935 s e pico de 649 MiB, aproximadamente
10,6 vezes menos tempo nesse ensaio. Oito tokens antes excediam o limite de
2 GiB por cópias; agora concluem. O arquivo contém 550 declarações ao reunir
os oito casos de comprimento; uma previsão executa no máximo 105 produtores
no caso de oito tokens, além dos cálculos elementares de cada expressão.

No Colab atual (2 CPUs, 12,67 GiB de RAM), dois processos não aceleraram este
fixture: composição total 27,407 s contra 26,205 s com um processo; tempo total
85,551 s contra 79,346 s. RSS observado até 692 MiB contra 493 MiB. São ensaios
únicos, sem repetição estatística. O ganho principal veio da representação sem
cópias. A concorrência permanece limitada pela memória e configurável.

A consulta atual de sessões confirmou somente o runtime CPU
`llm-inner-final-logits-eligible-20261005`; CUDA não estava disponível.
Nenhuma aceleração GPU é atribuída a estes resultados.

## Reprodução no Colab

```sh
python helpers/direct_sympy_architecture_run.py \
  docs/evidence/direct-sympy-test-checkpoint output \
  --reference docs/evidence/direct-sympy-reachable-ports/arm64-all-lengths-reference.json \
  --workers 2 --memory-mib 6144 --max-characters 2147483648 \
  --seconds-per-length 60
```

Use `colab_cli` do Access Broker para executar remotamente. Os arquivos em
`remote/` incluem os resultados, expressões de cada produtor, teste Rust e
manifests com hashes conferidos. As referências binárias comprimidas e os
relatórios ARM64 são apenas evidências de teste.

# Raiz F32 racional com menos dependências

A expansão aritmética da raiz F32 agora usa uma expressão racional de grau 5/5, fatorada em termos lineares e quadráticos com SymPy. A interpolação usa onze pontos Lobatto em 100 dígitos; as raízes polinomiais são obtidas em 70 dígitos, e os coeficientes finais são arredondados uma vez para F64. O teste reproduz todos esses coeficientes. São constantes de um algoritmo numérico universal, sem respostas do checkpoint.

A expressão reduz as ocorrências da entrada de nove para oito e elimina a etapa de refinamento anterior. Isso substitui o algoritmo interno da expansão, não permite reordenar operações do modelo nem afirma equivalência F64 irrestrita. A admissão continua restrita à fronteira de saída F32 com entrada positiva F32 finita; uma raiz F64 observável permanece sem essa transformação.

## Fidelidade e regressões

- 25.167.601 comparações nativas: todas as mantissas normalizadas e ambas as paridades, todos os subnormais positivos F32 e fronteiras de expoentes. Zero divergências, zero empates pares/ímpares que invalidem a prova. O escalonamento por potências de dois preserva o resultado F32 sem underflow/overflow da saída.
- Build e 21 integrações passaram, sem falhas/skips. A asserção regional foi reforçada para exigir menos que os 21.922 caracteres anteriores; a integração correspondente foi reexecutada depois dessa asserção.
- 30 regiões recompiladas com os novos hashes, sem importar resultados numéricos antigos. 7.800 comparações dos arquivos salvos com checkpoint de referência CPU carregado novamente, nenhuma divergência.
- Região positiva real: `coordinate-region.expr`, 19.532 caracteres, entradas X1/X2 entre 32 e 65.504; 576 comparações na geração e 1.028 comparações adicionais após leitura do arquivo, zero divergências.
- As seis funções regionais distintas foram preservadas nesta pasta. `partition-parity.json` identifica os arquivos e seus hashes; coalescência mantém nove retângulos, 518.432 caracteres de corpos únicos, cobertura de 181.559.808 padrões (4,50439453125%). Paridade regional amostrada não é prova exaustiva desses padrões.

## Efeito na expansão e limite atual

A coordenada inteira passou de 4.476.543.086.072 para 3.248.163.585.496 caracteres lógicos, redução de 27,44%. `coordinate-run.json` registra 25 produtores, emissão recusada pelo limite e nenhum estado antigo restaurado. O primeiro caminho ainda precisa de 8.648.306.495 caracteres de corpo e 4.992.480.549 de condições.

Na região problemática medida anteriormente, a expansão lógica caiu de 6.792.712 para 5.427.192 caracteres; a emissão de 1 MiB continua insuficiente. O ganho não foi convertido em alegação de velocidade: as execuções locais não formam benchmark controlado.

## Alternativa recusada

Uma forma que integra a escala ao racional 6/6 reduz a entrada para sete ocorrências, porém o teste nativo encontrou 80 divergências e um empate em 25.167.601 casos. Ela não foi incorporada. `rejected-wide-candidate.json` preserva a evidência. As sondas `probe.py` e `probe_wide.py` são experimentos reproduzíveis que escrevem somente em `artifacts/`; não participam do compilador.

## Pendências

As expressões são strings matemáticas; JSON registra estado/evidência. O mapa de testes está em `validation.json` e `docs/direct-string-validation.json`. A coordenada completa e sua paridade no domínio inteiro continuam pendentes, assim como último token com comprimento variável, todas as coordenadas e comparação atual controlada no Colab. As regiões usam um token, posição 0 e dimensão 2. Nenhum artefato parcial ou contagem de expansão substitui esses requisitos.

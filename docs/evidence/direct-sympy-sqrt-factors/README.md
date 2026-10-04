# Fatoração durante a compilação direta

A representação das expressões continua sendo string matemática. Os arquivos JSON desta pasta são evidência e estado de compilação, não a função final.

## Mudança e fidelidade

A expressão polinomial usada na expansão da raiz F32 foi fatorada com SymPy, com coeficientes derivados das constantes F64 exatas e cálculo em 70 dígitos. A forma fatorada reduz as ocorrências da dependência de entrada de 13 para 9. Isso muda a avaliação interna dessa expansão; não permite reordenar as operações do modelo. A validação nativa cobre 25.167.601 entradas, sem divergência nem empates que invalidem a certificação existente.

O teste de integração inicialmente falhou porque fixava o tamanho anterior de 28.282 caracteres. A expectativa foi atualizada para exigir redução de tamanho, mantendo os 576 casos de paridade e o escopo original. Depois da correção: build e 20 integrações passaram, sem falhas ou testes ignorados. `integration-tests.log` conserva a primeira falha; `integration-tests-final.log` registra o gate corrigido.

## Artefatos e avanço efetivo

- `coordinate-region.expr`: expressão direta real com 21.922 caracteres, dimensão 2, posição 0, entradas X1/X2 entre 32 e 65.504. A leitura posterior do arquivo confirma SHA e 1.028 comparações nativas sem divergência. Esse artefato cobre somente essa região e um token.
- `frontier.json`: cópia do estado após 36 tentativas limitadas, com 14 regiões completas. Elas cobrem 178.115.584 de 4.030.726.144 padrões de entrada (4,4189453125%). Ainda faltam 3.852.610.560 padrões.
- `partition-parity.json`: 3.640 comparações com as 14 expressões salvas e um checkpoint de referência carregado novamente; nenhuma divergência. Amostragem de paridade não é prova exaustiva de todos os padrões cobertos.
- `coalesced.json`: 14 regiões têm 4 corpos distintos; coalescência preserva exatamente a cobertura e reduz cópias de 3.410.767 para 623.657 caracteres. Essa união ainda é parcial.
- `coordinate-run.json`: composição de 25 dependências no domínio inteiro, sem reutilizar estado antigo. A expansão lógica cai de 11.756.851.964.536 para 4.476.543.086.072 caracteres (61,92%). São contagens antes da emissão, não um arquivo final. O primeiro caminho ainda exige 12.086.426.103 caracteres de corpo e 6.878.832.457 de condições; o limite impede sua emissão.
- `old-state-rejection.log`: o estado antigo foi rejeitado por mudança dos hashes do compilador. As expressões antigas não foram adotadas como prova da versão nova.

## Reprodução e pendências

`measure_coordinate.py` reproduz a medição inteira a partir da raiz do repositório. `validation.json` e `docs/direct-string-validation.json` mapeiam os testes e a evidência correspondente. A identidade de fonte em `coordinate-run.json` foi conferida novamente antes deste registro.

Os tempos locais não constituem comparação controlada de desempenho. Esta versão não foi validada novamente no Colab. A coordenada completa, sua paridade no domínio inteiro, o último token com comprimento variável e o vetor completo continuam pendentes. Nenhum desses itens é substituído pelos artefatos regionais ou pelas métricas de crescimento.

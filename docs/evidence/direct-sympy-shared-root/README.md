# Operação externa preservada durante a substituição

Ao compartilhar uma expressão pronta, o compilador registrava sua faixa
numérica e tipo, mas perdia a operação externa. O mesmo acontecia ao criar
um marcador temporário para uma substituição numérica. Com isso, operações
inversas adjacentes deixavam de cancelar, embora o produtor original já
tivesse a prova exigida.

A informação agora acompanha o marcador no mesmo domínio de entrada. Um
argumento inteiro só é exposto quando a largura de palavra já foi provada;
raízes desconhecidas e contextos diferentes continuam sem essa otimização.
Se a operação externa não muda, o marcador/literal original é restaurado.
Somente o cancelamento exato substitui seu argumento. Isso também evita
criar uma cópia desnecessária do argumento durante a restauração.

O método continua usando as passagens existentes de `factor()` e
`simplify()` antes de avançar. A mudança não reassocia operações flutuantes,
não troca algoritmos numéricos nem elimina ramificações arbitrariamente.

## Evidências

- `integration-tests.log`: 12 integrações passaram, incluindo as 11
  anteriores, com o teste de compartilhamento então composto por seis
  testes Python.
- `new-integration-tests.log`: execução final da integração de
  compartilhamento, ampliada para sete testes Python. Inclui a prova
  nativa do prefixo efetivamente emitido, além da composição completa
  da coordenada na posição zero.
- `pre-inverse.expr`: expressão efetiva da inversa da primeira normalização,
  com 5.212 caracteres e somente X1/X2 como variáveis. O teste lê esse
  arquivo, gera C++ e compara os bits com operações nativas F32. Todos os
  63.488 padrões Half finitos na primeira entrada são cruzados com 14
  valores da segunda: 888.832 comparações, zero divergências.
- A expressão pequena de duas substituições permanece validada em
  30.722 comparações nativas, incluindo zeros com sinal.
- A composição completa da dimensão 2 na posição zero passa em 60 casos
  contra um forward CPU recém-capturado, com zero divergências. Sua
  expansão estimada caiu de 15.345.975.626.240 para 13.773.588.294.816
  caracteres: redução adicional de 10,25%. São 25 produtores concluídos.
- `validation.json` registra o hash do prefixo, fontes numéricas, medidas
  e os limites de cada validação. Os estados antigos mantêm a rejeição
  por identidade dos helpers antes de qualquer restauração.

## Pendências

O prefixo emitido não é a coordenada final. A expansão completa ainda é
excessiva e não foi gravada nem validada como artefato. O adaptador continua
na posição zero; casos com vários tokens não provam a saída do último token.
As outras coordenadas não foram iniciadas. O resultado do benchmark Colab
anterior permanece separado: esta mudança foi validada localmente e não
possui uma nova comparação de desempenho no Colab.

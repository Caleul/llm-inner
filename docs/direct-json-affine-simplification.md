# Simplificação afim durante a substituição

O compilador reúne coeficientes e constantes diretamente sobre o JSON.
Após substituir produtores, compõe operações F32 antes de fechar seus
arredondamentos em bitwise. Operações originais e operações emitidas devem
ser exatas nas faixas e grades dyádicas certificadas: 24 bits de precisão
para F32, 53 para F64, expoentes representáveis e ausência de overflow.
O cálculo dos coeficientes usa inteiros BigInt e expoentes, sem aproximação
intermediária de coeficientes. Nenhuma equivalência sobre reais é admitida
como prova de equivalência IEEE.

A etapa sobre o JSON fechado visita todos os tipos de nós, incluindo
operações, comparações, conversões e ramos alcançáveis. Cada substituição
passa por precisão, álgebra afim, compartilhamento interno e simplificação
estrutural/condicional até ponto fixo. Nós sem transformação comprovada
permanecem explícitos. A opção incremental=false existe somente como
opção diagnóstica; a compilação principal mantém esse ciclo habilitado.

Condições numéricas refinam as faixas dentro de seu próprio ramo. Os fatos
não passam ao ramo complementar. Quando a condição não altera o domínio,
a análise reutiliza o mesmo cache. As provas mais fortes de produtores reutilizados são preservadas; uma
identidade não enfraquece sua grade numérica. As formas afins também são memoizadas,
evita-se repetir o percurso de produtores compartilhados. Nenhuma dessas
provas, tabelas ou referências é serializada no resultado final.

Zeros assinados são preservados. A composição exige que o resultado
original não possa ser -0 e mantém um acumulador +0 na expressão emitida.
Fronteiras de arredondamento não exatas permanecem como átomos explícitos;
suas próprias dependências também passam pela análise. As comparações,
máscaras e identidades inteiras continuam no simplificador estrutural.

O exemplo em examples/direct-json-affine-substitution.example.json mostra
14*(132X1 + 3X2 + 4X3) reduzido a 1848X1 + 42X2 + 56X3, com o +0 necessário
para preservar os bits. Declara domínio F16 finito com |Xn| <= 2^-12.
É um exemplo de substituição, não um checkpoint Llama nem redução de seu
domínio. Seu JSON foi relido e validado em 216 combinações de fronteira.

Validação final: 19 testes focados aprovados; regressão com 590 testes,
572 passes, as mesmas 15 falhas conhecidas, 3 skips e nenhum passe anterior
perdido. Os três testes históricos de outras coordenadas permanecem
adiados. O teste integrado recaptura o forward PyTorch e compara 60 casos
da coordenada posição 0/dimensão 2, com vários comprimentos. Tokens
posteriores não influenciam a posição zero; isso não prova outras posições.

O tamanho previsto dessa coordenada caiu de
3.525.760.683.880.951.562.623 para 3.525.522.711.179.479.875.721 bytes.
Ainda não é emitível. Essa implementação cobre o núcleo afim e submete
todos os nós à simplificação; identidades não lineares/racionais gerais
e refinamento de condições bitwise para faixas numéricas exigem regras
adicionais. Não conclui a compilação completa do Llama.

O mapa e os hashes desta rodada estão em affineSubstitutionValidation
de direct-json-validation.json. Alterações anteriores e instruções locais
do usuário foram preservadas.


## Reinício verificado com 63ef888

O commit solicitado já estava no HEAD. O build foi repetido e os hashes
dos fontes correspondem à evidência registrada para esse commit. Foram
inspecionados nove estados JSON antigos: todos sem unidades emitidas,
identidade versionada de compilador/checkpoint e cursor retomável. Nenhum
foi reutilizado ou modificado. Logs de crescimento não são savepoints.

A nova execução possui manifesto separado com hashes por conteúdo dos
fontes, instruções e arquivos Safetensors/config. Esse manifesto audita
o reinício; não implementa retomada. O escritor foi executado para posição
0/dimensão 2 com 64 MiB. Os eventos confirmam estabilização após cada
substituição, antes do consumidor seguinte. A preparação terminou, mas
a emissão foi rejeitada: 3.525.522.711.179.479.875.721 bytes previstos,
zero coordenadas emitidas e finalParity=false. O processo terminou; o
status pending do escritor não significa que exista um processo ativo.

O build passou e 23 testes focados foram aprovados. O teste integrado
recapturou PyTorch e verificou 60 resultados da expressão fechada em
vários comprimentos, incluindo fronteiras numéricas e sinais de zero.
Não é paridade de um arquivo final nem de outras posições. Não avançamos
às demais coordenadas. A regressão anterior de 590 testes foi preservada,
sem alegar uma nova execução integral. Logs e hashes desta rodada estão
em verifiedAffineCoordinateRestart no mapa de validação.

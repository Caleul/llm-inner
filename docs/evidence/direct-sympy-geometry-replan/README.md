# Retomada por geometria auditada

`--repartition-from ESTADO_ANTIGO` importa somente as divisões do domínio para um destino novo. A identidade do checkpoint, dimensão, posição, versão do formato e domínio raiz devem coincidir. Cada divisão é refeita na auditoria para provar cobertura sem perda ou sobreposição.

A operação não importa expressões, artefatos, provas numéricas, tentativas nem marcadores de sucesso. Todas as folhas voltam a pendentes e são recompiladas com a identidade atual do código e backend. `--resume` continua exigindo identidade completa e não pode ser combinado com esse modo. Um destino com estado existente é rejeitado antes de alterá-lo.

## Execução e validação

O import recompilou as 14 regiões anteriormente completas em 14 tentativas; a árvore original precisou de 36 tentativas para descobri-las. São contagens de trabalho, não um benchmark de tempo controlado. Mais oito tentativas na nova versão concluíram quatro regiões adicionais.

- 18 regiões completas, 179.099.648 padrões cobertos de 4.030.726.144 (4,443359375%).
- Avanço novo de 984.064 padrões; ainda faltam 3.851.626.496.
- 4.680 comparações nativas dos arquivos salvos com checkpoint CPU carregado novamente, nenhuma divergência. São amostras por região; não paridade exaustiva do domínio inteiro.
- Quatro expressões distintas, preservadas uma vez nesta pasta e identificadas por SHA em `coordinate-parity.json`. A união auditada equivale a cinco retângulos; os corpos somam 623.657 caracteres sem cópias repetidas.
- Build e 21 integrações passaram, zero falhas/skips. Os dois testes novos verificam limpeza dos resultados antigos, independência do estado original e rejeições de identidade e geometria inválida.
- Uma tentativa real de importar para o destino existente foi rejeitada, mantendo o SHA do manifesto intacto.

## Reprodução

```sh
python helpers/direct_sympy_partition_run.py CHECKPOINT NOVO_ESTADO --repartition-from ESTADO_ANTIGO --max-attempts 14 --region-seconds 12
python helpers/direct_sympy_partition_run.py CHECKPOINT NOVO_ESTADO --resume --max-attempts 8 --region-seconds 12
python helpers/direct_sympy_region_parity.py CHECKPOINT NOVO_ESTADO RELATORIO
```

`frontier.json` é uma cópia de auditoria; os caminhos originais dos artefatos pertencem ao estado local indicado no relatório. Para reproduzir a geração, use um destino novo. As expressões duráveis copiadas aqui têm o mesmo SHA das expressões realmente executadas pelo verificador.

## Limites

A coordenada inteira não foi emitida e não possui paridade final. A prova regional continua restrita a um token, posição 0 e dimensão 2. Último token com comprimento variável, vetor inteiro e comparação controlada atual no Colab permanecem pendentes. JSON contém evidência/estado; as expressões são strings matemáticas. Esta mudança reduz recomputação da descoberta geométrica, sem substituir a simplificação necessária para concluir o domínio.

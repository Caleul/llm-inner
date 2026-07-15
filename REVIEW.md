# Revisão técnica do conversor original

## Veredito

O programa original não decompila uma LLM de forma fiel. Ele reconhece alguns nomes de tensores e os encaixa em um bloco Transformer previamente presumido. Quando o checkpoint difere dessa forma, o programa não detecta a divergência: produz uma rede inventada e matematicamente diferente.

O problema não é somente cobertura de nomes. Safetensors, MLX e GGUF representam camadas distintas do problema:

1. **Contêiner**: onde metadados e bytes estão armazenados.
2. **Quantização**: como cada tensor lógico é codificado nos bytes.
3. **Arquitetura**: quais operações existem e como os tensores são conectados.
4. **Semântica numérica**: dtype, acumulação, arredondamento, ordem das reduções e cache.

A reescrita separa essas responsabilidades.

## Falhas de fidelidade encontradas

### 1. MAX_FEATURES alterava a rede

O código antigo usava `Math.min(inD, MAX_FEATURES)` dentro da soma da projeção. Assim, uma matriz com 1536 entradas era convertida em uma soma de apenas 10 termos. Isso não é uma visualização parcial: é outra função.

Na versão 2, a operação guarda a soma completa:

```text
y[o] = Σ(i=0..inFeatures-1) x[i] * W[o,i] + b[o]
```

`MAX_FEATURES` e `MAX_TERMS` limitam apenas o preview opcional.

### 2. A quantização era inferida incorretamente pelo dtype

`U16` e `I16` não significam “quatro valores de 4 bits”. O dtype do contêiner descreve o armazenamento, não o esquema de quantização. Também existiam os seguintes erros:

- `group_size = 64` fixo;
- somente `scales + biases` eram reconhecidos;
- ausência de overrides por módulo;
- nenhuma distinção entre affine, MXFP4, NVFP4, Q4_K, IQ2 e outros layouts;
- fórmula affine invertida: o código usava `(q - bias) * scale`, enquanto o formato MLX usa `scale * q + bias`;
- leitura do tensor inteiro para exibir poucas linhas.

A versão 2 delega apenas a dequantização MLX cujo contrato é conhecido ao
backend de referência:

- MLX: `mlx.core.dequantize`;
- Safetensors denso: `safetensors`.

O catálogo GGUF v2/v3 é lido localmente, com metadata e intervalos alinhados
validados. Tipos GGML empacotados são rejeitados até cada formato ter um
decodificador e layout verificados; um pacote Python instalado não transforma
um tipo ou largura de bits em semântica comprovada.

### 3. A arquitetura era presumida pelos nomes

O programa sempre gerava aproximadamente:

```text
RMSNorm → QKV → RoPE → Attention → residual
→ RMSNorm → MLP → residual
```

Essa hipótese não cobre, entre outros:

- normas Q/K/V;
- QKV fundido e seus layouts;
- MQA/GQA com shapes diferentes;
- atenção local/global por camada;
- RoPE diferente por tipo de camada;
- quatro normas por bloco;
- KV compartilhado;
- per-layer embeddings;
- `K = V`;
- MoE/router/experts;
- MLP double-wide ou projeções fundidas;
- state-space e linear attention;
- multimodalidade;
- bias nas projeções;
- embeddings amarrados ao LM head;
- softcap final;
- dtype e regras de cache.

A versão 2 usa `config.json` ou metadados GGUF e aceita apenas famílias com adaptador explícito. Sem adaptador, encerra com erro.

### 4. Detecção de Gemma por nome de tensor não funcionava

A condição `tensorName.includes("gemma")` normalmente é falsa porque nomes reais são como `model.layers.0.input_layernorm.weight`. Com isso, unit-offset RMSNorm e softcapping podiam ser omitidos ou aplicados incorretamente.

A versão 2 usa `model_type` da configuração.

### 5. Normas eram conectadas na posição errada

Em Llama/Mistral/Qwen, `post_attention_layernorm` é a norma aplicada ao estado após o residual de atenção e antes do MLP. Ela não normaliza a saída da atenção antes do residual.

Em Gemma 2/3, há normas adicionais de saída dos ramos. A versão 2 diferencia as duas topologias e possui testes para impedir regressão.

### 6. O I/O tinha custo desnecessário

O programa original abria o shard, relia os oito bytes iniciais e reinterpretava o cabeçalho para cada tensor. Depois carregava o tensor completo em um `number[]`, mesmo para produzir dez termos.

A versão 2:

- mantém handles dos shards abertos;
- interpreta cada cabeçalho uma vez;
- mantém um processo Python persistente para previews;
- solicita apenas linhas necessárias;
- não lê pesos quando `--include-weights` não é usado;
- representa projeções por referência, evitando JSONs de gigabytes.

## Sobre Gemma 4

O checkpoint usado no exemplo não pode ser representado fielmente pelo bloco genérico antigo. Gemma 4 inclui recursos como per-layer embeddings, dimensões de cabeça distintas para atenção global/local, KV sharing, possibilidade de `attention_k_eq_v`, RoPE por tipo de camada, variantes MoE e MLP double-wide.

Por esse motivo, esta versão **rejeita Gemma 4 deliberadamente**. O próximo passo correto é um adaptador dedicado baseado na implementação oficial, acompanhado de validação diferencial. Aceitar o arquivo e gerar uma aproximação seria pior do que falhar.

## Igualdade matemática e igualdade numérica

Mesmo um grafo matematicamente correto pode não reproduzir bits idênticos quando muda:

- BF16/F16/F32 de cada operação;
- dtype do acumulador;
- ordem de soma do produto matricial;
- implementação de `exp`, `tanh`, `rsqrt` e GELU;
- softmax com ou sem upcast;
- ordem de atualização do KV cache;
- rounding após cada kernel.

Portanto, a meta deve ser validada em camadas:

1. mesmos tensores e shapes;
2. mesmo grafo e parâmetros de configuração;
3. comparação de embeddings;
4. comparação após cada operação/camada;
5. comparação do KV cache;
6. logits completos;
7. decisões greedy/top-k;
8. bitwise quando o mesmo backend e ordem operacional permitirem; tolerância explicitamente definida nos demais casos.

## Estrutura da versão 2

- `src/safetensors.ts`: catálogo eficiente de shards e metadados.
- `helpers/tensor_bridge.py`: leitura/dequantização pelo backend de referência.
- `src/architecture.ts`: lowering de configuração+tensores para IR estruturado.
- `src/types.ts`: schema do IR.
- `src/render.ts`: equações humanas sem truncar a matemática.
- `src/compiler.ts`: pipeline de compilação.
- `test/`: testes das garantias de fidelidade já implementadas.

## Próxima etapa recomendada

Para chegar à proposta de “qualquer LLM”, não crie um adaptador genérico permissivo. Use duas rotas:

1. **Adaptadores estáticos estritos**, para arquiteturas conhecidas e auditadas.
2. **Captura do grafo do runtime oficial**, para modelos com código customizado: carregar a implementação declarada pela configuração, traçar/exportar o forward e então baixar esse grafo para o mesmo IR.

GGUF exige ainda um mapeamento por `general.architecture`, porque seus metadados e tensores não contêm uma descrição universal de todas as operações.

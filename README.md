# Model Decompiler v2

Reescrita do conversor de pesos para um IR matemático estruturado e verificável.

## Mudança principal

`MAX_FEATURES` não altera a rede. Ele limita apenas o preview embutido no JSON. Uma projeção continua sendo representada como:

```text
y[o] = Σ(i=0..in_features-1) x[i] * W[o,i] + b[o]
```

O IR guarda `inFeatures`, `outFeatures`, referência ao tensor, dtype e quantização. Os primeiros termos podem ser anexados para inspeção, mas nunca substituem a soma completa.

## Fontes

- Diretórios Hugging Face/Safetensors densos.
- Diretórios MLX/Safetensors quantizados, lendo `config.json`, incluindo overrides por módulo.
- Arquivos GGUF por meio do pacote oficial `gguf`.

A leitura numérica é delegada ao runtime de referência:

- `mlx.core.dequantize` para MLX (`affine`, `mxfp4`, `mxfp8`, `nvfp4` e modos suportados pela versão instalada).
- `gguf.dequantize` para tipos GGML/GGUF suportados pelo `gguf-py` instalado.

Isso evita implementar uma falsa “dequantização genérica por número de bits”. Q4_K, IQ2, MXFP4 e affine-4bit têm layouts e fórmulas diferentes.

## Instalação

```bash
npm install
npm run build
python3 -m pip install safetensors
# Em Apple Silicon, para modelos MLX:
python3 -m pip install mlx
# Para GGUF:
python3 -m pip install gguf
```

## Uso

```bash
node dist/src/cli.js \
  --source ./gemma-4-E4B-it-MLX-4bit \
  --output ./model.ir.json \
  --equations ./model.equations.txt \
  --max-features 10 \
  --max-terms 10 \
  --include-weights
```

Sem `--include-weights`, a compilação não dequantiza previews; ela apenas cataloga os tensores e gera o grafo.

## Loop autônomo sequencial

O loop usa **um Codex por vez**. Ao fim de um ciclo, o agente cria um handoff
atômico em `.agent-loop/handoffs/completed`; somente o runner externo o valida
e inicia o próximo. O agente nunca aciona seu sucessor diretamente.

Antes do primeiro uso, configure uma identidade Git local para os commits de
cada ciclo:

```bash
git init
git config user.name "Seu nome"
git config user.email "seu-email@exemplo.com"
git add .
git commit -m "chore: bootstrap autonomous agent loop"
```

Então execute, a partir da raiz do projeto:

```bash
npm run loop:start
```

O limite é de 20 handoffs aceitos, configurado em `agent-loop.config.json`.
O runner exige árvore Git limpa, um novo commit por ciclo, testes configurados,
um handoff válido e nenhuma flag `.agent-loop/STOP` antes de iniciar o próximo.

```bash
npm run loop:status
npm run loop:stop
```

`loop:stop` não mata o Codex ativo; ele evita que o sucessor seja iniciado e
permite que o ciclo atual termine de forma coerente. Os logs e mensagens finais
de cada ciclo ficam em `.agent-loop/runs/`; estado, handoffs e logs são
ignorados pelo Git para não violar a exigência de árvore limpa.

## Política de fidelidade

O compilador falha quando:

- não existe `config.json` para Safetensors;
- a arquitetura não tem adaptador registrado;
- um tensor crítico está ausente ou ambíguo;
- há QKV fundido sem layout conhecido;
- há MoE/AltUp/LAuReL ou outra semântica ainda não implementada.

Ele não cria bypasses nem zeros para componentes desconhecidos.

## Limites atuais

O adaptador atual cobre blocos decoder-only auditáveis de Llama, Mistral, Qwen 2/3 e Gemma 1/2/3-text. Modelos com código remoto, state-space layers, linear attention, MoE, multimodal completo, Gemma 3n/4 ou layouts QKV especiais precisam de adaptadores próprios ou extração do grafo do runtime oficial. Gemma 4 é rejeitado de propósito porque sua topologia inclui PLE, heads por tipo de camada, KV sharing e outras semânticas que o bloco genérico não representa.

A equivalência real deve ser confirmada por um validador diferencial: mesma entrada, mesmo dtype, comparação de embeddings, saída por camada, KV cache e logits contra Transformers/MLX/llama.cpp.


## O que “agnóstico” significa

O núcleo é agnóstico ao contêiner e à quantização por usar backends separados, mas a semântica da arquitetura não pode ser deduzida apenas pelos nomes dos tensores. Cada família precisa de um adaptador validado que leia `config.json`/metadados GGUF e construa o grafo exato. Arquiteturas não reconhecidas falham de forma explícita.

## Regra de igualdade

O IR não declara igualdade bitwise apenas por ter os mesmos pesos. Para reproduzir o runtime original também são necessários dtype de cálculo, dtype de acumulação, ordem das reduções, máscara, RoPE, cache KV e arredondamentos. O campo `fidelity` mantém a validação diferencial como requisito obrigatório.

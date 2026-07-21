export const gemma4RealCompareHtml = `<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Gemma 4 — BF16 × aritmética real simplificada</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#0b0d12; color:#eef1f7; }
    * { box-sizing:border-box; } body { margin:0; } main { width:min(1180px,calc(100% - 32px)); margin:36px auto 72px; }
    h1 { font-size:clamp(24px,4vw,40px); margin:0 0 8px; } .sub { color:#9ca6ba; margin:0 0 28px; }
    .panel,.result { background:#121620; border:1px solid #252c3b; border-radius:14px; padding:18px; }
    label { display:block; font-size:13px; color:#abb5c8; margin-bottom:7px; } textarea { width:100%; min-height:110px; resize:vertical; }
    textarea,input,select { color:#f6f7fb; background:#090c12; border:1px solid #343d50; border-radius:9px; padding:11px; font:inherit; }
    .actions { display:flex; gap:12px; align-items:end; margin-top:14px; } .field { width:150px; }
    button { border:0; border-radius:9px; padding:12px 18px; background:#7c5cff; color:white; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.5; cursor:wait; } #status { color:#aeb8cb; font-size:13px; }
    .grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin-top:16px; } .result h2 { margin:0 0 12px; font-size:17px; }
    .text { white-space:pre-wrap; min-height:72px; padding:12px; background:#090c12; border-radius:8px; }
    .tokens { color:#9eabca; overflow-wrap:anywhere; font-family:ui-monospace,monospace; font-size:12px; margin-top:10px; }
    .summary { margin-top:16px; } .ok { color:#65d394; } .bad { color:#ff7b89; }
    table { width:100%; border-collapse:collapse; font-size:13px; margin-top:12px; } th,td { text-align:left; border-bottom:1px solid #262d3c; padding:9px 7px; } th { color:#9ca6ba; }
    details { margin-top:14px; } pre { overflow:auto; background:#090c12; padding:12px; border-radius:8px; font-size:11px; }
    @media(max-width:760px){.grid{grid-template-columns:1fr}.actions{align-items:stretch;flex-direction:column}.field{width:100%}}
  </style>
</head>
<body><main>
  <h1>Gemma 4: execução diferencial real</h1>
  <p class="sub">Transformers eager BF16, matemática recomposta de compatibilidade e executor compilado direto sobre constant pool binário, com geração e KV cache reais.</p>
  <section class="panel">
    <label for="prompt">Prompt</label>
    <textarea id="prompt">The capital of France is</textarea>
    <div class="actions"><div class="field"><label for="tokens">Novos tokens</label><input id="tokens" type="number" min="1" max="64" value="3"></div><div class="field"><label for="threads">Threads (0 = automático)</label><input id="threads" type="number" min="0" max="256" value="0"></div><div class="field"><label for="precision">Precisão compilada</label><select id="precision"><option value="f32">F32</option><option value="f64">F64</option></select></div><div class="field"><label for="rounding">Arredondamento</label><select id="rounding"><option value="none">Somente final</option><option value="layer-bf16">BF16 por camada</option><option value="operation-bf16">BF16 por operação</option></select></div>
      <button id="run">Gerar e comparar</button><span id="status">Pronto.</span></div>
  </section>
  <div class="grid">
    <section class="result"><h2>Original — BF16</h2><div id="baselineText" class="text">—</div><div id="baselineTokens" class="tokens"></div></section>
    <section class="result"><h2 id="candidateTitle">Compilado (compatibilidade) — F32/F64 → BF16 final</h2><div id="candidateText" class="text">—</div><div id="candidateTokens" class="tokens"></div></section>
    <section class="result" id="directPanel" hidden><h2>Compilado direto — pool binário + tiles nativos</h2><div id="directText" class="text">—</div><div id="directTokens" class="tokens"></div></section>
  </div>
  <section class="panel summary" id="summary" hidden><strong id="verdict"></strong><div id="performance" class="tokens"></div><table><thead><tr><th>Passo</th><th>Tokens orig. / compat. / direto</th><th>Argmax compat. / direto</th><th>Logits BF16 divergentes</th><th>Erro máx.</th><th>Tempo orig. / compat. / direto</th></tr></thead><tbody id="steps"></tbody></table>
    <details><summary>Relatório JSON completo</summary><pre id="json"></pre></details></section>
</main>
<script>
const q = id => document.getElementById(id), run=q('run');
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
run.addEventListener('click', async () => {
  const requestStarted=Date.now(); run.disabled=true; q('status').textContent='Calculando os executores persistentes…'; q('summary').hidden=true; q('directPanel').hidden=true;
  try {
    const response=await fetch('/api/compare',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({prompt:q('prompt').value,maxNewTokens:Number(q('tokens').value),threads:Number(q('threads').value),precision:q('precision').value,roundingPolicy:q('rounding').value})});
    const data=await response.json(); if(!response.ok) throw new Error(data.error || 'Falha desconhecida');
    q('baselineText').textContent=data.baselineFullText; q('candidateText').textContent=data.candidateFullText;
    q('candidateTitle').textContent='Compilado (compatibilidade) — '+data.candidatePrecision.toUpperCase()+' / '+data.roundingPolicy+' → BF16 final';
    q('baselineTokens').textContent=JSON.stringify(data.baselineGeneratedTokenIds); q('candidateTokens').textContent=JSON.stringify(data.candidateGeneratedTokenIds);
    if(data.direct){q('directPanel').hidden=false;q('directText').textContent=data.direct.fullText;q('directTokens').textContent=JSON.stringify(data.direct.generatedTokenIds);}
    const allEqual=data.generatedTokensEqual&&(!data.direct||data.direct.tokensEqualBaseline); q('verdict').className=allEqual?'ok':'bad'; q('verdict').textContent=allEqual?'Todos os executores geraram os mesmos tokens.':'Há divergência: compatibilidade no passo '+data.firstDivergentStep+'; direto no passo '+(data.direct?.firstDivergentStep??'—')+'.';
    q('performance').textContent='Threads compat.: '+data.executionThreads+' · original: '+data.performance.baselineTokensPerSecond.toFixed(2)+' tok/s · compat.: '+data.performance.candidateTokensPerSecond.toFixed(2)+' tok/s · razão compat.: '+data.performance.candidateSpeedup.toFixed(2)+'×'+(data.direct?' · direto: '+data.direct.tokensPerSecond.toFixed(2)+' tok/s em '+data.direct.elapsedSeconds.toFixed(2)+'s · '+data.direct.linearBackend+' · threads diretas: '+data.direct.linearThreads+' · lotes lineares: '+(data.direct.linearBatchDispatches??0)+' ('+(data.direct.linearBatchedProjectionTiles??0)+' tiles)':'')+' · pico RSS compat.: '+(data.performance.processPeakRssBytes/1073741824).toFixed(2)+' GiB';
    q('steps').innerHTML=data.steps.map((s,index)=>{const d=data.direct?.steps?.[index];const directEqual=d?s.baselineToken===d.tokenId:null;return '<tr><td>'+s.step+'</td><td>'+s.baselineToken+' / '+s.candidateToken+' / '+(d?.tokenId??'—')+'</td><td>'+(s.metrics.argmaxEqual?'igual':'diferente')+' / '+(directEqual===null?'—':directEqual?'igual':'diferente')+'</td><td>'+(100*s.metrics.divergenceRate).toFixed(4)+'%</td><td>'+s.metrics.maxAbsError+'</td><td>'+s.baselineSeconds.toFixed(3)+'s / '+s.candidateSeconds.toFixed(3)+'s / '+(d?d.forwardSeconds.toFixed(3)+'s':'—')+'</td></tr>';}).join('');
    q('json').textContent=JSON.stringify(data,null,2); q('summary').hidden=false; q('status').textContent='Concluído em '+((Date.now()-requestStarted)/1000).toFixed(2)+'s totais.';
  } catch(error) { q('status').textContent='Erro: '+error.message; }
  finally { run.disabled=false; }
});
</script></body></html>`;

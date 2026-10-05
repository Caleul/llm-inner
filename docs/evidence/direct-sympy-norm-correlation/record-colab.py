"""Audit Colab's actual-file recovery and preserve the local test map."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
logs=Path('artifacts/direct-sympy-input-partitions/norm-correlation-diagnosis')
out=root/'colab';out.mkdir(exist_ok=True)
original=json.loads((logs/'colab-report.json').read_text())
recovery=json.loads((logs/'colab-recovery.json').read_text())
local=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert original['terminal'] and not original['success'] and 'bracket nesting level exceeded' in original['error']
assert recovery['terminal'] and recovery['success'] and not recovery['recompiled']
assert original['compilerIdentity']==recovery['compilerIdentity']
assert local['sources']==recovery['compilerIdentity']['sources']
assert local['checkpoint']==recovery['compilerIdentity']['checkpoint']
assert recovery['artifactSHA256']==digest_file(root/'mixed.expr')
assert original['mixedCompilation']['artifact']['sha256']==recovery['artifactSHA256']
assert recovery['corpusParity']['verifierSHA256']==digest_file(Path('helpers/direct_sympy_region_parity.py'))
assert recovery['corpusParity']['mismatches']==0 and recovery['corpusParity']['cases']==8196
assert len(recovery['exhaustive'])==2 and all(r['exitCode']==0 for r in recovery['exhaustive'])
exhaustive=[]
for name in ('colab-exhaustive-negative-positive','colab-exhaustive-positive-negative'):
    report=json.loads((logs/(name+'.json')).read_text())
    assert report['complete'] and report['mismatches']==0 and report['cases']==20980737
    assert report['sha256']==recovery['artifactSHA256']
    assert report['compilerIdentity']==recovery['compilerIdentity']
    assert report['verifierSHA256']==digest_file(root/'exhaustive.py')
    exhaustive.append(report)
    (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
cuda=json.loads((logs/'colab-cuda.log').read_text())
assert cuda['halfProducts']['cases']==888832 and cuda['halfProducts']['mismatches']==0
assert cuda['silu']['mismatches']==0 and cuda['silu']['cudaAdmitted']
assert cuda['quadraticThreshold']['cudaAdmitted']
assert all(cuda['quadraticThreshold'][k]==0 for k in ('cpuMismatches','cudaMismatches','squareBitMismatches'))
first,second=original['comparison']
assert first['artifactHashes']==second['artifactHashes']
assert all(p['mismatches']==0 for r in original['comparison'] for p in r['parity'])
validation={'session':'llm-inner-norm-correlation','compilerIdentity':recovery['compilerIdentity'],
    'sameNumericalSourcesAndCheckpointAsLocal':True,'localArtifact':'mixed.expr','artifactSHA256':recovery['artifactSHA256'],
    'corpusCases':8196,'corpusMismatches':0,'exhaustiveArtifactCases':sum(r['cases'] for r in exhaustive),
    'exhaustiveArtifactMismatches':0,'artifactRecompiledDuringRecovery':False,
    'clangParserFix':'-fbracket-depth=4096; original FP flags and expression retained',
    'comparison':{'patterns':original['comparisonPatterns'],'identicalArtifactHashes':True,
        'sequentialSeconds':first['seconds'],'parallelSeconds':second['seconds'],'speedRatio':original['speedRatio'],
        'sequentialPeakObservedRSSBytes':first['memory']['peakObservedRSSBytes'],
        'parallelPeakObservedRSSBytes':second['memory']['peakObservedRSSBytes'],
        'scope':original['comparisonScope']},
    'mixedCompilationSeconds':original['mixedCompilation']['seconds'],
    'cuda':cuda,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name,data in (('report-original.json',original),('recovery.json',recovery),('cuda.json',cuda),('validation.json',validation)):
    (out/name).write_text(json.dumps(data,indent=2)+'\n')
for name in ('colab-cuda.log','colab-recovery.log','colab-exhaustive-negative-positive.log','colab-exhaustive-positive-negative.log'):
    (out/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
mapfile=Path('docs/direct-string-validation.json');raw=mapfile.read_text()
existing=json.loads(raw)
if 'colabNormCorrelationValidation' in existing:
    if existing['colabNormCorrelationValidation']!=validation:
        raise ValueError('Historical Colab validation differs; append a new evidence entry instead')
else:
    assert raw.rstrip().endswith('}')
    mapfile.write_text(raw.rstrip()[:-1].rstrip()+',\n  "colabNormCorrelationValidation": '+json.dumps(validation,indent=2).replace('\n','\n  ')+'\n}\n')
print(json.dumps(validation),flush=True)

"""Audit fresh geometry, actual expressions, exhaustive parity and test history."""
import json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups
from direct_sympy_savepoints import digest_file

root=Path(__file__).resolve().parent
logs=Path('artifacts/direct-sympy-input-partitions/norm-correlation-diagnosis')
directory=Path('artifacts/direct-sympy-input-partitions/norm-correlation-state')
previous=root.parent/'direct-sympy-coupled-projections'
old=json.loads((previous/'frontier.json').read_text())
state=json.loads((directory/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert state['geometryReplan']['completeLeavesReset']==sum(n['status']=='complete' for n in old['tree'].values())
assert not state['geometryReplan']['numericalResultsReused']
assert not state['finalArtifactEmitted'] and not state['finalParity']
assert all(promote_tree(state['tree'],state['root'],n['domains'],{})[2]==0 for n in old['tree'].values() if n['status']=='complete')
assert all(p['compilerIdentity']==state['identity'] and p['parity']['mismatches']==0 for p in state['coveragePromotions'])
assert sum(p['addedInputPatterns'] for p in state['coveragePromotions'])==state['coveredInputPatterns']
references={}
for row in json.loads((previous/'artifact-map.json').read_text()):
    path=(previous/row['file']).resolve()
    assert digest_file(path)==row['sha256'];references[row['sha256']]=path
rows=[]
for key,node in state['tree'].items():
    if node['status']!='complete':continue
    sha=node['artifact']['sha256'];path=directory/node['artifact']['file']
    assert digest_file(path)==sha
    destination=references.get(sha)
    if destination is None:
        destination=root/('mixed.expr' if sha=='62a1d627e115eb51db84daf3e184110768a7b9a4b018a159ed81e6c895a1050b' else sha+'.expr')
        destination.write_bytes(path.read_bytes());references[sha]=destination.resolve()
    assert destination.read_bytes()==path.read_bytes()
    rows.append({'region':key,'domains':node['domains'],'file':os.path.relpath(destination,root),
        'sha256':sha,'characters':node['artifact']['characters']})
_,_,grouped=groups(directory,state['tree'])
exhaustive=[]
for name in ('current-exhaustive-negative-positive','current-exhaustive-positive-negative'):
    report=json.loads((logs/(name+'.json')).read_text())
    assert report['compilerIdentity']==state['identity'] and report['complete'] and report['cases']==20980737
    assert report['mismatches']==0 and report['verifierSHA256']==digest_file(root/'exhaustive.py')
    assert digest_file(report['artifact'])==report['sha256']
    assert promote_tree(state['tree'],state['root'],report['inputRankDomains'],{})[2]==0
    report['artifact']=os.path.relpath(references[report['sha256']],root)
    (root/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');exhaustive.append(report)
emission=json.loads((logs/'bounded-emission.json').read_text())
assert emission['artifact']['sha256']==exhaustive[0]['sha256']==exhaustive[1]['sha256']
assert emission['complete'] and emission['artifact']['complete'] and emission['artifact']['compilerAliases']==0
emission['artifact']['path']=os.path.relpath(references[emission['artifact']['sha256']],root)
(root/'emission.json').write_text(json.dumps(emission,indent=2)+'\n')
test=(logs/'corrected-test.log').read_text()
assert all(s in test for s in ('tests 31','pass 31','fail 0','skipped 0'))
assert 'pairs=83922948 distanceViolations=0 exclusionViolations=0' in (logs/'correlation-corrected-test.log').read_text()
project_test=(logs/'project-test.log').read_text()
failures=re.findall(r'^test at (.+)$',project_test,re.M)
baseline=(previous/'project-test.log').read_text()
assert len(failures)==15 and failures==re.findall(r'^test at (.+)$',baseline,re.M)
def totals(text):return {key:int(value) for key,value in re.findall(r'^ℹ (tests|pass|fail|skipped) (\d+)',text,re.M)}
assert totals(project_test)=={'tests':625,'pass':570,'fail':15,'skipped':40}
benchmark=json.loads((logs/'parallel-comparison.json').read_text())
assert benchmark['compilerIdentity']==state['identity'] and benchmark['identicalDomainsAndArtifactHashes']
assert benchmark['sequential']['artifactHashes']==benchmark['parallel']['artifactHashes']
assert all(r['mismatches']==0 for mode in ('sequential','parallel') for r in benchmark[mode]['parity'])
(root/'parallel-comparison.json').write_text(json.dumps(benchmark,indent=2)+'\n')
validation={'checkpoint':'docs/evidence/direct-sympy-test-checkpoint','compilerIdentity':state['identity'],
    'testFile':'test/direct-sympy-strings.test.ts','pythonTests':'helpers/direct_sympy_norm_correlation_test.py',
    'integrationTests':31,'passed':31,'failed':0,'skipped':0,
    'wholeProjectSuite':{'current':totals(project_test),'sameFailingTestLocations':failures,
        'baselineEvidence':'../direct-sympy-coupled-projections/validation.json','newFailingTestLocations':0},
    'constraintExhaustivePairs':83922948,'constraintViolations':0,
    'exhaustiveArtifactCases':sum(r['cases'] for r in exhaustive),'exhaustiveArtifactMismatches':0,
    'mixedArtifactCharacters':emission['artifact']['characters'],'mixedArtifactPaths':emission['artifact']['paths'],
    'freshPromotions':len(state['coveragePromotions']),
    'freshCorpusCases':sum(p['parity']['cases'] for p in state['coveragePromotions']),'freshCorpusMismatches':0,
    'previousCoveredInputPatterns':old['coveredInputPatterns'],
    'coveredInputPatterns':state['coveredInputPatterns'],'unfinishedInputPatterns':state['unfinishedInputPatterns'],
    'addedInputPatterns':state['coveredInputPatterns']-old['coveredInputPatterns'],
    'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
    'expressionRepresentation':'SymPy-compatible mathematical strings',
    'parallelComparison':{'report':'parallel-comparison.json','coveredPatterns':benchmark['coveredPatterns'],
        'speedRatio':benchmark['speedRatio'],'identicalDomainsAndArtifactHashes':True,'fullCoordinateTiming':False,
        'sequentialSeconds':benchmark['sequential']['seconds'],'parallelSeconds':benchmark['parallel']['seconds'],
        'sequentialPeakObservedRSSBytes':benchmark['sequential']['memory']['peakObservedRSSBytes'],
        'parallelPeakObservedRSSBytes':benchmark['parallel']['memory']['peakObservedRSSBytes']},
    'coalescing':{key:value for key,value in grouped.items() if key!='rectangles'},
    'rejectedWidening':{'inputRankDomains':{'X1':[-16384,-10240],'X2':[10241,15360]},
        'addedInputPatterns':0,'reason':'Larger region retains an additional MLP normalization; the two-vector proof test was invalid there. Compilation stopped at the flat artifact limit; no result admitted.'},
    'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('build.log','corrected-test.log','project-test.log','correlation-corrected-test.log','recompilation-central.log',
    'current-exhaustive-negative-positive.log','current-exhaustive-positive-negative.log','recompilation.log','correlation-wide-test.log','benchmark.log'):
    (root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
for name,data in (('validation.json',validation),('artifact-map.json',rows),('frontier.json',state)):
    (root/name).write_text(json.dumps(data,indent=2)+'\n')
mapfile=Path('docs/direct-string-validation.json');raw=mapfile.read_text()
existing=json.loads(raw)
if 'normCorrelationValidation' in existing:
    if existing['normCorrelationValidation']!=validation:
        raise ValueError('Historical validation differs; append a new evidence entry instead')
else:
    assert raw.rstrip().endswith('}')
    mapfile.write_text(raw.rstrip()[:-1].rstrip()+',\n  "normCorrelationValidation": '+json.dumps(validation,indent=2).replace('\n','\n  ')+'\n}\n')
print(json.dumps(validation),flush=True)

"""Audit the matching-work benchmark and compatible live parallel admission."""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/dominant-square-state')
s=json.loads((state/'frontier.json').read_text())
before=json.loads((root/'frontier-before.json').read_text())
r=json.loads((root/'continuation.json').read_text())
p=json.loads((root/'partition-parity.json').read_text())
b=json.loads((root/'benchmark.json').read_text())
assert s['identity']==before['identity']==r['compilerIdentity']==p['compilerIdentity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert r['controllerSHA256']==hashlib.sha256(Path('helpers/direct_sympy_partition_parallel.py').read_bytes()).hexdigest()
assert audit_tree(s['tree'],s['root'])==(s['coveredInputPatterns'],s['unfinishedInputPatterns'])
assert p['coveredInputPatterns']==r['coveredInputPatterns']==s['coveredInputPatterns']
assert r['nativeMismatches']==p['mismatches']==0
old=[key for key,node in before['tree'].items() if node['status']=='complete']
assert all(s['tree'][key]==before['tree'][key] for key in old)
assert r['addedInputPatterns']==s['coveredInputPatterns']-before['coveredInputPatterns']
for benchmark in (b['sequential'],b['parallel']):
    assert benchmark['compilerIdentity']==s['identity'] and benchmark['controllerSHA256']==r['controllerSHA256']
    assert benchmark['attempts']==4 and benchmark['addedInputPatterns']==553648128 and benchmark['nativeMismatches']==0
    assert all(w['peakObservedRSSBytes']<=benchmark['memoryBudgetBytes'] for w in benchmark['waves'])
assert b['identicalDomainsAndArtifactHashes']
assert all(w['peakObservedRSSBytes']<=r['memoryBudgetBytes'] for w in r['waves'])
log=(root/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',26),('pass',26),('fail',0),('skipped',0)))
rows=[]
for key,node in s['tree'].items():
    if node['status']!='complete':continue
    body=(state/node['artifact']['file']).read_bytes();sha=hashlib.sha256(body).hexdigest()
    assert sha==node['artifact']['sha256']
    filename='region-'+sha+'.expr';(root/filename).write_bytes(body)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(body)})
assert len(rows)==p['completedRegions']
validation={'build':True,'tests':26,'pass':26,'fail':0,'skipped':0,'parallelRegionTests':3,
 'benchmarkIdenticalDomainsAndHashes':True,'benchmarkCoveredPatterns':553648128,
 'benchmarkSequentialSeconds':b['sequential']['seconds'],'benchmarkParallelSeconds':b['parallel']['seconds'],
 'benchmarkSpeedRatio':b['speedRatio'],'benchmarkSequentialPeakObservedRSSBytes':max(w['peakObservedRSSBytes'] for w in b['sequential']['waves']),
 'benchmarkParallelPeakObservedRSSBytes':max(w['peakObservedRSSBytes'] for w in b['parallel']['waves']),
 'coveredBefore':before['coveredInputPatterns'],'addedInputPatterns':r['addedInputPatterns'],
 'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],
 'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],'completedRegions':len(rows),
 'distinctArtifacts':len({row['sha256'] for row in rows}),'oldLeavesPreserved':len(old),
 'parallelAttempts':r['attempts'],'parallelCompleted':sum(w['completed'] for w in r['waves']),
 'parallelNativeCases':r['nativeCases'],'allLeafNativeCases':p['cases'],'mismatches':0,
 'livePeakObservedCompileRSSBytes':max(w['peakObservedRSSBytes'] for w in r['waves']),
 'memoryBudgetBytes':r['memoryBudgetBytes'],'numericalCompilerSourcesChanged':False,
 'fullCoordinateSpeedupMeasured':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-parallel-regions/validation.json',**validation}
if 'parallelRegionValidation' not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "parallelRegionValidation": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)['parallelRegionValidation']==entry
for logfile in root.glob('*.log'):
    logfile.write_text('\n'.join(line.rstrip() for line in logfile.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))

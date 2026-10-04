"""Audit deadline correction, all compatible artifacts and exhaustive probe."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree,cardinality
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/dominant-square-state')
s=json.loads((state/'frontier.json').read_text());before=json.loads((root/'frontier-before.json').read_text())
p=json.loads((root/'final-parity.json').read_text())
reports=[json.loads((root/(name+'.json')).read_text()) for name in ('continuation','extended-continuation','retry-continuation')]
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert before['identity']==s['identity']==p['compilerIdentity']==identity
assert all(r['compilerIdentity']==identity and r['nativeMismatches']==0 for r in reports)
previous_controller=hashlib.sha256(subprocess.check_output(['git','show','50d7d2d:helpers/direct_sympy_partition_parallel.py'])).hexdigest()
assert reports[0]['controllerSHA256']==reports[1]['controllerSHA256']==previous_controller
assert reports[2]['controllerSHA256']==hashlib.sha256(Path('helpers/direct_sympy_partition_parallel.py').read_bytes()).hexdigest()
assert audit_tree(s['tree'],s['root'])==(s['coveredInputPatterns'],s['unfinishedInputPatterns'])
assert p['coveredInputPatterns']==s['coveredInputPatterns'] and p['mismatches']==0
old=[key for key,node in before['tree'].items() if node['status']=='complete']
assert all(s['tree'][key]==before['tree'][key] for key in old)
assert sum(r['addedInputPatterns'] for r in reports)==s['coveredInputPatterns']-before['coveredInputPatterns']
probe=json.loads((root/'extended-budget.json').read_text())
exhaustive=json.loads((root/'exhaustive-parity.json').read_text())
assert probe['complete'] and probe['artifact']['sha256']==exhaustive['sha256']==hashlib.sha256((root/'probe.expr').read_bytes()).hexdigest()
assert exhaustive['fullRegionEnumeration'] and exhaustive['cases']==cardinality(probe['inputDomains'])==40960 and exhaustive['mismatches']==0
assert promote_tree(s['tree'],s['root'],probe['inputDomains'],{})[2]==0
retry_attempts=[r for r in s['attempts'] if r.get('parallelRun') and r.get('timeoutSeconds')==30 and r['complete']]
recovered=[]
for attempt in retry_attempts:
    earlier=[a for a in s['attempts'] if a['region']==attempt['region'] and a.get('timeoutSeconds')==10 and 'wall-clock' in a.get('stop','')]
    if earlier:recovered.append({'region':attempt['region'],'domains':attempt['inputDomains'],'timeouts':[10,30]})
assert len(recovered)==2
log=(root/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',26),('pass',26),('fail',0),('skipped',0)))
assert 'Ran 4 tests' in (root/'unit.log').read_text()
rows=[]
for key,node in s['tree'].items():
    if node['status']!='complete':continue
    body=(state/node['artifact']['file']).read_bytes();sha=hashlib.sha256(body).hexdigest()
    assert sha==node['artifact']['sha256']
    filename='region-'+sha+'.expr'
    previous=root.parent/'direct-sympy-parallel-regions'/filename
    if previous.exists() and hashlib.sha256(previous.read_bytes()).hexdigest()==sha:
        (root/filename).unlink(missing_ok=True)
        filename='../direct-sympy-parallel-regions/'+filename
    else:(root/filename).write_bytes(body)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(body)})
assert len(rows)==p['completedRegions']
validation={'build':True,'tests':26,'pass':26,'fail':0,'skipped':0,'parallelRegionTests':4,
 'coveredBefore':before['coveredInputPatterns'],'addedInputPatterns':s['coveredInputPatterns']-before['coveredInputPatterns'],
 'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],
 'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],'completedRegions':len(rows),
 'distinctArtifacts':len({row['sha256'] for row in rows}),'oldLeavesPreserved':len(old),
 'parallelAttempts':sum(r['attempts'] for r in reports),'parallelCompleted':sum(w['completed'] for r in reports for w in r['waves']),
 'newNativeCases':sum(r['nativeCases'] for r in reports),'allLeafNativeCases':p['cases'],'mismatches':0,
 'peakObservedCompileRSSBytes':max(w['peakObservedRSSBytes'] for r in reports for w in r['waves']),
 'memoryBudgetBytes':6442450944,'deadlineRecoveredRegions':recovered,'probeCharacters':probe['artifact']['characters'],
 'exhaustiveProbeCases':exhaustive['cases'],'exhaustiveProbeMismatches':0,
 'numericalCompilerSourcesChanged':False,'fourWorkerSpeedupMeasured':False,
 'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-four-worker-continuation/validation.json',**validation}
if 'fourWorkerDeadlineValidation' not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "fourWorkerDeadlineValidation": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)['fourWorkerDeadlineValidation']==entry
for logfile in root.glob('*.log'):
    logfile.write_text('\n'.join(line.rstrip() for line in logfile.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))

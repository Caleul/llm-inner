"""Audit current-source artifacts and retain reproducible regional evidence."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups
from direct_sympy_savepoints import digest_file

root=Path(__file__).resolve().parent
logs=Path('artifacts/direct-sympy-input-partitions/central-guard-diagnosis')
directory=Path('artifacts/direct-sympy-input-partitions/coupled-projection-state')
previous=root.parent/'direct-sympy-early-projection'
old=json.loads((previous/'frontier.json').read_text());state=json.loads((directory/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert state['geometryReplan']['completeLeavesReset']==sum(n['status']=='complete' for n in old['tree'].values())
assert not state['geometryReplan']['numericalResultsReused']
assert not state['finalArtifactEmitted'] and not state['finalParity']
assert all(promote_tree(state['tree'],state['root'],n['domains'],{})[2]==0 for n in old['tree'].values() if n['status']=='complete')
assert all(p['compilerIdentity']==state['identity'] and not p['parity']['mismatches'] for p in state['coveragePromotions'])
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
        destination=root/('central-positive.expr' if sha=='b4920091a4eee6b90daa70e5350ea252563fc06774864a846da3473ab5889916' else sha+'.expr')
        destination.write_bytes(path.read_bytes());references[sha]=destination.resolve()
    assert destination.read_bytes()==path.read_bytes()
    rows.append({'region':key,'domains':node['domains'],'file':os.path.relpath(destination,root),
        'sha256':sha,'characters':node['artifact']['characters']})
_,_,grouped=groups(directory,state['tree'])
comparison=json.loads((logs/'comparison.json').read_text())
assert comparison['currentIdentity']==state['identity']
assert len(comparison['results'])==2
assert all(r['parity']['mismatches']==0 and digest_file(r['artifact']['path'])==r['artifact']['sha256'] for r in comparison['results'])
exhaustive=[]
for name in ('exhaustive-parity.json','exhaustive-negative-parity.json'):
    report=json.loads((logs/name).read_text())
    assert report['compilerIdentity']==state['identity'] and report['complete'] and report['cases']==20980737
    assert report['mismatches']==0 and report['verifierSHA256']==digest_file(root/'exhaustive.py')
    assert digest_file(report['artifact'])==report['sha256']
    report['artifact']=os.path.relpath(references[report['sha256']],root)
    (root/name).write_text(json.dumps(report,indent=2)+'\n');exhaustive.append(report)
failed=[]
for line in (logs/'extension.log').read_text().splitlines():
    if not line.startswith('{'):continue
    r=json.loads(line)
    if r.get('stop'):
        failed.append({'inputDomains':r['inputDomains'],'addedInputPatterns':r['addedInputPatterns'],
            'budgets':r['budgets'],'compilation':r.get('compilation'),'stop':r['stop']})
assert all(r['addedInputPatterns']==0 for r in failed)
negative_controls=[]
for name,rectangle in (('mixed-negative-positive-parity.json',{'X1':[-16384,-11264],'X2':[11264,15360]}),
    ('mixed-positive-negative-parity.json',{'X1':[11264,16384],'X2':[-15360,-11264]})):
    r=json.loads((logs/name).read_text())
    assert r['compilerIdentity']==state['identity'] and r['complete'] and r['cases']==20980737
    assert r['mismatches']==92413 and r['verifierSHA256']==digest_file(root/'exhaustive.py')
    assert promote_tree(state['tree'],state['root'],rectangle,{})[2]==r['cases']
    r['artifact']=os.path.relpath(references[r['sha256']],root)
    (root/name).write_text(json.dumps(r,indent=2)+'\n')
    negative_controls.append({'report':name,'cases':r['cases'],'mismatches':r['mismatches'],
        'inputDomains':rectangle,'admitted':False,'firstMismatches':r['firstMismatches']})
test=(logs/'test.log').read_text()
assert all(s in test for s in ('tests 30','pass 30','fail 0','skipped 0'))
project_test=(logs/'project-test.log').read_text();baseline_test=(logs/'baseline-test.log').read_text()
failures=re.findall(r'^test at (.+)$',project_test,re.M)
assert len(failures)==15 and failures==re.findall(r'^test at (.+)$',baseline_test,re.M)
baseline_identity=json.loads((logs/'baseline-identity.json').read_text())
assert baseline_identity['head'].startswith('abf5d72')
def totals(text):return {key:int(value) for key,value in re.findall(r'^ℹ (tests|pass|fail|skipped) (\d+)',text,re.M)}
whole_suite={'current':totals(project_test),'isolatedPreviousCommit':totals(baseline_test),
    'baselineIdentity':baseline_identity,'sameFailingTestLocations':failures,
    'newFailuresAmongExecutedTests':0,'scope':'Full current npm test; failing files rebuilt and rerun from isolated preceding HEAD.'}
assert whole_suite['current']=={'tests':624,'pass':570,'fail':15,'skipped':39}
assert whole_suite['isolatedPreviousCommit']=={'tests':31,'pass':16,'fail':15,'skipped':0}
before,after=comparison['results']
validation={'checkpoint':'docs/evidence/direct-sympy-test-checkpoint','compilerIdentity':state['identity'],
    'testFile':'test/direct-sympy-strings.test.ts','pythonTests':'helpers/direct_sympy_projection_constraints_test.py',
    'integrationTests':30,'passed':30,'failed':0,'skipped':0,
    'wholeProjectSuite':whole_suite,
    'constraintExhaustivePairs':20980737,'constraintViolations':0,
    'matchedRegionCharactersBefore':before['artifact']['characters'],'matchedRegionCharactersAfter':after['artifact']['characters'],
    'matchedRegionPathsBefore':before['artifact']['paths'],'matchedRegionPathsAfter':after['artifact']['paths'],
    'artifactReductionFraction':1-after['artifact']['characters']/before['artifact']['characters'],
    'matchedCorpusCases':sum(r['parity']['cases'] for r in comparison['results']),'matchedCorpusMismatches':0,
    'exhaustiveArtifactCases':sum(r['cases'] for r in exhaustive),'exhaustiveArtifactMismatches':0,
    'freshPromotions':len(state['coveragePromotions']),'freshCorpusCases':sum(p['parity']['cases'] for p in state['coveragePromotions']),
    'freshCorpusMismatches':0,'previousCoveredInputPatterns':old['coveredInputPatterns'],
    'coveredInputPatterns':state['coveredInputPatterns'],'unfinishedInputPatterns':state['unfinishedInputPatterns'],
    'addedInputPatterns':state['coveredInputPatterns']-old['coveredInputPatterns'],
    'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
    'expressionRepresentation':'SymPy-compatible mathematical strings',
    'coalescing':{key:value for key,value in grouped.items() if key!='rectangles'},
    'failedRegions':failed,'crossDomainNegativeControls':negative_controls,
    'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('build.log','test.log','project-test.log','baseline-test.log','baseline-build.log','projection.log','comparison.log','recompilation.log','exhaustive.log','exhaustive-negative.log'):
    (root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
for name,data in (('validation.json',validation),('artifact-map.json',rows),('frontier.json',state),('comparison.json',comparison)):
    (root/name).write_text(json.dumps(data,indent=2)+'\n')
# Preserve the existing test map byte for byte, appending this validation.
mapfile=Path('docs/direct-string-validation.json');raw=mapfile.read_text()
current=json.loads(raw)
if 'coupledProjectionValidation' in current:
    # This turn's last evidence block only; preserve every historical byte.
    prefix=raw.split(',\n  "coupledProjectionValidation": ',1)
    if len(prefix)!=2:raise ValueError('Evidence block is not independently replaceable')
    raw=prefix[0]+'\n}\n'
assert raw.rstrip().endswith('}')
mapfile.write_text(raw.rstrip()[:-1].rstrip()+',\n  "coupledProjectionValidation": '+json.dumps(validation,indent=2).replace('\n','\n  ')+'\n}\n')
print(json.dumps(validation),flush=True)

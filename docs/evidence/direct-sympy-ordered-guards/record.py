"""Admit actual emitted bytes and exhaustive parity under a frozen source identity."""
import json,re,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import replan,audit_tree
from direct_sympy_savepoints import digest_file,atomic,canonical
root=Path(__file__).resolve().parent;logs=Path('artifacts/direct-sympy-input-partitions/ordered-guard-state')
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
r=json.loads((logs/'emission.json').read_text());e=json.loads((logs/'exhaustive.json').read_text());p=json.loads((logs/'parity.json').read_text())
assert r['compilerIdentity']==e['compilerIdentity']==identity
assert r['complete'] and r['artifact']['compilerAliases']==0
assert e['complete'] and e['cases']==462422016 and e['mismatches']==p['mismatches']==0
assert r['paths']['orderedGuardEliminations']==46 and r['paths']['guardProofBudgetStops']==0
assert r['artifact']['sha256']==e['sha256']==p['sha256']==digest_file(logs/'broad.expr')
assert r['artifact']['characters']==22507034
shutil.copyfile(logs/'broad.expr',root/'broad.expr');assert digest_file(root/'broad.expr')==e['sha256']
artifact={'file':'broad.expr','characters':e['characters'],'sha256':e['sha256']}
(root/'artifact-map.json').write_text(json.dumps([{'inputDomains':r['inputDomains'],**artifact}],indent=2)+'\n')
old=json.loads((root.parent/'direct-sympy-branch-projection/frontier.json').read_text())
state=replan(old,identity,old['root'])
state['geometryReplan']={'source':'../direct-sympy-branch-projection/frontier.json','numericalResultsReused':False,'completeLeavesReset':sum(n['status']=='complete'for n in old['tree'].values())}
state['tree'],admitted,added=promote_tree(state['tree'],state['root'],r['inputDomains'],artifact)
covered,pending=audit_tree(state['tree'],state['root']);assert covered==added==e['cases']
state.update(coveredInputPatterns=covered,unfinishedInputPatterns=pending)
state['coveragePromotions']=[{'compilerIdentity':identity,'inputDomains':r['inputDomains'],'coveredBefore':0,'coveredAfter':covered,'addedInputPatterns':added,'sharedArtifact':artifact,'parity':p,'exhaustiveParity':{'cases':e['cases'],'mismatches':0},'admittedLeaves':admitted,'fullCoordinateParity':False}]
atomic(root/'frontier.json',canonical(state))
for name in ('emission.json','exhaustive.json','parity.json','memory.json','proof.log','test-final.log','project-test.log','exhaustive.log'):
 if name.endswith('.log'):(root/name).write_text('\n'.join(line.rstrip()for line in (logs/name).read_text().splitlines())+'\n')
 else:shutil.copyfile(logs/name,root/name)
target=(logs/'test-final.log').read_text();assert re.search(r'pass 34\b',target)and re.search(r'fail 0\b',target)and re.search(r'skipped 0\b',target)
whole=(logs/'project-test.log').read_text();counts={k:int(re.search(r'ℹ '+k+r' (\d+)',whole)[1])for k in ('tests','pass','fail','skipped')};assert counts=={'tests':628,'pass':570,'fail':15,'skipped':43}
failures=re.findall(r'^test at (\S+)',whole,re.M);old_validation=json.loads((root.parent/'direct-sympy-branch-projection/validation.json').read_text())
assert sorted(failures)==sorted(old_validation['wholeProjectSuite']['sameFailingTestLocations'])
validation={'checkpoint':'docs/evidence/direct-sympy-test-checkpoint','compilerIdentity':identity,'expressionRepresentation':'SymPy mathematical strings','integrationTests':34,'passed':34,'failed':0,'skipped':0,'coherentPathPythonTests':12,'wholeProjectSuite':{'current':counts,'sameFailingTestLocations':failures,'newFailingTestLocations':0},'artifact':artifact,'artifactPaths':28,'orderedGuardEliminations':46,'previousSerializedCharacters':30039520,'exhaustiveArtifactCases':e['cases'],'exhaustiveArtifactMismatches':0,'sampleCases':p['cases'],'sampleMismatches':0,'compilationSeconds':r['seconds'],'exhaustiveSeconds':e['seconds'],'coveredInputPatterns':covered,'unfinishedInputPatterns':pending,'historicalNumericalResultsReused':False,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'parallelSpeedupMeasured':False,'CUDAUsed':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');raw=path.read_text();previous=json.loads(raw);key='orderedGuardValidation'
if key in previous:assert previous[key]==validation
else:
 assert raw.rstrip().endswith('}')
 new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(validation,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:v for k,v in json.loads(new).items()if k!=key}==previous
 path.write_text(new)
print(json.dumps({'artifactCharacters':artifact['characters'],'exhaustivePairs':e['cases'],'mismatches':0,'covered':covered,'pending':pending,'fullCoordinateParity':False}))

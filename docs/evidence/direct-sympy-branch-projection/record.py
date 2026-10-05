"""Admit fresh artifact evidence; retain history without copying identical bytes."""
import json,re,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import replan,audit_tree
from direct_sympy_savepoints import digest_file,atomic,canonical
root=Path(__file__).resolve().parent
logs=Path('artifacts/direct-sympy-input-partitions/branch-projection-state')
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
r=json.loads((logs/'regional.json').read_text());e=json.loads((logs/'exhaustive.json').read_text());d=json.loads((logs/'diagnostic.json').read_text())
assert r['complete'] and r['artifact']['compilerAliases']==0
assert r['compilerIdentity']==e['compilerIdentity']==d['compilerIdentity']==identity
assert r['parity']['mismatches']==e['mismatches']==0 and e['complete'] and e['cases']==67125249
assert digest_file(logs/'mixed.expr')==e['sha256']==r['artifact']['sha256']
old_map=json.loads((root.parent/'direct-sympy-norm-projection/artifact-map.json').read_text())
old_file=next(row['file'] for row in old_map if row['sha256']==e['sha256'])
shared=(root.parent/'direct-sympy-norm-projection'/old_file).resolve()
assert digest_file(shared)==e['sha256'] and shared.stat().st_size==e['characters']
reference='../direct-sympy-directed-residual/mixed.expr'
assert (root/reference).resolve()==shared
artifact={'file':reference,'sha256':e['sha256'],'characters':e['characters']}
(root/'artifact-map.json').write_text(json.dumps([{'domains':r['inputDomains'],**artifact}],indent=2)+'\n')
old=json.loads((root.parent/'direct-sympy-norm-projection/frontier.json').read_text())
state=replan(old,identity,old['root'])
state['geometryReplan']={'source':'../direct-sympy-norm-projection/frontier.json','numericalResultsReused':False,'completeLeavesReset':sum(n['status']=='complete' for n in old['tree'].values())}
state['tree'],admitted,added=promote_tree(state['tree'],state['root'],r['inputDomains'],artifact)
covered,pending=audit_tree(state['tree'],state['root']);assert covered==added==67125249
state.update(coveredInputPatterns=covered,unfinishedInputPatterns=pending)
state['coveragePromotions']=[{'compilerIdentity':identity,'inputDomains':r['inputDomains'],'addedInputPatterns':added,'coveredBefore':0,'coveredAfter':covered,'sharedArtifact':artifact,'parity':r['parity'],'exhaustiveParity':{'cases':e['cases'],'mismatches':0},'admittedLeaves':admitted,'fullCoordinateParity':False}]
atomic(root/'frontier.json',canonical(state))
for name in ('regional.json','exhaustive.json','diagnostic.json','rms-proof.log','projection-proof.log','integration-final.log','full-suite.log'):
 if name.endswith('.log'):(root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines())+'\n')
 else:shutil.copyfile(logs/name,root/name)
integration=(logs/'integration-final.log').read_text();assert re.search(r'pass 34\b',integration) and re.search(r'fail 0\b',integration) and re.search(r'skipped 0\b',integration)
whole=(logs/'full-suite.log').read_text()
counts={k:int(re.search(r'ℹ '+k+r' (\d+)',whole)[1]) for k in ('tests','pass','fail','skipped')}
assert counts=={'tests':628,'pass':570,'fail':15,'skipped':43}
failures=re.findall(r'^test at (\S+)',whole,re.M)
previous=json.loads((root.parent/'direct-sympy-norm-projection/validation.json').read_text())['wholeProjectSuite']['sameFailingTestLocations']
assert sorted(failures)==sorted(previous)
validation={'checkpoint':'docs/evidence/direct-sympy-test-checkpoint','compilerIdentity':identity,'expressionRepresentation':'SymPy mathematical strings','integrationTests':34,'passed':34,'failed':0,'skipped':0,'wholeProjectSuite':{'current':counts,'sameFailingTestLocations':failures,'newFailingTestLocations':0},'nativeRMSPairs':462422016,'nativeRMSSmallBranchPairs':44294796,'nativeProjectionPairs':462422016,'nativeProjectionSelectedPairs':22147398,'nativeCellViolations':0,'exhaustiveArtifactCases':e['cases'],'exhaustiveArtifactMismatches':e['mismatches'],'artifact':artifact,'artifactPaths':5,'broadDiagnostic':{'paths':d['stats']['completedPaths'],'serializedCharacters':d['serializedCharacters'],'guardOccurrences':sum(len(row['guards']) for row in d['arms']),'uniqueGuardStrings':len({g['sha256'] for row in d['arms'] for g in row['guards']}),'stop':d['stop']},'coveredInputPatterns':covered,'unfinishedInputPatterns':pending,'historicalNumericalResultsReused':False,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');raw=path.read_text();previous_map=json.loads(raw);key='branchProjectionValidation'
if key in previous_map:assert previous_map[key]==validation
else:
 assert raw.rstrip().endswith('}')
 new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(validation,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:v for k,v in json.loads(new).items() if k!=key}==previous_map
 path.write_text(new)
print(json.dumps({'integration':34,'wholeProject':counts,'freshCoveredPairs':covered,'historicalResultsReused':False,'fullCoordinateParity':False}))

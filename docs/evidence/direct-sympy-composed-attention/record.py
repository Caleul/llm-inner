"""Audit fresh compilation, numerical parity and preserved coverage."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups
root=Path(__file__).resolve().parent
logs=Path('artifacts/direct-sympy-input-partitions/composed-rms-diagnosis')
directory=Path('artifacts/direct-sympy-input-partitions/composed-rms-state')
previous=root.parent/'direct-sympy-fixed-grid'
old=json.loads((previous/'frontier.json').read_text())
state=json.loads((directory/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert [n for n,h in old['identity']['sources'].items() if state['identity']['sources'][n]!=h]==['direct_sympy_layer_bounds.py']
count=sum(n['status']=='complete' for n in old['tree'].values())
assert state['geometryReplan']['completeLeavesReset']==count and not state['geometryReplan']['numericalResultsReused']
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert all(promote_tree(state['tree'],state['root'],n['domains'],{})[2]==0 for n in old['tree'].values() if n['status']=='complete')
assert not state['finalArtifactEmitted'] and not state['finalParity']
promotions=state['coveragePromotions']
assert all(p['compilerIdentity']==state['identity'] and not p['parity']['mismatches'] for p in promotions)
assert sum(p['addedInputPatterns'] for p in promotions)==state['coveredInputPatterns']
texts,_,stats=groups(directory,state['tree'])
references={}
for folder in (previous,root.parent/'direct-sympy-silu-followup',root.parent/'direct-sympy-silu-update-bounds',root.parent/'direct-sympy-tight-update-bounds',root.parent/'direct-sympy-final-publication'):
 for row in json.loads((folder/'artifact-map.json').read_text()):references[row['sha256']]=(folder/row['file']).resolve()
rows=[]
for key,node in state['tree'].items():
 if node['status']!='complete':continue
 sha=node['artifact']['sha256'];body=(directory/node['artifact']['file']).read_bytes()
 assert hashlib.sha256(body).hexdigest()==sha
 path=references.get(sha)
 if path is None:
  path=root/(sha+'.expr');path.write_bytes(body);references[sha]=path
 assert path.read_bytes()==body
 rows.append({'region':key,'domains':node['domains'],'file':os.path.relpath(path,root),'sha256':sha,'characters':len(body)})
witness=json.loads((root/'witness-parity.json').read_text())
diagnosis=json.loads((root/'diagnosis.json').read_text())
assert diagnosis['newBoundsSHA256']==state['identity']['sources']['direct_sympy_layer_bounds.py']
assert witness['cases']==8196 and not witness['mismatches']
assert witness['sha256']==hashlib.sha256((root/'composed.expr').read_bytes()).hexdigest()
assert diagnosis['results'][1]['artifact']['sha256']==witness['sha256']
compatibility=json.loads((root/'state-compatibility.json').read_text())
assert compatibility['rejected'] and compatibility['manifestUnchanged'] and compatibility['compilerIdentity']==state['identity']
log=(logs/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(n)+r'\b',log) for name,n in (('tests',28),('pass',28),('fail',0),('skipped',0)))
benchmark=json.loads((logs/'parallel-comparison.json').read_text())
assert benchmark['coveredPatterns']==880803840 and benchmark['identicalDomainsAndArtifactHashes']
assert all(not benchmark[k]['nativeMismatches'] for k in ('sequential','parallel'))
validation={'build':True,'tests':28,'pass':28,'fail':0,'skipped':0,
 'composedProjectionNativeCases':7110656,'correlatedRMSLinearNativeCases':5332992,
 'orderedProjectionNativeCases':3555328,'checkpointUpdateNativeCases':888832,
 'siluMagnitudeNativeCases':22530,'boundViolations':0,'originalNumericalOrderPreserved':True,
 'oldNumericalResultsReused':False,'oldCompleteDomainsRecompiled':count,
 'coveredInputPatterns':state['coveredInputPatterns'],'unfinishedInputPatterns':state['unfinishedInputPatterns'],
 'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
 'addedInputPatterns':state['coveredInputPatterns']-old['coveredInputPatterns'],
 'freshBroadRegionCompilations':len(promotions),'nativeCases':sum(p['parity']['cases'] for p in promotions),'nativeMismatches':0,
 'completeRegions':len(rows),'distinctExpressions':len(texts),'sharedBodyCharacters':stats['sharedBodyCharacters'],
 'matchedModelPreviousProducers':diagnosis['results'][0]['producers'],
 'matchedModelCurrentProducers':diagnosis['results'][1]['producers'],
 'matchedModelPreviousLogicalCharacters':diagnosis['results'][0]['logicalCharacters'],
 'matchedModelCurrentLogicalCharacters':diagnosis['results'][1]['logicalCharacters'],
 'matchedModelArtifactCharacters':witness['characters'],'matchedModelNativeCases':witness['cases'],'matchedModelNativeMismatches':0,
 'parallelComparison':benchmark,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('test.log','diagnosis.log','recompilation.log','parallel-comparison.json','benchmark.log'):
 (root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
for name,data in (('validation.json',validation),('artifact-map.json',rows),('coalescing-comparison.json',{'previous':{'coveredInputPatterns':old['coveredInputPatterns']},'current':stats})):
 (root/name).write_text(json.dumps(data,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(state,separators=(',',':'))+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text();key='composedAttentionBoundValidation'
entry={'report':'docs/evidence/direct-sympy-composed-attention/validation.json',**validation}
if key not in json.loads(original):
 formatted=json.dumps(entry,indent=2)
 updated=original.rstrip()[:-1].rstrip()+',\n  "'+key+'": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
 json.loads(updated);path.write_text(updated)
else:assert json.loads(original)[key]==entry
print(json.dumps(validation,indent=2))

"""Audit fresh compilation, numerical parity and preserved coverage."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups
root=Path(__file__).resolve().parent
logs=Path('artifacts/direct-sympy-input-partitions/early-projection-diagnosis')
directory=Path('artifacts/direct-sympy-input-partitions/early-projection-state')
previous=root.parent/'direct-sympy-neighbor-cells'
old=json.loads((previous/'frontier.json').read_text())
state=json.loads((directory/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert [n for n,h in old['identity']['sources'].items() if state['identity']['sources'][n]!=h]==['direct_sympy_checkpoint.py','direct_sympy_layer_bounds.py']
count=sum(n['status']=='complete' for n in old['tree'].values())
assert state['geometryReplan']['completeLeavesReset']==count and not state['geometryReplan']['numericalResultsReused']
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert all(promote_tree(state['tree'],state['root'],n['domains'],{})[2]==0 for n in old['tree'].values() if n['status']=='complete')
assert not state['finalArtifactEmitted'] and not state['finalParity']
promotions=state['coveragePromotions']
assert all(p['compilerIdentity']==state['identity'] and not p['parity']['mismatches'] for p in promotions)
assert sum(p['addedInputPatterns'] for p in promotions)==state['coveredInputPatterns']
texts,_,stats=groups(directory,state['tree'])
old_texts,_,old_stats=groups(Path('artifacts/direct-sympy-input-partitions/neighbor-cell-state'),old['tree'])
references={}
for folder in (previous,root.parent/'direct-sympy-composed-attention',root.parent/'direct-sympy-silu-followup',root.parent/'direct-sympy-silu-update-bounds',root.parent/'direct-sympy-tight-update-bounds',root.parent/'direct-sympy-final-publication'):
 for row in json.loads((folder/'artifact-map.json').read_text()):references[row['sha256']]=(folder/row['file']).resolve()
for name in ('early.expr','previous.expr'):
 path=root/name;references[hashlib.sha256(path.read_bytes()).hexdigest()]=path
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
assert diagnosis['newCheckpointSHA256']==state['identity']['sources']['direct_sympy_checkpoint.py']
assert diagnosis['newBoundsSHA256']==state['identity']['sources']['direct_sympy_layer_bounds.py']
assert all(p['cases']==8200 and not p['mismatches'] for p in witness)
assert all(p['sha256']==hashlib.sha256((root/name).read_bytes()).hexdigest() for p,name in zip(witness,('previous.expr','early.expr')))
assert all(row['artifact']['sha256']==p['sha256'] for row,p in zip(diagnosis['results'],witness))
compatibility=json.loads((root/'state-compatibility.json').read_text())
assert compatibility['rejected'] and compatibility['manifestUnchanged'] and compatibility['compilerIdentity']==state['identity']
log=(logs/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(n)+r'\b',log) for name,n in (('tests',29),('pass',29),('fail',0),('skipped',0)))
benchmark=json.loads((logs/'parallel-comparison.json').read_text())
assert benchmark['coveredPatterns']==16842752 and benchmark['identicalDomainsAndArtifactHashes']
assert all(not p['mismatches'] for k in ('sequential','parallel') for p in benchmark[k]['parity'])
validation={'build':True,'tests':29,'pass':29,'fail':0,'skipped':0,
 'earlyConstantProjectionHalfPairs':8421376,'earlyConstantProjectionMismatches':0,
 'halfNeighborCellNativeCases':507888,'allPositiveHalfCellsChecked':31743,'nonHalfEndpointsChecked':31742,
 'composedProjectionNativeCases':7110656,'correlatedRMSLinearNativeCases':5332992,
 'orderedProjectionNativeCases':3555328,'checkpointUpdateNativeCases':888832,
 'siluMagnitudeNativeCases':22530,'boundViolations':0,'originalNumericalOrderPreserved':True,
 'oldNumericalResultsReused':False,'oldCompleteDomainsRecompiled':count,
 'coveredInputPatterns':state['coveredInputPatterns'],'unfinishedInputPatterns':state['unfinishedInputPatterns'],
 'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
 'addedInputPatterns':state['coveredInputPatterns']-old['coveredInputPatterns'],
 'freshBroadRegionCompilations':len(promotions),'nativeCases':sum(p['parity']['cases'] for p in promotions),'nativeMismatches':0,
 'completeRegions':len(rows),'distinctExpressions':len(texts),'sharedBodyCharacters':stats['sharedBodyCharacters'],
 'previousDistinctExpressions':old_stats['distinctExpressions'],'previousSharedBodyCharacters':old_stats['sharedBodyCharacters'],
 'previousBodiesRetained':len(old_texts.keys()&texts.keys()),'newBodyCharacters':sum(len(texts[h]) for h in texts.keys()-old_texts.keys()),
 'removedBodyCharacters':sum(len(old_texts[h]) for h in old_texts.keys()-texts.keys()),
 'matchedModelPreviousProducers':diagnosis['results'][0]['producers'],
 'matchedModelConstantProjections':diagnosis['results'][1]['constantProjections'],
 'matchedModelCurrentProducers':diagnosis['results'][1]['producers'],
 'matchedModelPreviousLogicalCharacters':diagnosis['results'][0]['logicalCharacters'],
 'matchedModelCurrentLogicalCharacters':diagnosis['results'][1]['logicalCharacters'],
 'matchedModelArtifactCharacters':witness[1]['characters'],'matchedModelPreviousArtifactCharacters':witness[0]['characters'],'matchedModelNativeCases':sum(p['cases'] for p in witness),'matchedModelNativeMismatches':0,
 'parallelComparison':benchmark,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('build.log','projection.log','test.log','diagnosis.log','recompilation.log','parallel-comparison.json','benchmark.log','central-positive.json','central-positive-8m.json'):
 (root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
for name,data in (('validation.json',validation),('artifact-map.json',rows),('coalescing-comparison.json',{'previous':old_stats,'current':stats})):
 (root/name).write_text(json.dumps(data,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(state,separators=(',',':'))+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text();key='earlyProjectionValidation'
entry={'report':'docs/evidence/direct-sympy-early-projection/validation.json',**validation}
if key not in json.loads(original):
 formatted=json.dumps(entry,indent=2)
 updated=original.rstrip()[:-1].rstrip()+',\n  "'+key+'": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
 json.loads(updated);path.write_text(updated)
else:
 assert list(json.loads(original))[-1]==key
 formatted=json.dumps(entry,indent=2)
 prefix=original[:original.index('  \"'+key+'\": ')]
 updated=prefix+'  \"'+key+'\": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
 assert {k:v for k,v in json.loads(updated).items() if k!=key}=={k:v for k,v in json.loads(original).items() if k!=key}
 path.write_text(updated)
print(json.dumps(validation,indent=2))

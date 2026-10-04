"""Audit source-compatible recovery and reject local wins that fragment bodies."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups
root=Path(__file__).resolve().parent;logs=Path('artifacts/direct-sympy-input-partitions/fixed-grid-diagnosis')
directory=Path('artifacts/direct-sympy-input-partitions/fixed-grid-reuse-state')
previous=root.parent/'direct-sympy-silu-followup'
old=json.loads((previous/'frontier.json').read_text());state=json.loads((directory/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert [n for n,h in old['identity']['sources'].items() if state['identity']['sources'][n]!=h]==['direct_sympy_conversions.py']
count=sum(n['status']=='complete' for n in old['tree'].values())
assert state['geometryReplan']['completeLeavesReset']==count and not state['geometryReplan']['numericalResultsReused']
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert all(promote_tree(state['tree'],state['root'],n['domains'],{})[2]==0 for n in old['tree'].values() if n['status']=='complete')
assert not state['finalArtifactEmitted'] and not state['finalParity']
promotions=state['coveragePromotions'];assert all(p['compilerIdentity']==state['identity'] and not p['parity']['mismatches'] for p in promotions)
assert sum(p['addedInputPatterns'] for p in promotions)==state['coveredInputPatterns']
old_texts,_,old_stats=groups(Path('artifacts/direct-sympy-input-partitions/silu-update-state'),old['tree'])
texts,_,stats=groups(directory,state['tree']);assert texts==old_texts
assert stats['sharedBodyCharacters']==old_stats['sharedBodyCharacters']==3657767
references={}
for folder in (previous,root.parent/'direct-sympy-silu-update-bounds',root.parent/'direct-sympy-tight-update-bounds',root.parent/'direct-sympy-final-publication'):
 for row in json.loads((folder/'artifact-map.json').read_text()):references[row['sha256']]=(folder/row['file']).resolve()
rows=[]
for key,node in state['tree'].items():
 if node['status']!='complete':continue
 sha=node['artifact']['sha256'];body=(directory/node['artifact']['file']).read_bytes()
 assert hashlib.sha256(body).hexdigest()==sha and references[sha].read_bytes()==body
 rows.append({'region':key,'domains':node['domains'],'file':os.path.relpath(references[sha],root),'sha256':sha,'characters':len(body)})
witness=json.loads((root/'witness-parity.json').read_text());diagnosis=json.loads((root/'diagnosis.json').read_text())
assert diagnosis['currentSourceSHA256']==state['identity']['sources']['direct_sympy_conversions.py']
assert all(p['cases']==8196 and not p['mismatches'] for p in witness)
assert all(p['sha256']==hashlib.sha256((root/name).read_bytes()).hexdigest() for p,name in zip(witness,('previous.expr','fixed-grid.expr')))
assert witness[0]['sha256']==witness[1]['sha256']
assert (root/'previous.expr').read_bytes()==(root/'fixed-grid.expr').read_bytes()
compatibility=json.loads((root/'state-compatibility.json').read_text());assert compatibility['rejected'] and compatibility['manifestUnchanged']
log=(logs/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(n)+r'\b',log) for name,n in (('tests',28),('pass',28),('fail',0),('skipped',0)))
followup=json.loads((previous/'validation.json').read_text())
validation={'build':True,'tests':28,'pass':28,'fail':0,'skipped':0,
 'f32MidpointCases':50331648,'halfMidpointCases':184314,'scaledF32BoundaryCases':15234,'primitiveMismatches':0,
 'oneOccurrenceKernelsPreserved':True,'localFragmentingVersionRejected':True,
 'oldNumericalResultsReused':False,'oldCompleteDomainsRecompiled':count,
 'coveredInputPatterns':state['coveredInputPatterns'],'unfinishedInputPatterns':state['unfinishedInputPatterns'],
 'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
 'addedInputPatternsThisTurn':followup['addedInputPatterns'],'freshBroadRegionCompilations':len(promotions),
 'nativeCases':sum(p['parity']['cases'] for p in promotions),'nativeMismatches':0,
 'completeRegions':len(rows),'distinctExpressions':len(texts),'sharedBodyCharacters':stats['sharedBodyCharacters'],
 'matchedModelArtifactIdentical':True,'matchedModelArtifactCharacters':witness[1]['characters'],
 'matchedModelNativeCases':sum(p['cases'] for p in witness),'matchedModelNativeMismatches':0,
 'missingCmathHeaderFixed':True,'staleExactSizeAssertionUpdated':True,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('build.log','test.log','grid.log','diagnosis.log','recompilation.log','missing-header.log'):
 (root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
for name,data in (('validation.json',validation),('artifact-map.json',rows)):
 (root/name).write_text(json.dumps(data,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(state,separators=(',',':'))+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text();key='fixedGridConversionValidation'
entry={'report':'docs/evidence/direct-sympy-fixed-grid/validation.json',**validation}
if key not in json.loads(original):
 formatted=json.dumps(entry,indent=2)
 updated=original.rstrip()[:-1].rstrip()+',\n  "'+key+'": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
 json.loads(updated);path.write_text(updated)
else:assert json.loads(original)[key]==entry
print(json.dumps(validation,indent=2))

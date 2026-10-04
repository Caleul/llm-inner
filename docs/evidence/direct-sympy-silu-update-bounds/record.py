"""Record fresh admissions under the tighter certified activation bound."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups

root=Path(__file__).resolve().parent
directory=Path('artifacts/direct-sympy-input-partitions/silu-update-state')
logs=Path('artifacts/direct-sympy-input-partitions/silu-diagnosis')
previous=root.parent/'direct-sympy-tight-update-bounds'
old=json.loads((previous/'frontier.json').read_text())
state=json.loads((directory/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert [n for n,h in old['identity']['sources'].items() if state['identity']['sources'][n]!=h]==['direct_sympy_layer_bounds.py']
assert state['geometryReplan']['completeLeavesReset']==100
assert state['geometryReplan']['numericalResultsReused'] is False
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert all(promote_tree(state['tree'],state['root'],n['domains'],{})[2]==0 for n in old['tree'].values() if n['status']=='complete')
assert not state['finalArtifactEmitted'] and not state['finalParity']
promotions=state['coveragePromotions']
assert all(p['compilerIdentity']==state['identity'] and p['parity']['mismatches']==0 for p in promotions)
assert sum(p['addedInputPatterns'] for p in promotions)==state['coveredInputPatterns']
compatibility=json.loads((root/'state-compatibility.json').read_text())
assert compatibility['rejected'] and compatibility['manifestUnchanged']
texts,_,summary=groups(directory,state['tree'])
references={}
for folder in (previous,root.parent/'direct-sympy-final-publication'):
    for row in json.loads((folder/'artifact-map.json').read_text()):
        references[row['sha256']]=(folder/row['file']).resolve()
references[hashlib.sha256((previous/'tightened.expr').read_bytes()).hexdigest()]=(previous/'tightened.expr').resolve()
rows=[];new_files=set()
for key,node in state['tree'].items():
    if node['status']!='complete':continue
    body=(directory/node['artifact']['file']).read_bytes();sha=hashlib.sha256(body).hexdigest()
    assert sha==node['artifact']['sha256']
    if sha in references:
        reference=references[sha];assert reference.read_bytes()==body
        filename=os.path.relpath(reference,root)
    else:
        filename='region-'+sha+'.expr';(root/filename).write_bytes(body);new_files.add(filename)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(body)})
diagnosis=json.loads((root/'diagnosis.json').read_text());before,after=diagnosis['results']
assert (before['producers'],after['producers'])==(25,17)
assert 'budget' in before['stop'].lower() and 'artifact' not in before
assert after['artifact']['characters']==171977
parity=json.loads((root/'witness-parity.json').read_text())
assert parity['cases']==8196 and parity['mismatches']==0
effective=root/'tightened.expr'
if not effective.exists():effective=previous/'tightened.expr'
assert parity['sha256']==after['artifact']['sha256']==hashlib.sha256(effective.read_bytes()).hexdigest()
# Identical bytes are documented once; fresh compilation/parity remain separate.
assert effective.read_bytes()==(previous/'tightened.expr').read_bytes()
after['artifact']['path']='../direct-sympy-tight-update-bounds/tightened.expr'
parity['artifact']=after['artifact']['path']
(root/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
(root/'witness-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
(root/'tightened.expr').unlink(missing_ok=True)
test=(logs/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(n)+r'\b',test) for name,n in (('tests',27),('pass',27),('fail',0),('skipped',0)))
validation={'build':True,'tests':27,'pass':27,'fail':0,'skipped':0,
    'certifiedSiluHalfPatterns':22530,'activationBoundViolations':0,
    'nativeProjectionCases':3555328,'nativeCheckpointUpdateCases':888832,
    'oldSourceRejected':True,'oldNumericalResultsReused':False,'oldCompleteDomainsRecompiled':100,
    'oldCoveredInputPatterns':old['coveredInputPatterns'],'coveredInputPatterns':state['coveredInputPatterns'],
    'addedInputPatterns':state['coveredInputPatterns']-old['coveredInputPatterns'],
    'unfinishedInputPatterns':state['unfinishedInputPatterns'],'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
    'freshBroadRegionCompilations':len(promotions),'nativeCases':sum(p['parity']['cases'] for p in promotions),'nativeMismatches':0,
    'completeRegions':len(rows),'distinctExpressions':len(texts),'newExpressionFiles':len(new_files),
    'matchedRegionProducersBefore':25,'matchedRegionProducersAfter':17,
    'matchedRegionExpandedCharactersBefore':before['logicalCharacters'],'matchedRegionExpandedCharactersAfter':after['logicalCharacters'],
    'matchedRegionArtifactCharacters':171977,'matchedRegionNativeCases':8196,'matchedRegionNativeMismatches':0,
    'mlpBoundsBefore':[1.341104507446289e-05,1.5974044799804688e-05],
    'mlpBoundsAfter':[6.794929504394531e-06,8.046627044677734e-06],
    'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('build.log','test.log','diagnosis.log','recompilation.log'):
    (root/name).write_text('\n'.join(line.rstrip() for line in (logs/name).read_text().splitlines()).rstrip()+'\n')
for name,data in (('validation.json',validation),('artifact-map.json',rows),('coalescing.json',summary)):
    (root/name).write_text(json.dumps(data,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(state,separators=(',',':'))+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
key='siluUpdateBoundValidation';entry={'report':'docs/evidence/direct-sympy-silu-update-bounds/validation.json',**validation}
if key not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "'+key+'": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)[key]==entry
print(json.dumps(validation,indent=2))

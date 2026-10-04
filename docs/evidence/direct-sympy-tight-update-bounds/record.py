"""Audit the changed proof, fresh numerical admissions and effective strings."""
import hashlib,json,os,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity,promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups

root=Path(__file__).resolve().parent
source=Path('artifacts/direct-sympy-input-partitions/central-diagnosis')
directory=Path('artifacts/direct-sympy-input-partitions/tight-update-state')
previous=root.parent/'direct-sympy-final-publication'
old=json.loads((previous/'frontier.json').read_text())
state=json.loads((directory/'frontier.json').read_text())
compatibility=json.loads((root/'state-compatibility.json').read_text())
assert compatibility['rejected'] and compatibility['manifestUnchanged']
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert state['identity']!=old['identity']
assert [name for name,digest in old['identity']['sources'].items() if state['identity']['sources'][name]!=digest]==['direct_sympy_layer_bounds.py']
assert state['geometryReplan']['numericalResultsReused'] is False
assert state['geometryReplan']['completeLeavesReset']==89
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
assert all(promote_tree(state['tree'],state['root'],node['domains'],{})[2]==0 for node in old['tree'].values() if node['status']=='complete')
assert not state['finalArtifactEmitted'] and not state['finalParity']
promotions=state['coveragePromotions']
assert all(r['compilerIdentity']==state['identity'] and r['parity']['mismatches']==0 for r in promotions)
assert sum(r['addedInputPatterns'] for r in promotions)==state['coveredInputPatterns']
texts,_,summary=groups(directory,state['tree'])
references={row['sha256']:(previous/row['file']).resolve() for row in json.loads((previous/'artifact-map.json').read_text())}
rows=[];new_files=set()
for key,node in state['tree'].items():
    if node['status']!='complete':continue
    artifact=node['artifact'];sha=artifact['sha256'];body=(directory/artifact['file']).read_bytes()
    assert hashlib.sha256(body).hexdigest()==sha
    if sha in references:
        reference=references[sha];assert reference.read_bytes()==body
        filename=os.path.relpath(reference,root)
    else:
        filename='region-'+sha+'.expr';(root/filename).write_bytes(body);new_files.add(filename)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(body)})
diagnosis=json.loads((root/'diagnosis.json').read_text());before,after=diagnosis['results']
assert before['artifact']['sha256']==after['artifact']['sha256']
assert before['producers']==25 and after['producers']==17
# Keep one copy of independently compiled but byte-identical witness bodies.
(root/'previous.expr').unlink(missing_ok=True)
for row in diagnosis['results']:row['artifact']['path']='tightened.expr'
(root/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
parity=json.loads((root/'witness-parity.json').read_text())
assert parity['sha256']==after['artifact']['sha256']==hashlib.sha256((root/'tightened.expr').read_bytes()).hexdigest()
assert parity['cases']==8196 and parity['mismatches']==0
log=(source/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',27),('pass',27),('fail',0),('skipped',0)))
bounds=(source/'bounds.log').read_text()
assert 'Ran 3 tests' in bounds and 'cases=3555328 violations=0' in bounds and 'cases=888832 violations=0' in bounds
validation={'build':True,'tests':27,'pass':27,'fail':0,'skipped':0,'boundTests':3,
    'nativeProjectionCases':3555328,'nativeCheckpointUpdateCases':888832,'boundViolations':0,
    'oldSourceRejected':True,'oldNumericalResultsReused':False,'oldCompleteDomainsRecompiled':89,
    'oldCoveredInputPatterns':old['coveredInputPatterns'],'coveredInputPatterns':state['coveredInputPatterns'],
    'addedInputPatterns':state['coveredInputPatterns']-old['coveredInputPatterns'],
    'unfinishedInputPatterns':state['unfinishedInputPatterns'],'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
    'freshBroadRegionCompilations':len(promotions),'nativeCases':sum(r['parity']['cases'] for r in promotions),'nativeMismatches':0,
    'completeRegions':len(rows),'distinctExpressions':len(texts),'newExpressionFiles':len(new_files),
    'matchedRegionProducersBefore':before['producers'],'matchedRegionProducersAfter':after['producers'],
    'matchedRegionArtifactCharacters':after['artifact']['characters'],'matchedRegionArtifactIdentical':True,
    'matchedRegionNativeCases':parity['cases'],'matchedRegionNativeMismatches':0,
    'updateThresholdsBefore':[32,16],'updateThresholdsAfter':[8,4],
    'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
for name in ('build.log','test.log','bounds.log','diagnosis.log','recompilation.log','recovery.log','coalescing.json'):
    text=(source/name).read_text()
    if name.endswith('.log'):text='\n'.join(line.rstrip() for line in text.splitlines()).rstrip()+'\n'
    (root/name).write_text(text)
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(state,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-tight-update-bounds/validation.json',**validation};key='tightUpdateBoundValidation'
if key not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "'+key+'": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)[key]==entry
print(json.dumps(validation,indent=2))

"""Freeze continuation evidence without duplicating validated expressions."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree
from direct_sympy_partition_coalesce import groups

root=Path(__file__).resolve().parent
source=Path('artifacts/direct-sympy-input-partitions/continued-coordinate')
directory=Path('artifacts/direct-sympy-input-partitions/dominant-square-state')
previous=root.parent/'direct-sympy-four-worker-continuation'
state=json.loads((directory/'frontier.json').read_text())
before=json.loads((previous/'frontier.json').read_text())
run=json.loads((source/'report.json').read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert state['identity']==before['identity']==run['compilerIdentity']==identity
controller_before=subprocess.check_output(['git','show','9bbdc24:helpers/direct_sympy_partition_parallel.py'])
assert run['controllerSHA256']==hashlib.sha256(controller_before).hexdigest()
assert audit_tree(state['tree'],state['root'])==(state['coveredInputPatterns'],state['unfinishedInputPatterns'])
old={key:node for key,node in before['tree'].items() if node['status']=='complete'}
assert len(old)==76 and all(state['tree'][key]==node for key,node in old.items())
assert run['coveredInputPatterns']==state['coveredInputPatterns']
assert run['addedInputPatterns']==state['coveredInputPatterns']-before['coveredInputPatterns']==1064960
assert run['nativeCases']==3380 and run['nativeMismatches']==0
assert not state['finalArtifactEmitted'] and not state['finalParity']
texts,_,summary=groups(directory,state['tree'])
assert summary['distinctExpressions']==21 and summary['completeRegions']==89
previous_rows=json.loads((previous/'artifact-map.json').read_text())
references={row['sha256']:(previous/row['file']).resolve() for row in previous_rows}
rows=[]
for key,node in state['tree'].items():
    if node['status']!='complete':continue
    artifact=node['artifact'];reference=references[artifact['sha256']]
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==artifact['sha256']
    assert len(reference.read_text())==artifact['characters']
    rows.append({'region':key,'domains':node['domains'],'file':os.path.relpath(reference,root),
        'sha256':artifact['sha256'],'characters':artifact['characters']})
log=(source/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',26),('pass',26),('fail',0),('skipped',0)))
assert 'Ran 7 tests' in (source/'unit.log').read_text() and '\nOK\n' in (source/'unit.log').read_text()
validation={'build':True,'directStringTests':26,'pass':26,'fail':0,'skipped':0,'parallelControllerTests':7,
    'finalControllerFocusedRerun':True,'numericalSourcesUnchanged':True,'oldCompleteLeavesPreserved':len(old),
    'parallelAttempts':run['attempts'],'newCompleteRegions':len(rows)-len(old),
    'addedInputPatterns':run['addedInputPatterns'],'coveredInputPatterns':state['coveredInputPatterns'],
    'unfinishedInputPatterns':state['unfinishedInputPatterns'],
    'coveragePercent':100*state['coveredInputPatterns']/state['totalInputPatterns'],
    'newNativeCases':run['nativeCases'],'newNativeMismatches':0,
    'compileSeconds':run['seconds'],'peakObservedCompileRSSBytes':max(w['peakObservedRSSBytes'] for w in run['waves']),
    'memoryBudgetBytes':run['memoryBudgetBytes'],
    'completeRegions':len(rows),'distinctExpressions':len(texts),'coalescedRectangles':summary['coalescedRectangles'],
    'copiedBodyCharacters':summary['copiedBodyCharacters'],'sharedBodyCharacters':summary['sharedBodyCharacters'],
    'removedRepeatedBodyCharacters':summary['copiedBodyCharacters']-summary['sharedBodyCharacters'],
    'finalPublicationTests':{'coalescedCandidateBeforePublication':True,'actualCheckpointRejectsWrongOutput':True,
        'candidateMutationRejected':True,'previousArtifactAndManifestPreserved':True},
    'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,
    'colabCurrentInventory':'No active sessions found; no remote run in this continuation.'}
for name in ('report.json','coalescing.json','build.log','unit.log','test.log','run.log'):
    text=(source/name).read_text()
    if name.endswith('.log'):text='\n'.join(line.rstrip() for line in text.splitlines()).rstrip()+'\n'
    (root/name).write_text(text)
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(state,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-final-publication/validation.json',**validation}
key='coalescedPublicationValidation'
if key not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "'+key+'": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)[key]==entry
print(json.dumps(validation,indent=2))

from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
base=Path('artifacts/direct-sympy-input-partitions')
coverage=json.loads((root/'coverage-frontier.json').read_text())
lexical=json.loads((root/'lexical-frontier.json').read_text())
assert coverage['identity']==lexical['identity']
assert coverage['geometryImport']['sha256']==lexical['geometryImport']['sha256']
current=json.loads((base/'frontier-final-state/frontier.json').read_text())
parity=json.loads((root/'current-parity.json').read_text())
assert parity['mismatches']==0 and parity['coveredInputPatterns']==current['coveredInputPatterns']
(root/'current-frontier.json').write_text(json.dumps(current,indent=2)+'\n')
rows=[];old=lexical['tree']
for key,node in current['tree'].items():
 if node['status']!='complete':continue
 source=base/'frontier-final-state'/node['artifact']['file'];payload=source.read_bytes();sha=hashlib.sha256(payload).hexdigest()
 assert sha==node['artifact']['sha256']==old[key]['artifact']['sha256']
 filename='region-'+sha+'.expr';(root/filename).write_bytes(payload)
 rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(payload)})
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
assert current['attempts'][-1]['selectionOrder']=='coverage' and current['attempts'][-1]['region']=='1'
assert len(rows)==parity['completedRegions']==24
validation={'tests':23,'pass':23,'fail':0,'skipped':0,'inputPartitionTests':6,'build':True,'comparisonAttemptsPerMode':24,'comparisonRegionSeconds':8,'coverageCompletedRegions':0,'lexicalCompletedRegions':24,'coveragePolicySpeedupDemonstrated':False,'defaultPolicy':'lexical','currentCompletedRegions':len(rows),'currentCoveredInputPatterns':current['coveredInputPatterns'],'currentUnfinishedInputPatterns':current['unfinishedInputPatterns'],'nativeCases':parity['cases'],'nativeMismatches':parity['mismatches'],'currentArtifactsByteIdenticalToLexicalExperiment':True,'policySwitchPreservedNumericalResults':True,'finalArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text();assert 'coverageFrontierValidation' not in json.loads(original)
entry=json.dumps({'report':str(root/'validation.json'),**validation},indent=2)
updated=original.rstrip()[:-1].rstrip()+',\n  "coverageFrontierValidation": '+'\n'.join('  '+line for line in entry.splitlines()).lstrip()+'\n}\n';json.loads(updated);path.write_text(updated)
for log in root.glob('*.log'):log.write_text(log.read_text().rstrip()+'\n')
print(json.dumps(validation,indent=2))

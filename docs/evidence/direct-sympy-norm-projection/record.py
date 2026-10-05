"""Record emitted expressions only after fresh compilation and parity admission."""
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity, promote_tree
from direct_sympy_partition_run import audit_tree
from direct_sympy_savepoints import digest_file

root = Path(__file__).resolve().parent
logs = Path('artifacts/direct-sympy-input-partitions/norm-projection-diagnosis')
directory = Path('artifacts/direct-sympy-input-partitions/norm-projection-state')
previous = root.parent / 'direct-sympy-directed-residual'
old = json.loads((previous / 'frontier.json').read_text())
state = json.loads((directory / 'frontier.json').read_text())
identity = live_identity('docs/evidence/direct-sympy-test-checkpoint', 2)
assert state['identity'] == identity
assert audit_tree(state['tree'], state['root']) == (state['coveredInputPatterns'], state['unfinishedInputPatterns'])
assert not state['geometryReplan']['numericalResultsReused']
assert not state['finalArtifactEmitted'] and not state['finalParity']
assert all(promote_tree(state['tree'], state['root'], n['domains'], {})[2] == 0
           for n in old['tree'].values() if n['status'] == 'complete')
assert all(p['compilerIdentity'] == identity and p['parity']['mismatches'] == 0
           for p in state['coveragePromotions'])
assert sum(p['addedInputPatterns'] for p in state['coveragePromotions']) == state['coveredInputPatterns']
references = {}
for row in json.loads((previous / 'artifact-map.json').read_text()):
    path = (previous / row['file']).resolve()
    assert digest_file(path) == row['sha256']
    references[row['sha256']] = path
emission = json.loads((logs / 'wide-emission.json').read_text())
assert emission['complete'] and emission['artifact']['complete'] and emission['artifact']['compilerAliases'] == 0
actual = logs / 'wide-mixed.expr'
assert digest_file(actual) == emission['artifact']['sha256']
assert actual.stat().st_size == emission['artifact']['characters']
same = json.loads((logs / 'wide-same-emission.json').read_text())
assert same['complete'] and same['artifact']['complete'] and same['artifact']['compilerAliases'] == 0
assert digest_file(logs / 'wide-same.expr') == same['artifact']['sha256']
if same['artifact']['sha256'] not in references:
    references[same['artifact']['sha256']] = (root / 'same.expr').resolve()
    references[same['artifact']['sha256']].write_bytes((logs / 'wide-same.expr').read_bytes())
assert references[same['artifact']['sha256']].read_bytes() == (logs / 'wide-same.expr').read_bytes()
rejected = json.loads((logs / 'broad-emission.json').read_text())
assert not rejected['complete'] and 'budget' in rejected['stop'].lower()
expression = actual.read_text()
assert set(re.findall(r'\bX\d+\b', expression)) == {'X1', 'X2'}
assert not re.search(r'\b(?:CompileValue\d*|CASBoundary\d*|R16|R32|sqrt|Silu16)\s*\(', expression)
assert references[emission['artifact']['sha256']].read_bytes() == actual.read_bytes()
rows = []
for key, node in state['tree'].items():
    if node['status'] != 'complete':
        continue
    sha = node['artifact']['sha256']
    path = directory / node['artifact']['file']
    assert digest_file(path) == sha
    if sha not in references:
        references[sha] = (root / (sha + '.expr')).resolve()
        references[sha].write_bytes(path.read_bytes())
    assert references[sha].read_bytes() == path.read_bytes()
    rows.append({'region': key, 'domains': node['domains'],
                 'file': os.path.relpath(references[sha], root),
                 'sha256': sha, 'characters': node['artifact']['characters']})
exhaustive = []
for name in ('exhaustive-wide-negative-positive', 'exhaustive-wide-positive-negative', 'exhaustive-wide-positive-positive', 'exhaustive-wide-negative-negative'):
    report = json.loads((logs / (name + '.json')).read_text())
    assert report['compilerIdentity'] == identity and report['complete']
    assert report['mismatches'] == 0 and report['cases'] == report['totalCases'] == 67125249
    expected_sha = same['artifact']['sha256'] if name.endswith(('positive-positive', 'negative-negative')) else emission['artifact']['sha256']
    assert report['sha256'] == expected_sha
    report['artifact'] = os.path.relpath(references[report['sha256']], root)
    (root / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    exhaustive.append(report)
proof = (logs / 'proof.log').read_text()
assert 'Correlated RMS projection bounds: cases=5332992 violations=0' in proof and 'Ran 10 tests' in proof and '\nOK\n' in proof
test = (logs / 'test.log').read_text()
assert all(s in test for s in ('tests 32', 'pass 32', 'fail 0', 'skipped 0'))
project = (logs / 'project-test.log').read_text()
failures = re.findall(r'^test at (.+)$', project, re.M)
assert len(failures) == 15 and failures == re.findall(r'^test at (.+)$', (previous / 'project-test.log').read_text(), re.M)
totals = {key: int(value) for key, value in re.findall(r'^ℹ (tests|pass|fail|skipped) (\d+)', project, re.M)}
assert totals == {'tests': 626, 'pass': 570, 'fail': 15, 'skipped': 41}
benchmark = json.loads((logs / 'parallel-comparison.json').read_text())
assert benchmark['compilerIdentity'] == identity and benchmark['identicalDomainsAndArtifactHashes']
assert benchmark['sequential']['artifactHashes'] == benchmark['parallel']['artifactHashes']
assert all(r['mismatches'] == 0 for mode in ('sequential', 'parallel') for r in benchmark[mode]['parity'])
(root / 'parallel-comparison.json').write_text(json.dumps(benchmark, indent=2) + '\n')
validation = {'checkpoint': 'docs/evidence/direct-sympy-test-checkpoint', 'compilerIdentity': identity,
    'expressionRepresentation': 'SymPy-compatible mathematical strings',
    'testFile': 'test/direct-sympy-strings.test.ts', 'pythonTests': 'helpers/direct_sympy_layer_bounds_test.py',
    'integrationTests': 32, 'passed': 32, 'failed': 0, 'skipped': 0,
    'wholeProjectSuite': {'current': totals, 'sameFailingTestLocations': failures, 'newFailingTestLocations': 0},
    'boundProofCases': 5332992, 'boundViolations': 0, 'pythonTestCount': 10,
    'exhaustiveArtifactCases': sum(r['cases'] for r in exhaustive), 'exhaustiveArtifactMismatches': 0,
    'artifactCharacters': emission['artifact']['characters'], 'artifactPaths': emission['artifact']['paths'],
    'artifactSHA256': emission['artifact']['sha256'], 'freshPromotions': len(state['coveragePromotions']),
    'sameSignArtifactCharacters': same['artifact']['characters'], 'sameSignArtifactSHA256': same['artifact']['sha256'],
    'rejectedBroadRegion': {'domains': rejected['inputDomains'], 'logicalCharacters': rejected['logicalCharacters'],
        'addedInputPatterns': 0, 'reason': rejected['stop']},
    'parallelComparison': {'speedRatio': benchmark['speedRatio'], 'identicalDomainsAndArtifactHashes': True,
        'sequentialSeconds': benchmark['sequential']['seconds'], 'parallelSeconds': benchmark['parallel']['seconds'],
        'coveredPatterns': benchmark['coveredPatterns'], 'fullCoordinateTiming': False,
        'sequentialPeakObservedRSSBytes': benchmark['sequential']['memory']['peakObservedRSSBytes'],
        'parallelPeakObservedRSSBytes': benchmark['parallel']['memory']['peakObservedRSSBytes']},
    'freshCorpusCases': sum(p['parity']['cases'] for p in state['coveragePromotions']), 'freshCorpusMismatches': 0,
    'previousCoveredInputPatterns': old['coveredInputPatterns'], 'coveredInputPatterns': state['coveredInputPatterns'],
    'unfinishedInputPatterns': state['unfinishedInputPatterns'],
    'addedInputPatterns': state['coveredInputPatterns'] - old['coveredInputPatterns'],
    'coveragePercent': 100 * state['coveredInputPatterns'] / state['totalInputPatterns'],
    'fullCoordinateArtifactEmitted': False, 'fullCoordinateParity': False, 'multipleTokenParity': False}
emission['artifact']['path'] = os.path.relpath(references[emission['artifact']['sha256']], root)
same['artifact']['path'] = os.path.relpath(references[same['artifact']['sha256']], root)
(root / 'same-emission.json').write_text(json.dumps(same, indent=2) + '\n')
(root / 'rejected-broad-emission.json').write_text(json.dumps(rejected, indent=2) + '\n')
for name, data in (('emission.json', emission), ('validation.json', validation), ('artifact-map.json', rows), ('frontier.json', state)):
    (root / name).write_text(json.dumps(data, indent=2) + '\n')
for name in ('build.log', 'test.log', 'project-test.log', 'proof.log', 'recompilation.log', 'wide-emission.log', 'wide-same-emission.log', 'broad-emission.log', 'exhaustive-wide-negative-positive.log', 'exhaustive-wide-positive-negative.log', 'exhaustive-wide-positive-positive.log', 'exhaustive-wide-negative-negative.log', 'benchmark.log'):
    (root / name).write_text('\n'.join(line.rstrip() for line in (logs / name).read_text().splitlines()).rstrip() + '\n')
mapfile = Path('docs/direct-string-validation.json')
raw = mapfile.read_text()
existing = json.loads(raw)
if 'normProjectionValidation' in existing:
    if existing['normProjectionValidation'] != validation:
        raise ValueError('Historical evidence differs; append a new entry instead')
else:
    assert raw.rstrip().endswith('}')
    mapfile.write_text(raw.rstrip()[:-1].rstrip() + ',\n  "normProjectionValidation": ' + json.dumps(validation, indent=2).replace('\n', '\n  ') + '\n}\n')
print(json.dumps(validation), flush=True)

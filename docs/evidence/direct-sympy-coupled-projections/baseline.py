"""Reproduce the full-suite failures from an isolated, unchanged HEAD archive."""
import json,subprocess,tempfile
from pathlib import Path

workspace=Path.cwd();logs=workspace/'artifacts/direct-sympy-input-partitions/central-guard-diagnosis'
head=subprocess.check_output(['git','rev-parse','abf5d72'],text=True).strip()
with tempfile.TemporaryDirectory(prefix='llm-inner-baseline-') as temporary:
    root=Path(temporary)
    archived=subprocess.Popen(['git','archive',head],stdout=subprocess.PIPE)
    unpacked=subprocess.run(['tar','-x','-C',str(root)],stdin=archived.stdout)
    archived.stdout.close()
    assert archived.wait()==0 and unpacked.returncode==0
    for name in ('node_modules','artifacts','.agent-loop'):
        if not (root/name).exists():(root/name).symlink_to(workspace/name,target_is_directory=True)
    build=subprocess.run(['npm','run','build'],cwd=root,capture_output=True,text=True)
    (logs/'baseline-build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0
    files=['dist/test/gemma4-real-calibration.test.js','dist/test/gemma4-runtime-reduction-trace-coverage.test.js',
        'dist/test/literal-scalar-substitution-example.test.js','scripts/agent-task-contract.test.mjs']
    run=subprocess.run(['node','--test',*files],cwd=root,capture_output=True,text=True)
    (logs/'baseline-test.log').write_text(run.stdout+run.stderr)
    # A successful baseline would contradict the pre-existing-failure hypothesis.
    assert run.returncode==1
    (logs/'baseline-identity.json').write_text(json.dumps({'head':head,'method':'Git archive, fresh build, same local artifacts and environment',
        'testFiles':files,'exitCode':run.returncode},indent=2)+'\n')

"""Rebuild the prior compiler for the exhaustive signed-zero region."""
import shutil,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='llm-inner-unit-baseline-')as directory:
 helpers=Path(directory)/'helpers';shutil.copytree('helpers',helpers)
 for name in ('direct_sympy_arithmetic.py','direct_sympy_conversions.py'):
  (helpers/name).write_bytes(subprocess.check_output(['git','show','79d1437:helpers/'+name]))
 with (root/'near-zero-baseline.log').open('w')as log:
  subprocess.run([sys.executable,str(root/'near_zero.py'),str(helpers),'near-zero-baseline'],stdout=log,stderr=log,check=True)

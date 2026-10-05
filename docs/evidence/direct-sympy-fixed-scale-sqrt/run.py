"""Compare the last committed root kernel with the current code; same budgets."""
import shutil,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='llm-inner-root-baseline-')as directory:
 helpers=Path(directory)/'helpers';shutil.copytree('helpers',helpers)
 before=subprocess.check_output(['git','show','048de2f:helpers/direct_sympy_sqrt.py'])
 (helpers/'direct_sympy_sqrt.py').write_bytes(before)
 for label,path in (('baseline',helpers),('current',Path('helpers').resolve())):
  with (root/(label+'.log')).open('w')as log:
   subprocess.run([sys.executable,str(root/'run_one.py'),str(path),label],stdout=log,stderr=log,check=True)

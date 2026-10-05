"""Compare the last committed guard compiler with the current code; same budgets."""
import shutil,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='llm-inner-guard-baseline-')as directory:
 helpers=Path(directory)/'helpers';shutil.copytree('helpers',helpers)
 before=subprocess.check_output(['git','show','d5dd566:helpers/direct_sympy_coherent_paths.py'])
 (helpers/'direct_sympy_coherent_paths.py').write_bytes(before)
 for label,path in (('baseline',helpers),('current',Path('helpers').resolve())):
  with (root/(label+'.log')).open('w')as log:
   subprocess.run([sys.executable,str(root/'run_one.py'),str(path),label],stdout=log,stderr=log,check=True)

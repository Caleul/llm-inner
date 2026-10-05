"""Reconstruct the last pushed streaming implementation for the recipe audit."""
import shutil,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='llm-inner-composition-baseline-')as directory:
 helpers=Path(directory)/'helpers';shutil.copytree('helpers',helpers)
 name='direct_sympy_streaming_literals.py'
 (helpers/name).write_bytes(subprocess.check_output(['git','show','9d227df:helpers/'+name]))
 for label,path in (('recipe-before',helpers),('recipe-after',Path('helpers').resolve())):
  with (root/(label+'.log')).open('w')as log:
   subprocess.run([sys.executable,str(root/'audit_recipe.py'),str(path),label],stdout=log,stderr=log,check=True)

"""Inspect original recipes and selected bodies without admitting a coordinate."""
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals,ALIASES
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_cover_regions import live_identity
root=Path(__file__).resolve().parent;label=sys.argv[2]
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=32*1024**2))as m:
 with streaming_literals(m)as r:
  expression=m.coordinate(2)
  recipes={name:r.definition_recipes[int(ALIASES.fullmatch(alias)[1])]
   for name,alias in r.names.items()if 'gated:' in name and ALIASES.fullmatch(alias)}
  report={'compilerIdentity':identity,'recipes':recipes}
  arm=next(CoherentPaths(r).arms(expression))
  report['firstArm']={'bodyCharacters':arm.view.size(arm.expression),
   'guardCharacters':sum(g.view.size(g.expression)for g in arm.guards),'decisions':len(arm.guards)}
assert identity==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
(root/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'recipes':recipes,'firstArm':report['firstArm']}))

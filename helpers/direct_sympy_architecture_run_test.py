"""Real runner publication paths and parity must use original final-row indices."""
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from direct_sympy_architecture import ArchitecturePlan,last_position_indices,prune_plan
from direct_sympy_architecture_run import unit
from direct_sympy_operators import OperatorBlock


class RunnerScopeTests(unittest.TestCase):
    def test_last_position_range_and_invalid_pruning(self):
        for length in range(1,9):
            plan=ArchitecturePlan((OperatorBlock(('X1',),tuple('X1' for _ in range(length*4))),),('output',),length,2,4,0,{})
            self.assertEqual(last_position_indices(plan),tuple(range((length-1)*4,length*4)))
            self.assertEqual(len(prune_plan(plan,last_position_indices(plan))[-1].outputs),4)
            with self.assertRaises(ValueError):prune_plan(plan,[-1])
            with self.assertRaises(ValueError):prune_plan(plan,[length*4])
            with self.assertRaises(ValueError):prune_plan(plan,[])
            with self.assertRaises(ValueError):prune_plan(plan,[1,0])
            with self.assertRaises(ValueError):prune_plan(plan,[0,0])
        constant=ArchitecturePlan((OperatorBlock(('X1',),('X1',)),OperatorBlock(('X2',),('1.0',))),('input','constant'),1,1,1,0,{})
        pruned=prune_plan(constant,[0])
        self.assertEqual(pruned,(OperatorBlock((),('1.0',)),))

    def test_runner_selects_prunes_rereads_and_compares_only_final_logits(self):
        plan=ArchitecturePlan((OperatorBlock(('X1','X2','X3','X4'),tuple('X1' for _ in range(8))),),('output',),2,2,4,0,{})
        values=(-0.,1.,2.,3.)
        def compose(selected,compiler,budget,*,output_indices,on_wave):
            self.assertEqual(selected,plan)
            self.assertEqual(output_indices,(4,5,6,7))
            on_wave({'completedJobs':1})
            return tuple(repr(v) for v in values),{'completedJobs':1}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'reference.json'
            reference.write_text(json.dumps({'cases':[{'inputBits':[[0,0],[0,0]],'label':'scope',
                'logitF64Bits':[['deliberately-wrong']*4,['0x'+struct.pack('>d',v).hex() for v in values]]}]}))
            args=SimpleNamespace(output=str(root/'result'),unit_length=2,seconds_per_length=30,
                checkpoint='unused',max_characters=512*1024**2,workers=2,memory_mib=4096,
                reference=str(reference),lower=False)
            with patch('direct_sympy_architecture.architecture_plan',return_value=plan),patch('direct_sympy_architecture.compose_architecture',side_effect=compose):
                self.assertEqual(unit(args),0)
            result=json.loads((root/'result/result.json').read_text())
            self.assertEqual(result['verifiedLogits'],4)
            self.assertEqual(result['mismatches'],[])
            self.assertEqual([r['position'] for r in result['expressions']],[1]*4)
            self.assertTrue(all((root/'result'/f'position-1-logit-{i}.work.expr').is_file() for i in range(4)))
            self.assertFalse(list((root/'result').glob('position-0-*')))
            self.assertFalse(result['finalArtifactParity'])


if __name__=='__main__':unittest.main()

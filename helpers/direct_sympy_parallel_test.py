"""Ordered continuation composition, branch contexts and memory admission."""
import ast
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from direct_sympy_checkpoint import CheckpointStrings,f32
from direct_sympy_parallel import ParallelBudget,parallel_lanes,resident_bytes
from direct_sympy_strings import Domain,StringCompiler,syntax


def evaluate(expression,values):
    functions={'R32':f32,'Piecewise':lambda *arms:next(v for v,c in arms if c)}
    return eval(compile(ast.Expression(syntax(expression)),'<parallel-test>','eval'),{'__builtins__':{},**functions},values)


def sequential(terms,compiler,domains):
    lanes=[]
    for lane in range(4):
        expression='0.0'
        for term in terms[lane::4]:
            expression=compiler.substitute('R32(X999999998 + X999999999)','X999999998',expression,domains)
            expression=compiler.substitute(expression,'X999999999',term,domains)
        lanes.append(expression)
    return lanes


class ParallelTests(unittest.TestCase):
    def test_pairs_compose_original_rounding_order_with_odd_blocks(self):
        terms=['16777216.0','0.0','0.0','0.0','1.0','0.0','0.0','0.0','-16777216.0']
        compiler=StringCompiler()
        actual,stats=parallel_lanes(terms,compiler,{},ParallelBudget(2,4*1024**3,1))
        expected=sequential(terms,StringCompiler(),{})
        self.assertEqual([evaluate(e,{}) for e in actual],[evaluate(e,{}) for e in expected])
        self.assertEqual(evaluate(actual[0],{}),0.0)
        self.assertEqual(f32(16777216.0+f32(1.0-16777216.0)),1.0)
        self.assertEqual(stats['completedBlocks'],9)
        self.assertEqual(stats['completedPairMerges'],5)
        self.assertEqual(stats['seededLanes'],4)
        self.assertEqual(stats['maxWorkersAdmitted'],2)
        self.assertTrue(all(e[2:4]==('factor','simplify') for e in compiler.events))
        self.assertTrue(all('X' not in e and 'CAS' not in e for e in actual))

    def test_branches_keep_their_own_context_and_signed_zero(self):
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        branch='Piecewise((X1 * 0.5, X1 < 0.0), (X1 * 2.0, True))'
        terms=[branch,'-0.0','0.0','-0.0','X1','0.0','-0.0','0.0',branch]
        compiler=StringCompiler()
        actual,stats=parallel_lanes(terms,compiler,domains,ParallelBudget(2,4*1024**3,1))
        expected=sequential(terms,StringCompiler(),domains)
        for x in [-1.0,-2**-24,-0.0,0.0,2**-24,0.5,1.0]:
            for a,b in zip(actual,expected):
                self.assertEqual(struct.pack('d',evaluate(a,{'X1':x})),struct.pack('d',evaluate(b,{'X1':x})))
        self.assertTrue(all('X2' not in e and 'CAS' not in e for e in actual))
        self.assertTrue(all(e[2:4]==('factor','simplify') for e in compiler.events))
        self.assertGreater(stats['completedPairMerges'],0)

    def test_memory_budget_rejects_before_dispatch_and_validates_configuration(self):
        compiler=StringCompiler()
        with self.assertRaisesRegex(ValueError,'cannot admit one block'):
            parallel_lanes(['X1'],compiler,{'X1':Domain(F(-1),F(1),-24,False)},ParallelBudget(2,1,1))
        self.assertEqual(compiler.events,[])
        for args in [(0,100,1),(2,0,1),(2,100,0)]:
            with self.assertRaises(ValueError):ParallelBudget(*args)
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        one_worker_limit=2*resident_bytes()+512*1024**2
        _,stats=parallel_lanes(['X1','X1'],StringCompiler(),domains,ParallelBudget(8,one_worker_limit,1))
        self.assertEqual(stats['maxWorkersAdmitted'],1)
        import multiprocessing
        children={p.pid for p in multiprocessing.active_children()}
        with self.assertRaises(ValueError):
            parallel_lanes(['X1'],StringCompiler(max_characters=1),domains,ParallelBudget(2,4*1024**3,1))
        self.assertEqual({p.pid for p in multiprocessing.active_children()},children)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'checkpoint not configured')
    def test_checkpoint_working_coordinate_parallel_matches_fresh_reference(self):
        import torch
        torch.set_num_threads(1)
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with CheckpointStrings(checkpoint,StringCompiler(),lower_conversions=False,parallel_budget=ParallelBudget(2,4*1024**3,1)) as model:
            expression=model.coordinate(2)
            self.assertTrue(model.parallel_events)
        with tempfile.TemporaryDirectory() as directory:
            reference=Path(directory)/'reference.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),checkpoint,str(reference)],check=True,capture_output=True)
            corpus=json.loads(reference.read_text())
            functions={'R32':f32,'R16':lambda x:struct.unpack('e',struct.pack('e',x))[0],'sqrt':math.sqrt,
                'Silu16':lambda x:float(torch.nn.functional.silu(torch.tensor(x,dtype=torch.float16)).item())}
            program=compile(ast.Expression(syntax(expression)),'<parallel-coordinate>','eval')
            for row in corpus['cases']:
                values={f'X{i+1}':struct.unpack('e',struct.pack('H',bits))[0] for i,bits in enumerate(row['inputBits'][0])}
                actual=eval(program,{'__builtins__':{},**functions},values)
                self.assertEqual('0x'+struct.pack('>d',actual).hex(),row['logitF64Bits'][0][2],row['label'])
            print(f'Parallel working coordinate parity: cases={len(corpus["cases"])} mismatches=0; finalParity=false')


if __name__=='__main__':unittest.main()

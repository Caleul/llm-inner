import ast
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from direct_sympy_architecture import architecture_plan,prune_plan,compose_architecture,last_position_indices
from direct_sympy_parallel import ParallelBudget
from direct_sympy_strings import StringCompiler,syntax


def numeric_functions():
    import torch
    def rounded(kind,value):
        try:return struct.unpack(kind,struct.pack(kind,value))[0]
        except OverflowError:return math.copysign(math.inf,value)
    return {'R32':lambda x:rounded('f',x),'R16':lambda x:rounded('e',x),'sqrt':math.sqrt,'And':lambda *conditions:all(conditions),
        'Silu16':lambda x:float(torch.nn.functional.silu(torch.tensor(x,dtype=torch.float16)).item()),
        'Exp32':lambda x:float(torch.exp(torch.tensor(x,dtype=torch.float32)).item())}


class LazyBranches(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='Piecewise':
            result=ast.Constant(float('nan'))
            for arm in reversed(node.args):
                value,condition=arm.elts
                result=ast.IfExp(condition,value,result)
            return ast.copy_location(result,node)
        return node


def program(text):
    tree=LazyBranches().visit(ast.Expression(syntax(text)))
    return compile(ast.fix_missing_locations(tree),'<architecture-expression>','eval')


def evaluate_plan(plan,values):
    functions=numeric_functions()
    for block in plan.blocks:
        inputs=dict(zip(block.inputs,values))
        values=tuple(eval(program(text),{'__builtins__':{},**functions},inputs) for text in block.outputs)
    return values


@unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
class ArchitectureTests(unittest.TestCase):
    def test_entire_architecture_all_logits_and_causal_lengths_against_native(self):
        import torch
        torch.set_num_threads(1)
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),checkpoint,str(path),'--all-lengths'],check=True,capture_output=True)
            corpus=json.loads(path.read_text());plans={};programs={};compared=0
            functions=numeric_functions()
            for row in corpus['cases']:
                length=len(row['inputBits'])
                if length not in plans:
                    plans[length]=architecture_plan(checkpoint,length)
                    programs[length]=[tuple(program(text) for text in block.outputs) for block in plans[length].blocks]
                values=tuple(struct.unpack('e',struct.pack('H',bits))[0] for token in row['inputBits'] for bits in token)
                for block,code in zip(plans[length].blocks,programs[length]):
                    bindings=dict(zip(block.inputs,values))
                    values=tuple(eval(p,{'__builtins__':{},**functions},bindings) for p in code)
                expected=[bit for token in row['logitF64Bits'] for bit in token]
                actual=['0x'+struct.pack('>d',value).hex() for value in values]
                self.assertEqual(actual,expected,(length,row['label']))
                compared+=len(actual)
            print(f'Whole architecture all-logit parity: cases={len(corpus["cases"])} logits={compared} lengths={sorted(plans)} mismatches=0; finalArtifactParity=false')

    def test_final_position_pruning_keeps_causal_attention_and_native_bits(self):
        import torch
        torch.set_num_threads(1)
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),checkpoint,str(path),'--all-lengths'],check=True,capture_output=True)
            corpus=json.loads(path.read_text());plans={};compared=0
            for case in corpus['cases']:
                length=len(case['inputBits'])
                if length not in plans:
                    full=architecture_plan(checkpoint,length)
                    blocks=prune_plan(full,last_position_indices(full))
                    self.assertEqual(len(blocks[-1].outputs),full.vocab)
                    # Every token supplies a key and a value to the final query.
                    self.assertEqual(set(blocks[0].inputs),{f'X{i+1}' for i in range(length*full.width)})
                    before=sum(e not in b.inputs for b in full.blocks for e in b.outputs)
                    after=sum(e not in b.inputs for b in blocks for e in b.outputs)
                    if length>1:self.assertLess(after,before)
                    from dataclasses import replace
                    plans[length]=replace(full,blocks=blocks)
                values=tuple(struct.unpack('e',struct.pack('H',bits))[0] for row in case['inputBits'] for bits in row)
                actual=['0x'+struct.pack('>d',v).hex() for v in evaluate_plan(plans[length],values)]
                self.assertEqual(actual,case['logitF64Bits'][-1],(length,case['label']))
                compared+=len(actual)
            print(f'Last-position parity: cases={len(corpus["cases"])} logits={compared} lengths={sorted(plans)} mismatches=0; finalArtifactParity=false')

    def test_complete_architecture_pair_composition_all_first_token_coordinates(self):
        import torch
        torch.set_num_threads(1)
        plan=architecture_plan(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],1)
        outputs,stats=compose_architecture(plan,StringCompiler(max_characters=512*1024**2),ParallelBudget(2,4*1024**3))
        self.assertEqual(len(outputs),plan.vocab)
        self.assertEqual(stats['operatorPairMerges'],len(plan.blocks)-1)
        functions=numeric_functions();codes=[program(text) for text in outputs]
        cases=[(-65504.,65504.),(-0.,0.),(2**-24,-2**-24),(1.,-1.),(.03125,-.015625)]
        for values in cases:
            expected=evaluate_plan(plan,values);bindings={f'X{i+1}':v for i,v in enumerate(values)}
            actual=[eval(code,{'__builtins__':{},**functions},bindings) for code in codes]
            self.assertEqual([struct.pack('d',x) for x in actual],[struct.pack('d',x) for x in expected])
        self.assertTrue(all(not set(__import__('re').findall(r'\bX[1-9][0-9]*\b',value))-{'X1','X2'} for value in outputs))
        selected=prune_plan(plan,[2]);self.assertEqual(len(selected[-1].outputs),1)
        print(f'Whole architecture pair parity: logits={len(outputs)*len(cases)} stages={len(plan.blocks)} merges={stats["operatorPairMerges"]} mismatches=0; finalArtifactParity=false')


if __name__=='__main__':unittest.main()

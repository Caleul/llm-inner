"""Parity of the working string, not admission of a final compiled artifact."""
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

from direct_sympy_checkpoint import CheckpointStrings,f32
from direct_sympy_strings import StringCompiler,syntax


class CheckpointStringTests(unittest.TestCase):
    def test_square_substitutes_once_only_for_certified_finite_half(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"config.json").write_text(json.dumps({"model_type":"llama","hidden_size":2,"num_hidden_layers":1}))
            compiler=StringCompiler()
            builder=CheckpointStrings(directory,compiler)
            result=builder.op("*","X1","X1")
            self.assertEqual(ast.dump(syntax(result)),ast.dump(syntax("X1 ** 2")))
            self.assertEqual(builder.conversions.value_kind(syntax(result)),"f32")
            self.assertTrue(builder.conversions.no_negative_zero(syntax(result)))
            self.assertEqual(len(compiler.substitution_events),1)
            self.assertTrue(all(e[2:4]==("factor","simplify") for e in compiler.events))
            builder.conversions.half_values.clear();builder.conversions.f32_values.clear()
            before=len(compiler.substitution_events)
            result=builder.op("*","X1","X1")
            self.assertEqual(len(compiler.substitution_events)-before,2)
            self.assertIn("R32",result)

    @unittest.skipUnless(os.environ.get("LLM_INNER_DIRECT_JSON_CHECKPOINT"),"Checkpoint validation fixture not configured")
    def test_re_read_working_string_matches_fresh_reference_coordinate(self):
        import torch
        checkpoint=os.environ["LLM_INNER_DIRECT_JSON_CHECKPOINT"]
        compiler=StringCompiler(max_characters=1048576)
        with CheckpointStrings(checkpoint,compiler,lower_conversions=False) as builder:
            expression=builder.coordinate(2)
            self.assertTrue(all(e[2:4]==("factor","simplify") for e in compiler.events))
            self.assertNotIn("CASBoundary",expression)
            self.assertNotIn("X99999999",expression)
        with tempfile.TemporaryDirectory(prefix="direct-sympy-parity-") as directory:
            path=Path(directory)/"coordinate.work.expr"
            path.write_text(expression+"\n")
            # Read the actual string back from disk before executing it.
            tree=syntax(path.read_text())
            program=compile(ast.Expression(tree),str(path),"eval")
            reference=Path(directory)/"reference.json"
            subprocess.run([sys.executable,str(Path(__file__).with_name("capture_direct_json_reference.py")),
                checkpoint,str(reference)],check=True,capture_output=True)
            corpus=json.loads(reference.read_text())
            half=lambda x:struct.unpack("e",struct.pack("e",x))[0]
            silu=lambda x:float(torch.nn.functional.silu(torch.tensor(x,dtype=torch.float16)).item())
            functions={"R32":f32,"R16":half,"sqrt":math.sqrt,"Silu16":silu}
            compared=0
            for row in corpus["cases"]:
                values={f"X{i+1}":struct.unpack("e",struct.pack("H",bits))[0]
                    for i,bits in enumerate(row["inputBits"][0])}
                actual=eval(program,{"__builtins__":{},**functions},values)
                bits="0x"+struct.pack(">d",actual).hex()
                self.assertEqual(bits,row["logitF64Bits"][0][2],row["label"])
                compared+=1
            self.assertEqual(compared,60)
            print(f"Working string parity: {compared} exact cases, position=0 dimension=2; numerical primitives remain, finalParity=false")


if __name__=="__main__":unittest.main()

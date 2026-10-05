import ast
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_scalar import ScalarProgram
from direct_sympy_scalar_rust import ScalarRustExpression,write_scalar_rust_candidate
from direct_sympy_strings import StringCompiler


class ScalarRustTests(unittest.TestCase):
    def test_emit_only_defined_local_dependencies_and_lazy_branch_bodies(self):
        emitter=ScalarRustExpression(('X1',),('X2',))
        _,text=emitter.emit(ast.parse('Piecewise((R16(X2), X1 > 0), (-0.0, True))',mode='eval').body,'f64')
        self.assertIn('if ',text);self.assertIn('round_binary16(v2)',text)
        with self.assertRaisesRegex(ValueError,'Unresolved scalar'):emitter.emit(ast.Name(id='X3'))
        with self.assertRaises(ValueError):emitter.emit(ast.parse('Unknown(X1)',mode='eval').body)
        with self.assertRaisesRegex(ValueError,'immediate F32'):emitter.emit(ast.parse('sqrt(X1)',mode='eval').body)
        _,root=emitter.emit(ast.parse('R32(sqrt(X1))',mode='eval').body)
        self.assertEqual(root,'round_binary32(root_binary32(input[0]))')

    def test_reject_incomplete_domain_or_unproved_numeric_enclosures(self):
        value=ScalarProgram(('X1',),(),('X1',))
        certificate={'inputDomain':'All finite Half values, including both signed zeros',
            'layers':[{'gateAbsMaximum':1/16,'scoreAbsMaximum':.01}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'out.rs'
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                write_scalar_rust_candidate(path,{1:value},{1:certificate},width=1,vocab=1,context=2,compiler=StringCompiler())
            bad={**certificate,'layers':[{'gateAbsMaximum':1.,'scoreAbsMaximum':.01}]}
            with self.assertRaisesRegex(ValueError,'Activation domain'):
                write_scalar_rust_candidate(path,{1:value},{1:bad},width=1,vocab=1,context=1,compiler=StringCompiler())
            bad={**certificate,'layers':[{'gateAbsMaximum':.01,'scoreAbsMaximum':1.}]}
            with self.assertRaisesRegex(ValueError,'Exponential domain'):
                write_scalar_rust_candidate(path,{1:value},{1:bad},width=1,vocab=1,context=1,compiler=StringCompiler())
            self.assertFalse(path.exists())

    def test_budget_failure_cannot_publish_partial_rust(self):
        value=ScalarProgram(('X1',),(('X2','R32(X1 + 1.0)'),),('X2',))
        certificate={'inputDomain':'All finite Half values, including both signed zeros',
            'layers':[{'gateAbsMaximum':.01,'scoreAbsMaximum':.01}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'out.rs'
            with patch('direct_sympy_scalar_rust.numerical_bodies',return_value='// closed bodies\n'):
                with self.assertRaisesRegex(ValueError,'budget'):
                    write_scalar_rust_candidate(path,{1:value},{1:certificate},width=1,vocab=1,context=1,compiler=StringCompiler(),max_bytes=1)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()

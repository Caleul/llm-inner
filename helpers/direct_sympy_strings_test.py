import ast
from fractions import Fraction as F
import struct
import unittest
from unittest.mock import patch
import direct_sympy_strings as engine

from direct_sympy_strings import Domain,StringCompiler,syntax


class StringCompilerTests(unittest.TestCase):
    def test_cached_proof_still_runs_cas_and_owns_the_numeric_context(self):
        compiler=StringCompiler()
        positive={"X1":Domain(F(1),F(4),0,True)}
        mixed={"X1":Domain(F(-4),F(4),0,False)}
        self.assertEqual(compiler.stabilize("X1-X1",positive),"0")
        with patch.object(engine.sp,"factor",wraps=engine.sp.factor) as factor, patch.object(engine.sp,"simplify",wraps=engine.sp.simplify) as simplify:
            self.assertEqual(compiler.stabilize("X1-X1",positive),"0")
            self.assertGreaterEqual(factor.call_count,1)
            self.assertGreaterEqual(simplify.call_count,1)
        self.assertIn("X1",compiler.stabilize("X1-X1",mixed))
        self.assertTrue(any(event[-1]=="cached-fixed-point" for event in compiler.events))

    def test_factor_after_each_substitution_reaches_inputs(self):
        compiler=StringCompiler()
        domains={f"X{i}":Domain(F(1),F(8),0,True) for i in range(1,7)}
        result=compiler.compile("14*X1 + 32*X2 + 54*X3",
            {"X1":"132*X4 + 3*X5 + 4*X6","X2":"X5","X3":"X6"},domains)
        self.assertEqual(result,"2*(924*X4 + 37*X5 + 55*X6)")
        self.assertTrue(all(event[2:4]==("factor","simplify") for event in compiler.events))
        self.assertTrue(any(event[-1]=="certified" for event in compiler.events))

    def test_common_factor_is_reduced(self):
        compiler=StringCompiler()
        domains={"X1":Domain(F(1),F(4),0,True),"X2":Domain(F(1),F(4),0,True)}
        result=compiler.stabilize("3*(2*X1 + 5*X2) + 7*(2*X1 + 5*X2)",domains)
        self.assertEqual(result,"10*(2*X1 + 5*X2)")

    def test_reused_dependencies_are_not_mistaken_for_cycles(self):
        domains={f"X{i}":Domain(F(1),F(4),0,True) for i in range(1,7)}
        result=StringCompiler().compile("X3+X4",{"X3":"X5","X4":"X5","X5":"X6"},domains)
        self.assertEqual(result,"2*X6")

    def test_branch_contexts_do_not_leak(self):
        compiler=StringCompiler()
        domains={"X1":Domain(F(-4),F(4),0,False)}
        result=compiler.substitute("Piecewise((X2, X1>0), (X2, True))","X2","X1-X1",domains)
        self.assertIn("((0), X1 > 0)",result)
        self.assertIn("X1 - X1",result)
        self.assertEqual(domains["X1"].minimum,-4)
        self.assertTrue(any(event[0]==(0,) and event[-1]=="certified" for event in compiler.events))
        self.assertTrue(any(event[0]==(1,) and event[-1]=="IEEE-barrier" for event in compiler.events))

    def test_nested_piecewise_is_processed_in_its_own_context(self):
        compiler=StringCompiler()
        domains={"X1":Domain(F(-4),F(4),0,False)}
        result=compiler.substitute("F32Add(Piecewise((X2, X1>0), (X2, True)), 0)","X2","X1-X1",domains)
        self.assertIn("Piecewise",result)
        self.assertTrue(any("nested" in event[0] and event[-1]=="certified" for event in compiler.events))

    def test_cancellation_across_inexact_ieee_operations_is_rejected(self):
        compiler=StringCompiler()
        domains={"X1":Domain(F(1),F(2**24),0,True)}
        source="(X1+1)-X1"
        result=compiler.stabilize(source,domains)
        self.assertNotEqual(result,"1")
        # Native F32 witness: factor/simplify alone would return 1.
        f32=lambda x:struct.unpack("f",struct.pack("f",x))[0]
        self.assertEqual(f32(f32(2**24+1)-2**24),0)

    def test_signed_zero_is_preserved(self):
        result=StringCompiler().stabilize("X1 * 0",{"X1":Domain(F(-1),F(1),-24,False)})
        self.assertNotEqual(result,"0")
        negative=eval(compile(ast.Expression(syntax(result)),"<math>","eval"),{"__builtins__":{}},{"X1":-1.0})
        self.assertEqual(struct.pack("d",negative),struct.pack("d",-0.0))

    def test_only_whole_variable_identifiers_are_substituted(self):
        result=StringCompiler().substitute("X1+X10","X1","X2+1",
            {"X2":Domain(F(1),F(4),0,True),"X10":Domain(F(1),F(4),0,True)})
        self.assertIn("X10",result)
        self.assertIn("X2",result)

    def test_non_dyadic_literals_and_rounding_boundaries_remain(self):
        compiler=StringCompiler()
        domains={"X1":Domain(F(1),F(4),0,True)}
        self.assertIn("0.1",compiler.stabilize("X1+0.1",domains))
        result=compiler.stabilize("F32Add(F32Mul(X1,3), F32Mul(X1,7))",domains)
        self.assertIn("F32Add",result)
        self.assertEqual(result.count("F32Mul"),2)

    def test_unreachable_branch_does_not_expand_its_dependency(self):
        compiler=StringCompiler()
        domains={"X1":Domain(F(1),F(4),0,True)}
        result=compiler.compile("Piecewise((X2, X1<0), (X1, True))",{"X2":"X2+1"},domains)
        self.assertNotIn("X2",result)

    def test_cycles_limits_and_executable_syntax_are_rejected(self):
        compiler=StringCompiler(max_characters=64)
        with self.assertRaisesRegex(ValueError,"Cyclic"):
            compiler.compile("X2",{"X2":"X2+1"},{})
        with self.assertRaisesRegex(ValueError,"budget"):
            compiler.substitute("X1+X1","X1","+".join(["X2"]*40),{})
        for expression in ("__import__('os')", "X1.__class__", "[X1]", "(lambda: 1)()"):
            with self.assertRaises(ValueError): syntax(expression)


if __name__=="__main__":
    unittest.main()

import ast
from fractions import Fraction as F
import struct
import unittest
from unittest.mock import patch
import direct_sympy_strings as engine

from direct_sympy_strings import Domain,StringCompiler,syntax


class StringCompilerTests(unittest.TestCase):
    def test_repeated_word_decisions_propagate_through_both_paths(self):
        domains={"X1":Domain(F(-65504),F(65504),-24,False)}
        guard="Bits64(X1) < 9223372036854775808"
        source=f"Piecewise((Piecewise((X1,{guard}),(777.0,True)),{guard}),(Piecewise((999.0,{guard}),(-X1,True)),True))"
        compiler=StringCompiler();result=compiler.stabilize(source,domains)
        self.assertEqual(result.count("Piecewise("),1)
        self.assertNotIn("777",result);self.assertNotIn("999",result)
        self.assertTrue(any(e[-1]=="known-true" for e in compiler.condition_events))
        self.assertTrue(any(e[-1]=="known-false" for e in compiler.condition_events))
        self.assertTrue(all(e[2:4]==("factor","simplify") for e in compiler.events))
        region=compiler.stabilize(f"R32(Piecewise((X1,{guard}),(-X1,True)))",domains)
        compiler.register_completed_region(region,domains)
        composed=f"Piecewise(({region},{guard}),({region},True))"
        self.assertEqual(compiler.compact_regions(composed,compiler.context(domains)),(composed,{}))
        self.assertEqual(compiler.stabilize(composed,domains).count("Piecewise("),1)

    def test_truth_fact_cache_keeps_siblings_and_signed_zero_separate(self):
        domains={"X1":Domain(F(-1),F(1),-24,False)}
        compiler=StringCompiler();condition=syntax("Bits64(X1) < 9223372036854775808")
        source="Piecewise(((-0.0), Bits64(X1) < 9223372036854775808), (X1, True))"
        positive=compiler.branch_facts.assume(condition,True,())
        negative=compiler.branch_facts.assume(condition,False,())
        self.assertEqual(ast.dump(syntax(compiler.stabilize(source,domains,facts=positive))),ast.dump(syntax("-0.0")))
        self.assertEqual(compiler.stabilize(source,domains,facts=negative),"X1")
        self.assertIn("Piecewise",compiler.stabilize(source,domains))
        self.assertIn("Piecewise",compiler.stabilize(source,domains))

    def test_boolean_path_facts_preserve_nan_relational_negation(self):
        domains={"X1":Domain(F(-1),F(1),-24,False),"X2":Domain(F(-1),F(1),-24,False)}
        compiler=StringCompiler();a="Bits64(X1)<9223372036854775808";b="Bits64(X2)<9223372036854775808"
        source=f"Piecewise((Piecewise((1.0,{a}),(2.0,True)), And({a},{b})),(3.0,True))"
        result=compiler.stabilize(source,domains)
        self.assertEqual(result.count("Piecewise("),1);self.assertNotIn("2.0",result)
        # not(x < 0) cannot imply x >= 0 for a computed NaN.
        nan="Float64(9221120237041090560)"
        source=f"Piecewise((Piecewise((1.0,{nan}>=0),(2.0,True)),Not({nan}<0)),(3.0,True))"
        result=compiler.stabilize(source,domains)
        self.assertEqual(result.count("Piecewise("),2)

    def test_completed_region_envelope_preserves_literal_expression_and_budget(self):
        domains={"X1":Domain(F(-4),F(4),0,False)}
        compiler=StringCompiler(max_characters=20000)
        region=compiler.stabilize("R32("+" + ".join(["R32(X1 + (-0.0))"]*200)+")",domains)
        compiler.register_completed_region(region,domains)
        reference=StringCompiler(max_characters=20000).substitute("R32(X7*2)","X7",region,domains)
        with patch.object(engine,"syntax",wraps=engine.syntax) as parse,patch.object(engine.sp,"factor",wraps=engine.sp.factor) as factor,patch.object(engine.sp,"simplify",wraps=engine.sp.simplify) as simplify:
            actual=compiler.substitute("R32(X7*2)","X7",region,domains)
            self.assertLess(max(len(call.args[0]) for call in parse.call_args_list),200)
            self.assertGreaterEqual(factor.call_count,1);self.assertGreaterEqual(simplify.call_count,1)
        self.assertEqual(ast.dump(syntax(actual)),ast.dump(syntax(reference)))
        self.assertNotIn("CASStableRegion",actual)
        self.assertTrue(any(e[-1]=="completed-regions-envelope" for e in compiler.events))
        # Compounds reuse admission too; a call cannot merge with another
        # identifier or become a callable when a cache marker replaces it.
        with patch.object(engine,"syntax",wraps=engine.syntax) as parse:
            compiler.substitute("R32(X8)","X8","R32("+region+"*0.5)",domains)
        self.assertLess(max(len(call.args[0]) for call in parse.call_args_list),200)
        own=compiler.context(domains)
        self.assertEqual(compiler.compact_regions("My"+region,own),("My"+region,{}))
        for suffix in ("X1","1","(X1)"," (X1)"):
            invalid=region+suffix
            self.assertEqual(compiler.compact_regions(invalid,own),(invalid,{}))
            with self.assertRaises((ValueError,SyntaxError)):compiler.stabilize(invalid,domains)
        compiler.max_characters=len(actual)-1
        with self.assertRaises(ValueError):compiler.stabilize(actual,domains)

    def test_completed_regions_reopen_for_external_branches_and_changed_domains(self):
        domains={"X1":Domain(F(-4),F(4),0,False)}
        compiler=StringCompiler();reference=StringCompiler()
        region=compiler.stabilize("R32(Piecewise((X1-X1,X1>0),(X1-X1,True)))",domains)
        compiler.register_completed_region(region,domains)
        expression="Piecewise(("+region+",X1>0),("+region+",True))"
        self.assertEqual(compiler.compact_regions(expression,compiler.context(domains)),(expression,{}))
        self.assertEqual(compiler.stabilize(expression,domains),reference.stabilize(expression,domains))
        word_branch="Piecewise(("+region+",Bits64(X1)>1),("+region+",True))"
        compact,regions=compiler.compact_regions(word_branch,compiler.context(domains))
        self.assertEqual((compact,regions),(word_branch,{}))
        self.assertEqual(compiler.stabilize(word_branch,domains),reference.stabilize(word_branch,domains))
        positive={"X1":Domain(F(1),F(4),0,True)}
        self.assertEqual(compiler.compact_regions(region,compiler.context(positive)),(region,{}))
        self.assertEqual(compiler.stabilize(region,positive),reference.stabilize(region,positive))
        with patch.object(engine,"_syntax_cache_limit",1):
            self.assertEqual(compiler.stabilize(region,positive),reference.stabilize(region,positive))

    def test_over_budget_composition_simplifies_before_literal_allocation(self):
        domains={"X1":Domain(F(-4),F(4),0,False)}
        compiler=StringCompiler(max_characters=10000)
        region=compiler.stabilize("R32("+" + ".join(["R32(X1+(-0.0))"]*100)+")",domains)
        compiler.register_completed_region(region,domains)
        compiler.max_characters=len(region)+100
        with patch.object(engine.sp,"factor",wraps=engine.sp.factor) as factor,patch.object(engine.sp,"simplify",wraps=engine.sp.simplify) as simplify:
            result=compiler.substitute("Piecewise((X9+X9,False),(1.0,True))","X9",region,domains)
        self.assertGreater(factor.call_count,0);self.assertGreater(simplify.call_count,0)
        self.assertEqual(float(engine.symbolic(syntax(result))),1.0)
        self.assertNotIn("CASStableRegion",result)
        self.assertEqual(compiler.substitution_events[-1][-1],"admitted-after-budget-simplification")
        self.assertTrue(compiler.budget_events[-1][-1])
        with self.assertRaisesRegex(ValueError,"budget before allocation"):
            compiler.substitute("R32(X9+X9)","X9",region,domains)
        self.assertFalse(compiler.budget_events[-1][-1])
        before=len(compiler.budget_events)
        with self.assertRaises(ValueError):
            compiler.substitute("Piecewise((X9+X9,X1>0),(1.0,True))","X9",region,domains)
        self.assertEqual(len(compiler.budget_events),before)

    def test_substitution_budget_records_growth_without_parsing_the_replacement(self):
        compiler=StringCompiler(max_characters=10)
        replacement="X2+X2+X2"
        with patch.object(engine,"syntax",side_effect=AssertionError("Over-budget replacement must not be parsed")):
            with self.assertRaisesRegex(ValueError,"occurrences=2 estimatedCharacters=23 limit=10"):
                compiler.substitute("X1+X1","X1",replacement,{})
        self.assertEqual(compiler.failed_substitution,("X1+X1","X1",replacement))
        self.assertEqual(compiler.substitution_events[-1],("X1",5,8,2,23,None,"budget"))

    def test_repeated_grammar_admission_returns_independent_trees(self):
        first=syntax("R32(X12345 + (-0.0))")
        first.args[0].left.id="Changed"
        second=syntax("R32(X12345 + (-0.0))")
        self.assertEqual(second.args[0].left.id,"X12345")
        self.assertIsInstance(second.args[0].right,ast.UnaryOp)
        for _ in range(2):
            with self.assertRaises(ValueError):syntax("R32(X12345.__class__)")

    def test_grammar_cache_is_bounded_and_eviction_preserves_validation(self):
        with patch.object(engine,"_validated_syntax",engine.OrderedDict()),patch.object(engine,"_validated_characters",0),patch.object(engine,"_syntax_cache_limit",10):
            syntax("X12345")
            syntax("X67890")
            self.assertLessEqual(engine._validated_characters,10)
            self.assertNotIn("X12345",engine._validated_syntax)
            self.assertEqual(syntax("X12345").id,"X12345")
            self.assertLessEqual(engine._validated_characters,10)

    def test_equal_protected_calls_share_only_the_cas_atom(self):
        view,protected=engine.cas_view("R32(X1)+R32(X1)+R32(-0.0)+R32(0.0)")
        self.assertEqual(len(protected),3)
        self.assertEqual(view.left.left.left.id,view.left.left.right.id)
        self.assertNotEqual(view.left.right.id,view.right.id)
        source="R32(X1)+R32(X1)"
        compiler=StringCompiler()
        result=compiler.stabilize(source,{"X1":Domain(F(-1),F(1),-24,False)})
        self.assertEqual(result.count("R32("),2)
        self.assertNotIn("CASBoundary",result)

    def test_cached_envelope_does_not_reopen_descendant_branches(self):
        expression="Piecewise((Piecewise((X1, X1>0), (-X1, True)), X1<2), (0, True))"
        view,protected=engine.cas_view(expression,keep_piecewise=True)
        calls=[n for n in ast.walk(view) if isinstance(n,ast.Call) and n.func.id=="Piecewise"]
        self.assertEqual(len(calls),1)
        self.assertTrue(any(value.startswith("Piecewise(") for value in protected.values()))
        compiler=StringCompiler()
        domains={"X1":Domain(F(-4),F(4),0,False)}
        result=compiler.stabilize(expression,domains)
        branch_events=[event for event in compiler.events if len(event[0])>=2]
        self.assertTrue(branch_events)
        self.assertEqual(compiler.stabilize(expression,domains),result)

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

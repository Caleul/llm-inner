"""Scoped algebra over closed producers, verified in native emitted code."""
import ast
import os
from fractions import Fraction as F
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_arithmetic import factor_certified,simplify_arithmetic
from direct_sympy_conversions import ConversionSession,FiniteSource
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax


class ExactFactorTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture required')
    def test_full_coordinate_compiler_growth_does_not_regress_after_factoring(self):
        from direct_sympy_checkpoint import CheckpointStrings
        from direct_sympy_streaming_literals import streaming_literals
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        def compile_size():
            with CheckpointStrings(checkpoint,StringCompiler(max_characters=8388608)) as model:
                with streaming_literals(model) as registry:
                    expression=model.coordinate(2)
                    return registry.size(expression)
        with patch('direct_sympy_arithmetic.factor_certified',side_effect=lambda text,session:text):
            baseline=compile_size()
        actual=compile_size()
        self.assertLessEqual(actual,baseline)
        print(f'Complete composition growth regression: characters={baseline}->{actual}; finalArtifactEmitted=false')

    def test_factored_quadratic_keeps_its_specialized_activation_certificate(self):
        from direct_sympy_silu import quadratic_source,quadratic_subnormal_guard,quadratic_tandem_source
        session,key=self.session()
        session.completed[key]=FiniteSource(-3/128,3/128,-24)
        x='CompileValue0()'
        forms=[f'{x} * (0.5 + {x} * 0.25)',f'{x} * (0.5 + {x}/4)',
            f'{x} * (({x} + 2)/4)',f'({x} * ({x} + 2))/4']
        expected=quadratic_tandem_source(syntax(forms[0]),session)
        guard=quadratic_subnormal_guard(syntax(forms[0]),session)
        for text in forms:
            self.assertEqual(ast.unparse(quadratic_source(syntax(text),session)),x)
            self.assertEqual(quadratic_tandem_source(syntax(text),session),expected)
            self.assertEqual(quadratic_subnormal_guard(syntax(text),session),guard)
        for text in [f'{x} * ({x} + 2)/3',f'{x} * ({x} + 3)/4',f'{x}*{x}/4 + {x}/2']:
            self.assertIsNone(quadratic_source(syntax(text),session))
        session.completed[key]=FiniteSource(-1,1,-24)
        for text in forms:self.assertIsNone(quadratic_source(syntax(text),session))

    def session(self):
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(-1),F(1),-24,False)},input_dtype='f16')
        key=session.key(syntax('CompileValue0()'))
        session.completed[key]=FiniteSource(-1,1,-24)
        session.half_values.add(key)
        session.no_negative_zero_values.add(key)
        return session,key

    def test_common_producer_is_factored_and_native_parity_preserves_both_input_zeros(self):
        session,_=self.session()
        sources=['3.0*CompileValue0() + 7.0*CompileValue0()',
            '3.0*(2.0*CompileValue0() + 5.0) + 7.0*(2.0*CompileValue0() + 5.0)',
            'CompileValue0()/2.0 + CompileValue0()/4.0']
        results=[]
        import direct_sympy_arithmetic as arithmetic
        with patch.object(arithmetic.sympy,'factor',wraps=arithmetic.sympy.factor) as factor,patch.object(arithmetic.sympy,'simplify',wraps=arithmetic.sympy.simplify) as simplify:
            for expression in sources:
                result=simplify_arithmetic(expression,session)
                self.assertLess(len(result),len(expression))
                self.assertEqual(result.count('CompileValue0()'),1)
                self.assertNotIn('CASExactLeaf',result)
                self.assertEqual(simplify_arithmetic(result,session),result)
                wrapped=simplify_arithmetic('R16('+expression+')',session)
                self.assertEqual(wrapped.count('CompileValue0()'),1)
                results.append(result)
            self.assertGreater(factor.call_count,0);self.assertGreater(simplify.call_count,0)
        # The alias denotes a real closed finite Half producer. Restore it
        # before compiling either version; no runtime temporary is emitted.
        def emitted(text):
            return cpp(syntax(text.replace('CompileValue0()','(X1 + 0.0)')))
        functions='\n'.join(f'double before{i}(double X1){{return {emitted(a)};}}\ndouble after{i}(double X1){{return {emitted(b)};}}' for i,(a,b) in enumerate(zip(sources,results)))
        checks=''.join(f'mismatches+=word<uint64_t>(before{i}(x))!=word<uint64_t>(after{i}(x));cases++;' for i in range(len(results)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'factor.cpp';binary=root/'factor'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+functions+\
                '\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0;for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));if(x < -1 || x > 1)continue;'+checks+'}std::printf("Exact producer factoring: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            run=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=92166 mismatches=0',run.stdout);print(run.stdout,end='')

    def test_inexact_arithmetic_unknown_calls_and_negative_zero_remain_barriers(self):
        session,key=self.session()
        expression='3.0*CompileValue0() + 7.0*CompileValue0()'
        session.no_negative_zero_values.clear()
        self.assertEqual(factor_certified(expression,session),expression)
        session.no_negative_zero_values.add(key)
        for bound in [FiniteSource(-1e20,1e20,0),FiniteSource(-1,1,None)]:
            session.completed[key]=bound
            self.assertEqual(factor_certified(expression,session),expression)
        session.completed[key]=FiniteSource(-1,1,-24)
        for text in ['0.1*CompileValue0() + 0.2*CompileValue0()',
            '3.0*Unknown(X1) + 7.0*Unknown(X1)',
            'CompileValue0()/3.0 + CompileValue0()/7.0',
            '0.5 + CompileValue0()*0.25',
            'CompileValue0()*(0.5 + CompileValue0()*0.25)']:
            self.assertEqual(factor_certified(text,session),text)
        session.completed.pop(key)
        self.assertEqual(factor_certified(expression,session),expression)

    def test_narrow_path_proofs_never_escape_to_sibling_or_original_session(self):
        session,key=self.session()
        source='3.0*CompileValue0() + 7.0*CompileValue0()'
        session.completed[key]=FiniteSource(-1e20,1e20,0)
        before=dict(session.completed)
        with session.branch_context(session.domains,{key:FiniteSource(-1,1,-24)},()):
            self.assertNotEqual(factor_certified(source,session),source)
        self.assertEqual(session.completed,before)
        self.assertEqual(factor_certified(source,session),source)


if __name__=='__main__':unittest.main()

"""Eliminate whole producers only after exact signed IEEE cell proofs."""
import ast
from fractions import Fraction as F
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_conversions import ConversionSession,FiniteSource
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax,refine


class ConstantCellTests(unittest.TestCase):
    def test_complete_producers_disappear_with_native_all_finite_half_parity(self):
        cases=[('R32(1.0 + X1 * 8.881784197001252e-16)', 'double(float(1.0+x*0x1p-50))'),
            ('R16(1.0 + X1 * 9.094947017729282e-13)', 'double(_Float16(1.0+x*0x1p-40))'),
            ('R32(sqrt(1.0 + X1 * 8.881784197001252e-16))','double(float(std::sqrt(1.0+x*0x1p-50)))'),
            ('R32(1.0 / R32(sqrt(1.0 + X1 * 8.881784197001252e-16)))','double(float(1.0/double(float(std::sqrt(1.0+x*0x1p-50)))))'),
            ('R16(-1.0 + X1 * 9.094947017729282e-13)','double(_Float16(-1.0+x*0x1p-40))')]
        compiled=[]
        for expression,reference in cases:
            session=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False)},input_dtype='f16')
            # No numerical lowering is required after eliminating the source.
            with patch('direct_sympy_conversions.lower_finite_conversion',side_effect=AssertionError('Unnecessary conversion expansion')):
                result=session.close(expression)
            self.assertNotIn('X1',result);self.assertNotIn('sqrt',result)
            self.assertEqual(float(result),-1.0 if expression.startswith('R16(-') else 1.0)
            self.assertGreater(session.arithmetic_eliminated,0)
            compiled.append((result,reference))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            functions=[];checks=[]
            for index,(result,reference) in enumerate(compiled):
                artifact=root/f'cell-{index}.expr';artifact.write_text(result)
                functions.append(f'double candidate{index}(double X1){{return '+cpp(syntax(artifact.read_text()))+';}')
                checks.append(f'cases++;if(word<uint64_t>(candidate{index}(x))!=word<uint64_t>({reference}))return 1;')
            source=root/'native.cpp';binary=root/'native'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+ '\n'.join(functions)+'''\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;
double x=word<_Float16>(uint16_t(bits));'''+''.join(checks)+'''}
std::printf("Constant-cell native parity: cases=%u mismatches=0\\n",cases);}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('cases=317440 mismatches=0',tested.stdout);print(tested.stdout,end='')

    def test_midpoint_payloads_zero_signs_overflow_and_impurity_are_barriers(self):
        def cell(low,high,kind='R16',positive_zero=False):
            session=ConversionSession(StringCompiler(),{'X1':Domain(F(low),F(high),-1074,positive_zero)})
            return session.constant_rounding_cell(syntax(kind+'(X1)'))
        half=1+2**-11
        self.assertEqual(cell(1,half),1.0) # Even center owns the tie.
        self.assertIsNone(cell(1,math.nextafter(half,math.inf)))
        self.assertIsNone(cell(half,1+2**-10)) # Odd center excludes its tie.
        self.assertEqual(cell(math.nextafter(half,math.inf),1+2**-10),1+2**-10)
        f32=1+2**-24
        self.assertEqual(cell(1,f32,'R32'),1.0)
        self.assertIsNone(cell(1,math.nextafter(f32,math.inf),'R32'))
        for low,high in ((-2**-26,2**-26),(0,0),(0,2**-26),(-2**-26,0)):
            self.assertIsNone(cell(low,high))
        self.assertEqual(cell(0,0,positive_zero=True),0.0)
        self.assertEqual(math.copysign(1,cell(-2**-26,-2**-27)),-1)
        self.assertEqual(math.copysign(1,cell(2**-27,2**-26)),1)
        self.assertIsNone(cell(65520,65520))
        session=ConversionSession(StringCompiler(),{})
        node=syntax('unknown()');key=session.key(node);session.completed[key]=FiniteSource(1,1,0)
        session.converted_regions.add(key)
        self.assertIsNone(session.constant_rounding_cell(syntax('R16(unknown())')))
        session.pure_numeric_regions.add(key)
        self.assertEqual(session.constant_rounding_cell(syntax('R16(unknown())')),1)

    def test_branch_cells_and_virtual_purity_do_not_leak_to_siblings(self):
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(-2),F(2),-24,False)},input_dtype='f16')
        expression='Piecewise((R16(1.0 + X1 * 0.00048828125), And(X1 >= 0.0, X1 <= 0.5)), (R16(1.0 + X1 * 0.001953125), True))'
        result=session.close(expression);node=syntax(result)
        self.assertEqual(float(ast.unparse(node.args[0].elts[0])),1.0)
        self.assertNotEqual(ast.unparse(node.args[1].elts[0]),'1.0')
        self.assertIsNone(session.constant_rounding_cell(syntax('R16(1.0 + X1 * 0.00048828125)')))
        narrowed=refine(session.domains,syntax('Or(X1 < 0.0, X1 > 0.5)'),False)
        self.assertEqual((narrowed['X1'].minimum,narrowed['X1'].maximum),(F(0),F(1,2)))
        self.assertIsNone(refine(session.domains,syntax('And(X1 < 0.0, X1 >= 0.0)'),True))
        self.assertEqual(refine(session.domains,syntax('Not(X1 < 0.0)'),True),dict(session.domains))
        self.assertEqual(refine(session.domains,syntax('Or(X1 < 0.0, X1 > 0.5)'),True),dict(session.domains))
        marker=session.key(syntax('temporary()'))
        with session.branch_context(session.domains,{},()):session.pure_numeric_regions.add(marker)
        self.assertNotIn(marker,session.pure_numeric_regions)
        # Closed literal substitution establishes purity locally before its
        # numeric cell is queried; the large body need not be restored.
        wide=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False)},input_dtype='f16')
        source=wide.close('R16(X1/3.0)')
        result=wide.compose_closed('R32(1.0 + X999999998 * 8.881784197001252e-16)',{'X999999998':source})
        self.assertEqual(float(result),1.0)
        self.assertNotIn('CASNumeric',result)


    def test_zero_magnitude_eliminates_the_producer_but_keeps_its_sign(self):
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False)},input_dtype='f16')
        source=session.close('R16(X1/3.0)')
        result=session.compose_closed('R16(X999999998*'+repr(2**-80)+')',{'X999999998':source})
        self.assertLess(len(result),len(source))
        self.assertNotIn('R16(',result)
        bounds=session.bounds(syntax(result));self.assertEqual((bounds.minimum,bounds.maximum),(0,0))
        self.assertFalse(session.no_negative_zero(syntax(result)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifact=root/'signed-zero.expr';artifact.write_text(result)
            native=root/'native.cpp';binary=root/'native'
            native.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+'double candidate(double X1){return '+cpp(syntax(artifact.read_text()))+';}'+"\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));double expected=double(_Float16(double(_Float16(x/3.0))*0x1p-80));cases++;if(word<uint64_t>(candidate(x))!=word<uint64_t>(expected))return 1;}std::printf(\"Signed-zero cell native parity: cases=%u mismatches=0\\n\",cases);}")
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(native),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('cases=63488 mismatches=0',tested.stdout);print(tested.stdout,end='')


if __name__=='__main__':unittest.main()

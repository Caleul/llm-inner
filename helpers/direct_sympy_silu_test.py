import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction as F
import torch
from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp
from direct_sympy_silu import supported,expand,quadratic_subnormal_guard,quadratic_tandem_source


class BoundedSiluTests(unittest.TestCase):
    def test_emitted_word_kernel_matches_all_bounded_half_scalar_and_vector_values(self):
        self.verify_emitted(F(1,16),22530,1000,24)

    def test_smaller_quadratic_kernel_matches_every_half_in_its_certified_domain(self):
        self.verify_emitted(F(3,128),19458,450,7)
        # Enlarging this certificate to 1/32 is numerically wrong.
        x=struct.unpack('e',struct.pack('H',42992))[0]
        quadratic=struct.unpack('e',struct.pack('e',struct.unpack('f',struct.pack('f',x*(0.5+x*0.25)))[0]))[0]
        self.assertNotEqual(quadratic,torch.nn.functional.silu(torch.tensor(x,dtype=torch.float16)).item())

    def test_quadratic_guard_is_the_exact_threshold_preimage_for_all_admitted_half_values(self):
        session=ConversionSession(StringCompiler(),{'X1':Domain(-F(3,128),F(3,128),-24,False)},input_dtype='f16')
        raw=syntax('X1*(0.5+X1*0.25)')
        guard=quadratic_subnormal_guard(raw,session)
        self.assertEqual(guard.count('X1'),1)
        cases=0
        for bits in range(65536):
            x=struct.unpack('e',struct.pack('H',bits))[0]
            if not abs(x)<=3/128:continue
            shifted=F(x)+F(2)**-25
            self.assertEqual(F(x+2**-25),shifted)
            self.assertEqual(F((x+2**-25)**2),shifted**2)
            expected=abs(struct.unpack('f',struct.pack('f',x*(0.5+x*0.25)))[0])<2**-14
            self.assertEqual((x+2**-25)**2<2**-26,expected,(bits,x))
            cases+=1
        self.assertEqual(cases,19458)
        untyped=ConversionSession(StringCompiler(),session.domains)
        self.assertIsNone(quadratic_subnormal_guard(raw,untyped))
        wide=ConversionSession(StringCompiler(),{'X1':Domain(-F(1,32),F(1,32),-24,False)},input_dtype='f16')
        self.assertIsNone(quadratic_subnormal_guard(raw,wide))
        for expression in ('X1*(0.5+X1*0.5)','X1*(0.5+X2*0.25)','X1*0.5'):
            self.assertIsNone(quadratic_subnormal_guard(syntax(expression),session))
        from unittest.mock import patch
        with patch('direct_sympy_silu.quadratic_tandem_source',return_value=None):
            with patch('direct_sympy_silu.quadratic_subnormal_guard',return_value=None):baseline=session.close('Silu16(X1)')
            result=session.close('Silu16(X1)')
        self.assertEqual(baseline.count('X1'),7)
        self.assertEqual(result.count('X1'),6)
        self.assertLess(len(result),len(baseline))
        print('Quadratic Half threshold preimage: cases=19458 mismatches=0 sourceCopies=7->6')

    def test_completed_square_is_exact_except_raw_negative_zero_and_closes_at_half_boundary(self):
        from unittest.mock import patch
        session=ConversionSession(StringCompiler(),{'X1':Domain(-F(3,128),F(3,128),-24,False)},input_dtype='f16')
        raw=syntax('X1*(0.5+X1*0.25)')
        magnitude,sign=quadratic_tandem_source(raw,session)
        self.assertEqual(sign,'X1');self.assertEqual(magnitude.count('X1'),1)
        cases=0;zero_differences=0
        for bits in range(65536):
            x=struct.unpack('e',struct.pack('H',bits))[0]
            if not abs(x)<=3/128:continue
            expected=F(x)*(F(1,2)+F(x)/4)
            old=x*(.5+x*.25);new=((x+1.0)**2-1.0)*.25
            self.assertEqual(F(old),expected);self.assertEqual(F(new),expected)
            if expected:self.assertLessEqual(abs(expected.numerator).bit_length(),37)
            if struct.pack('d',old)!=struct.pack('d',new):
                self.assertEqual(bits,0x8000);zero_differences+=1
            cases+=1
        self.assertEqual((cases,zero_differences),(19458,1))
        untyped=ConversionSession(StringCompiler(),session.domains)
        self.assertIsNone(quadratic_tandem_source(raw,untyped))
        wide=ConversionSession(StringCompiler(),{'X1':Domain(-F(1,32),F(1,32),-24,False)},input_dtype='f16')
        self.assertIsNone(quadratic_tandem_source(raw,wide))
        with patch('direct_sympy_silu.quadratic_tandem_source',return_value=None):baseline=session.close('Silu16(X1)')
        from direct_sympy_tandem import lower_tandem
        candidate=lower_tandem(magnitude,session.bounds(raw),session.compiler,session.domains,integer_word_exact=True,sign_expression=sign,small_condition=quadratic_subnormal_guard(raw,session))
        self.assertEqual(baseline.count('X1'),6);self.assertEqual(candidate.count('X1'),4)
        # The short input remains in the smaller existing form; composed
        # producers can make the four-occurrence candidate smaller instead.
        self.assertEqual(session.close('Silu16(X1)'),baseline)
        self.verify_emitted(F(3,128),19458,450,4,kernel=candidate)
        # Native all-Half parity, including -0, is independently covered by
        # test_smaller_quadratic_kernel_matches_every_half_in_its_certified_domain.
        print('Completed square tandem certificate: cases=19458 rawNegativeZeroDifferences=1 sourceCopies=6->4')

    def test_activation_closes_a_previously_certified_literal_in_the_same_context(self):
        self.verify_emitted(F(3,64),21506,10000,100,previous=True)

    def verify_emitted(self,bound,expected_cases,max_characters,max_references,previous=False,kernel=None):
        session=ConversionSession(StringCompiler(),{'X1':Domain(-bound,bound,-24,False)},input_dtype='f16')
        producer=session.close('R16(X1*0.5)') if previous else 'X1'
        result=session.close('Silu16('+producer+')')
        self.assertEqual(session.activations_closed,1)
        if kernel is not None:result=kernel
        self.assertLessEqual(len(result),max_characters)
        self.assertLessEqual(result.count("X1"),max_references)
        for primitive in ('R16(','R32(','Silu16(','exp(','sqrt(','CASNumericRegion'):
            self.assertNotIn(primitive,result)
        pairs=[];values=[]
        for bits in range(65536):
            x=struct.unpack('e',struct.pack('H',bits))[0]
            if abs(x)<=bound:
                operand=struct.unpack('e',struct.pack('e',x*0.5))[0] if previous else x
                expected=torch.nn.functional.silu(torch.tensor(operand,dtype=torch.float16)).item()
                pairs.append((bits,struct.unpack('H',struct.pack('e',expected))[0]));values.append(operand)
        self.assertEqual(len(pairs),expected_cases)
        vector=torch.nn.functional.silu(torch.tensor(values,dtype=torch.float16)).view(torch.int16).tolist()
        self.assertEqual([b for _,b in pairs],[b&65535 for b in vector])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gold=root/'gold.bin';gold.write_bytes(b''.join(struct.pack('HH',a,b) for a,b in pairs))
            expression=root/'activation.work.expr';expression.write_text(result+'\n')
            source=root/'silu.cpp';binary=root/'silu'
            source.write_text('#include <cmath>\n#include <cstdio>\n#include <cstdint>\n#include <cstring>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double compiled(double X1){return '+cpp(syntax(expression.read_text()))+';}\n'+'''int main(int argc,char**argv){FILE*f=std::fopen(argv[1],"rb");uint16_t pair[2];unsigned cases=0,mismatches=0;while(std::fread(pair,2,2,f)==2){double x=word<_Float16>(pair[0]);double actual=compiled(x),expected=word<_Float16>(pair[1]);mismatches+=word<uint64_t>(actual)!=word<uint64_t>(expected);cases++;}std::printf("Bounded SiLU word parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary),str(gold)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn(f'cases={expected_cases} mismatches=0',run.stdout);print(run.stdout,end='')
        full=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False)},input_dtype='f16')
        self.assertFalse(supported('X1',full))
        self.assertEqual(full.close('Silu16(X1)'),'Silu16(X1)')
        with self.assertRaisesRegex(ValueError,'certificate requires'):expand('X1',full)
        untyped=ConversionSession(StringCompiler(),session.domains)
        self.assertFalse(supported('X1',untyped))


if __name__=='__main__':unittest.main()

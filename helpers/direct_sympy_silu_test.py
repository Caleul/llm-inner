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
from direct_sympy_silu import supported,expand


class BoundedSiluTests(unittest.TestCase):
    def test_emitted_word_kernel_matches_all_bounded_half_scalar_and_vector_values(self):
        self.verify_emitted(F(1,16),22530,1000,24)

    def test_smaller_quadratic_kernel_matches_every_half_in_its_certified_domain(self):
        self.verify_emitted(F(3,128),19458,450,7)
        # Enlarging this certificate to 1/32 is numerically wrong.
        x=struct.unpack('e',struct.pack('H',42992))[0]
        quadratic=struct.unpack('e',struct.pack('e',struct.unpack('f',struct.pack('f',x*(0.5+x*0.25)))[0]))[0]
        self.assertNotEqual(quadratic,torch.nn.functional.silu(torch.tensor(x,dtype=torch.float16)).item())

    def test_activation_closes_a_previously_certified_literal_in_the_same_context(self):
        self.verify_emitted(F(3,64),21506,10000,100,previous=True)

    def verify_emitted(self,bound,expected_cases,max_characters,max_references,previous=False):
        session=ConversionSession(StringCompiler(),{'X1':Domain(-bound,bound,-24,False)},input_dtype='f16')
        producer=session.close('R16(X1*0.5)') if previous else 'X1'
        result=session.close('Silu16('+producer+')')
        self.assertEqual(session.activations_closed,1)
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

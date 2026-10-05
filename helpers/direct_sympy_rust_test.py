"""Native Rust candidate parity; this fixture is not the Llama artifact."""
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from direct_sympy_rust import emit_scalar,write_rust_candidate,validate_rust_candidate


def word(value):return '0x'+struct.pack('>d',value).hex()


class RustCandidateTests(unittest.TestCase):
    def test_residual_primitives_and_internal_inputs_are_rejected(self):
        for expression in ['R16(X1)','R32(X1)','sqrt(X1)','Silu16(X1)','Exp32(X1)',
                           'CompileValue0()','CASNumericRegion0()','X3','X1**3']:
            with self.assertRaises(ValueError):emit_scalar(expression,2)
        self.assertIn('input[0]',emit_scalar('X1**2',2))

    def test_incomplete_domain_and_output_budget_preserve_previous_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.rs';path.write_text('previous\n')
            for by_length,budget in [({1:['X1']*4},4096),({1:['X1']*4,2:['X1']*4},32),
                                    ({1:['X1']*4,2:['R16(X1)']*4},4096)]:
                with self.assertRaises(ValueError):write_rust_candidate(path,by_length,width=2,vocab=4,context=2,max_bytes=budget)
                self.assertEqual(path.read_text(),'previous\n')
                self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    @unittest.skipUnless(shutil.which('rustc'),'Rust compiler required')
    def test_actual_rust_preserves_last_position_words_and_lazy_length_selection(self):
        expressions={};cases=[]
        pairs=[(-0.,0.),(0.,-0.),(-65504.,2**-24),(2**-24,-2**-24),(1.,3.)]
        for length in range(1,9):
            a,b=f'X{2*length-1}',f'X{2*length}'
            expressions[length]=[f'({a}) + ({b})',f'Float64(U64Xor(Bits64({a}),9223372036854775808))',
                f'Float64(U64Or(U64And(Bits64({a}),9223372036854775808),U64And(Bits64({b}),9223372036854775807)))',
                f'Piecewise((Float64(U64Add(18446744073709551615,4607182418800017409)), U64Shr(Bits64({a}),64) == 0), ({b}, True))']
            for a,b in pairs:
                rows=[[1.,-1.]]*(length-1)+[[a,b]]
                inputs=[[struct.unpack('<H',struct.pack('<e',x))[0] for x in row] for row in rows]
                expected=[word(a+b),word(-a),word(math.copysign(abs(b),a)),word(1.)]
                cases.append({'inputBits':inputs,'logitF64Bits':[[word(999.)]*4]*(length-1)+[expected]})
        reference={'width':2,'vocab':4,'context':8,'torch':'2.12.1','backend':'cpu-arm64-eager','cases':cases}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.candidate.rs'
            emitted=write_rust_candidate(path,expressions,width=2,vocab=4,context=8)
            self.assertEqual(emitted['finalParity'],False)
            source=path.read_text()
            self.assertNotIn('let ',source);self.assertNotIn('R16',source)
            self.assertNotIn('checkpoint',source);self.assertNotIn('fn sqrt',source)
            result=validate_rust_candidate(path,reference,width=2,vocab=4,context=8)
            self.assertEqual(result['comparisons'],160);self.assertEqual(result['mismatches'],0)
            reference['cases'][0]['logitF64Bits'][-1][0]=word(42.)
            with self.assertRaisesRegex(ValueError,'bit parity failed'):
                validate_rust_candidate(path,reference,width=2,vocab=4,context=8)

    def test_reference_target_or_missing_lengths_cannot_admit_a_candidate(self):
        for reference in [{'width':2,'vocab':4,'context':8,'torch':'2.11.0','backend':'cpu-arm64-eager'},
                          {'width':2,'vocab':4,'context':8,'torch':'2.12.1','backend':'cpu-arm64-eager','cases':[]}]:
            with self.assertRaises(ValueError):validate_rust_candidate('missing.rs',reference,width=2,vocab=4,context=8)

    @unittest.skipUnless(shutil.which('rustc'),'Rust compiler required')
    def test_expanded_half_rounding_runs_as_rust_without_conversion_helpers(self):
        from fractions import Fraction as F
        from direct_sympy_strings import StringCompiler,Domain
        from direct_sympy_conversions import ConversionSession
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False)},input_dtype='f16')
        expression=session.close('R16(R32(X1/3.0))')
        cases=[]
        for bits in [0,0x8000,1,0x8001,0x03ff,0x0400,0x3c00,0xbc00,0x3555,0x7bff,0xfbff]:
            value=struct.unpack('<e',struct.pack('<H',bits))[0]
            single=struct.unpack('<f',struct.pack('<f',value/3.0))[0]
            expected=struct.unpack('<e',struct.pack('<e',single))[0]
            cases.append({'inputBits':[[bits]],'logitF64Bits':[[word(expected)]]})
        reference={'width':1,'vocab':1,'context':1,'torch':'2.12.1','backend':'cpu-arm64-eager','cases':cases}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'expanded.rs'
            write_rust_candidate(path,{1:[expression]},width=1,vocab=1,context=1)
            result=validate_rust_candidate(path,reference,width=1,vocab=1,context=1)
            self.assertEqual(result['comparisons'],len(cases))


if __name__=='__main__':unittest.main()

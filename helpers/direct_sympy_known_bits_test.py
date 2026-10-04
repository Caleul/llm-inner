"""Unsigned mask certificates, including overflow and opaque selected roots."""
import ast
import random
import struct
import unittest
from unittest.mock import patch

from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_words import MASK,known_bits,simplify_words
from fractions import Fraction


FUNCTIONS={'Bits64':lambda x:struct.unpack('Q',struct.pack('d',x))[0],
    'U64And':lambda a,b:a&b,'U64Or':lambda a,b:a|b,
    'U64Add':lambda a,b:(a+b)&MASK,'U64Mul':lambda a,b:(a*b)&MASK,
    'U64Shr':lambda a,b:a>>b}


class KnownBitTests(unittest.TestCase):
    def test_rewrites_and_masks_cover_all_16bit_words_and_64bit_overflow(self):
        sources=[
            'U64And(U64Mul(Bits64(X1),4096),18446744073709547520)',
            'U64And(U64Add(U64Mul(Bits64(X1),16),U64Mul(Bits64(X2),16)),15)',
            'U64Or(U64Or(Bits64(X1),1024),1024)',
            'U64And(U64Or(U64And(Bits64(X1),18446744073709550591),1024),1024)']
        programs=[]
        for text in sources:
            result=simplify_words(text,StringCompiler(),{})
            self.assertLess(len(result),len(text))
            zero,one=known_bits(syntax(text))
            self.assertEqual(zero&one,0)
            programs.append((compile(text,'original','eval'),compile(result,'simplified','eval'),zero,one))
        rng=random.Random(871)
        words=list(range(65536))+[MASK,0,1<<63,(1<<63)-1]
        words.extend(rng.getrandbits(64) for _ in range(4096))
        comparisons=0
        for bits in words:
            inputs={'X1':struct.unpack('d',struct.pack('Q',bits))[0],
                'X2':struct.unpack('d',struct.pack('Q',rng.getrandbits(64)))[0]}
            for original,candidate,zero,one in programs:
                value=eval(original,{'__builtins__':{},**FUNCTIONS},inputs)
                self.assertEqual(value,eval(candidate,{'__builtins__':{},**FUNCTIONS},inputs))
                self.assertEqual(value&zero,0)
                self.assertEqual(value&one,one)
                comparisons+=1
        print(f'Known word mask parity: comparisons={comparisons} mismatches=0')

    def test_reduced_values_still_pass_mandatory_cas_stabilization(self):
        compiler=StringCompiler()
        with patch.object(compiler,'stabilize',wraps=compiler.stabilize) as stabilize:
            result=simplify_words('U64And(U64Mul(Bits64(X1),16),15)',compiler,{})
        self.assertEqual(result,'0');self.assertGreater(stabilize.call_count,0)
        self.assertTrue(all(event[2:4]==('factor','simplify') for event in compiler.events))

    def test_completed_word_payload_proof_is_local_to_its_numeric_context(self):
        compiler=StringCompiler();domains={'X1':Domain(Fraction(-1),Fraction(1),-24)}
        text='Float64(U64Mul(Bits64(X1),4096))'
        compiler.expression_size=lambda expression:len(expression)
        compiler.register_completed_region(text,domains,word_closed=True)
        self.assertTrue(compiler.copy_completed_word_root(text,'CompileValue0()',domains))
        source='U64And(Bits64(CompileValue0()),18446744073709547520)'
        reduced=simplify_words(source,compiler,domains)
        self.assertEqual(ast.dump(syntax(reduced)),ast.dump(syntax('U64Mul(Bits64(X1),4096)')))
        self.assertEqual(ast.dump(syntax(simplify_words(source,compiler,{}))),ast.dump(syntax(source)))

    def test_unproved_words_and_floating_arithmetic_are_not_opened(self):
        for text in ['U64And(X1,15)','U64And(U64Mul(Bits64(X1),16.0),15)',
                     'Bits64(X1*4096.0)','R32(X1*4096.0+X2*4096.0)']:
            self.assertEqual(known_bits(syntax(text)),(0,0))
            self.assertEqual(ast.dump(syntax(simplify_words(text,StringCompiler(),{}))),ast.dump(syntax(text)))


if __name__=='__main__':unittest.main()

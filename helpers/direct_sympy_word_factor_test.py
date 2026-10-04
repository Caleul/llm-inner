"""Unsigned modular CAS proofs; never reassociate floating reductions."""
import ast
from fractions import Fraction
import random
import struct
import unittest
from unittest.mock import patch

import direct_sympy_words as words
from direct_sympy_strings import Domain,StringCompiler,syntax


FUNCTIONS={'Bits64':lambda x:struct.unpack('Q',struct.pack('d',x))[0],
    'U64Add':lambda a,b:(a+b)&words.MASK,
    'U64Mul':lambda a,b:(a*b)&words.MASK}


class WordFactorTests(unittest.TestCase):
    def test_cas_factor_and_simplify_reduce_shared_word_after_substitution(self):
        compiler=StringCompiler()
        with patch.object(words.sp,'factor',wraps=words.sp.factor) as factor,patch.object(words.sp,'simplify',wraps=words.sp.simplify) as simplify:
            result=compiler.substitute('U64Add(U64Mul(X9, 3), U64Mul(X9, 7))','X9','Bits64(X1)',{})
        self.assertEqual(result,'U64Mul(10, Bits64(X1))')
        self.assertTrue(compiler.word_factor_events)
        self.assertGreaterEqual(factor.call_count,2)
        self.assertGreaterEqual(simplify.call_count,1)
        self.assertEqual(compiler.stabilize(result,{}),result)
        self.assertNotIn('WordFactor',result)
        # Exact structural comparison does not allocate printed copies of
        # opaque floating subtrees. Nested rewrites must invalidate parents.
        nested='U64Add(U64Mul(U64Add(U64Mul(Bits64(X1),3),U64Mul(Bits64(X1),7)),3),U64Mul(U64Mul(10,Bits64(X1)),7))'
        with patch.object(words.ast,'dump',side_effect=AssertionError('Printed subtree comparison')):
            node,changes=words.factor_word_polynomials(syntax(nested))
        self.assertGreater(changes,0)
        self.assertEqual(ast.unparse(node),'U64Mul(100, Bits64(X1))')

    def test_repeated_words_and_overflow_match_all_f16_patterns_and_f64_boundaries(self):
        sources=[
            'U64Add(U64Mul(Bits64(X1), 3), U64Mul(Bits64(X1), 7))',
            'U64Add(U64Mul(Bits64(X1), 18446744073709551615), Bits64(X1))',
            'U64Add(U64Mul(Bits64(X1), Bits64(X2)), U64Mul(Bits64(X1), Bits64(X3)))',
            'U64Add(U64Mul(Bits64(X1), Bits64(X1)), U64Mul(Bits64(X1), Bits64(X2)))',
        ]
        programs=[]
        for source in sources:
            result=StringCompiler().stabilize(source,{})
            self.assertLess(len(result),len(source))
            programs.append((compile(source,'original','eval'),compile(result,'factored','eval')))
        values=[float(struct.unpack('e',struct.pack('H',bits))[0]) for bits in range(65536)]
        bits=[0,1,2**52-1,2**52,0x3ff0000000000000,0x7fefffffffffffff,0x7ff0000000000000,0x7ff8000000000001,2**63,words.MASK]
        rng=random.Random(917)
        bits.extend(rng.getrandbits(64) for _ in range(4096))
        values.extend(struct.unpack('d',struct.pack('Q',word))[0] for word in bits)
        comparisons=0
        for value in values:
            inputs={'X1':value,'X2':struct.unpack('d',struct.pack('Q',rng.getrandbits(64)))[0],'X3':struct.unpack('d',struct.pack('Q',rng.getrandbits(64)))[0]}
            for original,factored in programs:
                self.assertEqual(eval(original,{'__builtins__':{},**FUNCTIONS},inputs),eval(factored,{'__builtins__':{},**FUNCTIONS},inputs))
                comparisons+=1
        print(f'Modular word factor parity: comparisons={comparisons} mismatches=0')

    def test_floating_order_and_unproved_words_are_barriers(self):
        for source in ('R32(X1 * X1 + X1 * X2)',
                       'U64Add(U64Mul(X1, 3), U64Mul(X1, 7))',
                       'U64Add(U64Mul(Bits64(-0.0), 3), U64Mul(Bits64(0.0), 7))',
                       'U64Add(U64Mul(Bits64(X1), 3.0), U64Mul(Bits64(X1), 7))'):
            compiler=StringCompiler();result=compiler.stabilize(source,{})
            self.assertEqual(ast.dump(syntax(result)),ast.dump(syntax(source)))
            self.assertFalse(compiler.word_factor_events)

    def test_factoring_search_rejects_exponential_polynomial_before_cas(self):
        expression='Bits64(X1)'
        for i in range(2,18):expression=f'U64Mul({expression}, U64Add(Bits64(X1), Bits64(X{i})))'
        node=syntax(expression)
        with patch.object(words.sp,'factor',side_effect=AssertionError('Unbounded expansion sent to CAS')):
            self.assertIs(words.factor_word_polynomial(node),node)

    def test_branch_conditions_and_bodies_retain_their_own_context(self):
        domain={'X1':Domain(Fraction(-1),Fraction(1),-24,False)}
        compiler=StringCompiler()
        source='Piecewise((U64Add(U64Mul(Bits64(X1), 3), U64Mul(Bits64(X1), 7)), X1 > 0), (U64Add(U64Mul(Bits64(X1), 5), U64Mul(Bits64(X1), 6)), True))'
        result=compiler.stabilize(source,domain)
        self.assertIn('U64Mul(10, Bits64(X1))',result)
        self.assertIn('U64Mul(11, Bits64(X1))',result)
        self.assertIn('X1 > 0',result)
        self.assertTrue(any(event[-1]=='branch-contexts' for event in compiler.events))
        guard=compiler.stabilize('Piecewise((1, U64Add(U64Mul(Bits64(X1), 3), U64Mul(Bits64(X1), 7)) < 100), (2, True))',domain)
        self.assertIn('U64Mul(10, Bits64(X1)) < 100',guard)


if __name__=='__main__':unittest.main()

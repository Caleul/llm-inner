import struct
import unittest
from fractions import Fraction as F

from direct_sympy_checkpoint import f32
from direct_sympy_operators import OperatorBlock
from direct_sympy_parallel import ParallelBudget
from direct_sympy_scalar import ScalarProgram,compose_scalar_operators
from direct_sympy_strings import Domain,StringCompiler


class ScalarCompositionTests(unittest.TestCase):
    def test_pairwise_locals_preserve_order_branches_and_signed_zeros(self):
        blocks=(OperatorBlock(('X1',),('R32(X1 + 16777216.0)',)),
            OperatorBlock(('X2',),('R32(X2 + 1.0)','X2')),
            OperatorBlock(('X3','X4'),('Piecewise((-0.0, X3 == X4), (R32(X3 - X4), True))',)))
        domains={'X1':Domain(F(-65504),F(65504),-24,False)}
        compiler=StringCompiler(max_characters=1024**2)
        result,stats=compose_scalar_operators(blocks,compiler,domains,ParallelBudget(2,2*1024**3))
        self.assertEqual(stats['compositionLevels'],2)
        self.assertEqual(stats['scalarPreparations'],3)
        self.assertEqual(stats['scalarPairMerges'],2)
        self.assertEqual(len(result.definitions),3)
        self.assertGreater(stats['factorSimplifyEvents'],0)
        for x in [-65504.,-0.,0.,1.,65504.]:
            a=f32(x+16777216.);b=f32(a+1.)
            expected=-0. if a==b else f32(b-a)
            self.assertEqual(struct.pack('d',result.evaluate((x,),{'R32':f32})[0]),struct.pack('d',expected))

    def test_repeated_producer_is_defined_once_not_copied(self):
        leaf='R32('+ ' + '.join(['X1']*100)+')'
        blocks=(OperatorBlock(('X1',),(leaf,)),OperatorBlock(('X2',),('R32('+ ' + '.join(['X2']*100)+')',)))
        result,stats=compose_scalar_operators(blocks,StringCompiler(max_characters=1024**2),
            {'X1':Domain(F(-1),F(1),-24,False)},ParallelBudget(2,2*1024**3))
        self.assertEqual(len(result.definitions),2)
        self.assertLess(stats['scalarCharacters'],2000)
        self.assertEqual(result.evaluate((1.,),{'R32':f32}),(10000.,))

    def test_unresolved_forward_or_duplicate_dependencies_are_rejected(self):
        with self.assertRaises(ValueError):ScalarProgram(('X1',),(('X2','X3 + 1'),),('X2',)).validate()
        with self.assertRaises(ValueError):ScalarProgram(('X1',),(('X1','X1 + 1'),),('X1',)).validate()


if __name__=='__main__':unittest.main()

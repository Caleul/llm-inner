"""Diagnostic sharing preserves literal arithmetic, lazy branches and inputs."""
from fractions import Fraction as F
import struct
import unittest

from direct_sympy_architecture_test import program
from direct_sympy_checkpoint import f32
from direct_sympy_strings import Domain,StringCompiler
from direct_sympy_working_program import WorkingProgram


class WorkingProgramTests(unittest.TestCase):
    def compiler(self,leaf):
        compiler=StringCompiler(max_characters=8*1024**2)
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        compiler.register_completed_region(leaf,domains)
        return compiler

    def test_lazy_regions_do_not_evaluate_an_unreachable_division_and_keep_zero_signs(self):
        leaf='R32(1.0 / X1)'
        text=f'Piecewise((-0.0, X1 == 0.0), ({leaf}, True))'
        functions={'R32':f32}
        compiled=WorkingProgram(text,self.compiler(leaf),functions)
        for value in [-1.,-0.,0.,1.]:
            expected=eval(program(text),{'__builtins__':{},**functions},{'X1':value})
            self.assertEqual(struct.pack('>d',compiled({'X1':value})),struct.pack('>d',expected))

    def test_repeated_literals_preserve_rounding_order_and_cache_is_reset_for_each_input(self):
        leaf='R32(X1 + 16777216.0)'
        text=f'R32(R32({leaf} + 1.0) - {leaf})'
        compiled=WorkingProgram(text,self.compiler(leaf),{'R32':f32})
        for value in [-1.,-0.,0.,2**-24,1.]:
            expected=eval(program(text),{'__builtins__':{},'R32':f32},{'X1':value})
            self.assertEqual(struct.pack('>d',compiled({'X1':value})),struct.pack('>d',expected))

    def test_wide_repetition_parses_each_literal_once_and_retains_exact_file_text(self):
        leaf='R32('+ ' + '.join(['X1']*100)+')'
        text='R32(Add('+','.join([leaf]*1000)+'))'
        functions={'R32':f32,'Add':lambda *terms:sum(terms)}
        compiler=self.compiler(leaf)
        compiled=WorkingProgram(text,compiler,functions)
        self.assertLess(compiled.parsed_characters,len(text)//4)
        self.assertEqual(len(compiled.programs),2)
        for value in [-1.,-0.,0.,2**-24,1.]:
            expected=eval(program(text),{'__builtins__':{},**functions},{'X1':value})
            self.assertEqual(struct.pack('>d',compiled({'X1':value})),struct.pack('>d',expected))


if __name__=='__main__':unittest.main()

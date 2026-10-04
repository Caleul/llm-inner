import ast
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys
import random

from direct_sympy_conversions import FiniteSource,lower_finite_conversion,ConversionSession,binary_exponent
from direct_sympy_strings import StringCompiler,syntax,Domain
from fractions import Fraction as F
from direct_sympy_words import simplify_words
from direct_sympy_signatures import StructuralSignatures
from direct_sympy_tandem import supported as tandem_supported,lower_tandem


def cpp(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Constant):
        if type(node.value) is bool:return "true" if node.value else "false"
        if type(node.value) is int:return "UINT64_C("+str(node.value)+")"
        return repr(node.value)
    if isinstance(node,ast.UnaryOp):
        if isinstance(node.op,ast.Not):return "(!("+cpp(node.operand)+"))"
        return "(-double("+cpp(node.operand)+"))" if isinstance(node.op,ast.USub) else cpp(node.operand)
    if isinstance(node,ast.BinOp):
        if isinstance(node.op,ast.Pow):return "std::pow(double("+cpp(node.left)+"), double("+cpp(node.right)+"))"
        op={ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/"}[type(node.op)]
        return "(double("+cpp(node.left)+") "+op+" double("+cpp(node.right)+"))"
    if isinstance(node,ast.Compare):
        op={ast.Lt:"<",ast.LtE:"<=",ast.Gt:">",ast.GtE:">=",ast.Eq:"==",ast.NotEq:"!="}[type(node.ops[0])]
        return "("+cpp(node.left)+" "+op+" "+cpp(node.comparators[0])+")"
    if isinstance(node,ast.BoolOp):return "("+(" && " if isinstance(node.op,ast.And) else " || ").join(cpp(x) for x in node.values)+")"
    if isinstance(node,ast.Call):
        name=node.func.id
        if name=="Not":return "(!("+cpp(node.args[0])+"))"
        if name in ("And","Or"):return "("+(" && " if name=="And" else " || ").join(cpp(x) for x in node.args)+")"
        if name=="Piecewise":
            result="UINT64_C(0)"
            for pair in reversed(node.args):result="("+cpp(pair.elts[1])+" ? "+cpp(pair.elts[0])+" : "+result+")"
            return result
        args=[cpp(x) for x in node.args]
        if name=="F64FromU64":return "double(uint64_t("+args[0]+"))"
        if name=="U64FromF64":return "uint64_t(double("+args[0]+"))"
        if name=="Bits64":return "word<uint64_t>(double("+args[0]+"))"
        if name=="Float64":return "word<double>(uint64_t("+args[0]+"))"
        if name=="sqrt":return "std::sqrt(double("+args[0]+"))"
        op={"U64And":"&","U64Or":"|","U64Shr":">>","U64Add":"+","U64Mul":"*"}[name]
        return "(uint64_t("+args[0]+") "+op+" uint64_t("+args[1]+"))"
    raise AssertionError(ast.dump(node))


class ConversionStringTests(unittest.TestCase):
    def test_certified_frontier_composition_closes_each_arm_before_restoring_operands(self):
        import torch
        domains={'X1':Domain(-F(1,64),F(1,64),-24,False),'X2':Domain(-F(1,64),F(1,64),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        activation=session.close('Silu16(X1)');up=session.close('R16(X2/3.0)')
        original_bounds=session.closed_literals[activation][0]
        template='R16(R32(X999999998 * X999999999))';bindings={'X999999998':activation,'X999999999':up}
        with patch.object(session,'close_frontier_candidates',side_effect=lambda compact,baseline,*_:baseline):baseline=session.compose_closed(template,bindings)
        result=session.compose_closed(template,bindings)
        self.assertLess(len(result),len(baseline))
        self.assertTrue(any(event[-1] for event in session.frontier_events))
        self.assertEqual(session.branch_depth,0)
        self.assertEqual(session.value_kind(syntax(result)),'half')
        self.assertEqual(session.closed_literals[activation][0],original_bounds)
        for name in ('CASNumericRegion','CASStableRegion','R16(','R32(','Silu16('):self.assertNotIn(name,result)
        pairs=[];inputs=[]
        for bits in range(65536):
            value=struct.unpack('e',struct.pack('H',bits))[0]
            if abs(value)<=1/64:pairs.append(bits);inputs.append(value)
        gold=torch.nn.functional.silu(torch.tensor(inputs,dtype=torch.float16)).view(torch.int16).tolist()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'gold.bin';data.write_bytes(b''.join(struct.pack('HH',a,b&65535) for a,b in zip(pairs,gold)))
            source=root/'frontier.cpp';binary=root/'frontier'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double candidate(double X1,double X2){return '+cpp(syntax(result))+';}\n'+'double baseline(double X1,double X2){return '+cpp(syntax(baseline))+';}\n'+'''int main(int argc,char**argv){FILE*f=std::fopen(argv[1],"rb");uint16_t pair[2];unsigned cases=0,mismatches=0;while(std::fread(pair,2,2,f)==2){double x=word<_Float16>(pair[0]),a=word<_Float16>(pair[1]);for(double y:{-0x1p-6,-0x1p-14,-0x1p-24,-0.0,0.0,0x1p-24,0x1p-14,0x1p-6}){double u=double(_Float16(y/3.0)),expected=double(_Float16(float(a*u)));uint64_t got=word<uint64_t>(candidate(x,y));mismatches+=got!=word<uint64_t>(expected);mismatches+=got!=word<uint64_t>(baseline(x,y));cases++;}}std::printf("Certified frontier composition parity: cases=%u mismatches=%u characters=%u->%u\\n",cases,mismatches,'''+str(len(baseline))+','+str(len(result))+''');return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary),str(data)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=147472 mismatches=0',run.stdout);print(run.stdout,end='')

    def test_conversion_closure_owns_ordered_arm_bounds_and_native_parity(self):
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        expression='Piecewise((R16(X1/3.0), X1 < -0.0001220703125), (R16(X1/3.0), X1 < 0.0001220703125), (R16(X1/3.0), True))'
        with patch('direct_sympy_conversions.lower_finite_conversion',wraps=lower_finite_conversion) as lower:
            result=session.close(expression)
        certificates=[call.args[2] for call in lower.call_args_list]
        self.assertEqual(len(certificates),3)
        self.assertLess(max(abs(certificates[1].minimum),abs(certificates[1].maximum)),2**-14)
        self.assertLess(certificates[0].maximum,0)
        self.assertGreater(certificates[2].minimum,0)
        self.assertEqual(dict(session.domains),domains)
        self.assertEqual(session.bounds(syntax('X1')).minimum,-1)
        self.assertEqual(session.branch_depth,0)
        self.assertNotIn('R16(',result)
        self.assertEqual(result.count('4544132024016830464'),2)
        # A later unrelated conversion must retain its whole-input enclosure.
        with patch('direct_sympy_conversions.lower_finite_conversion',wraps=lower_finite_conversion) as lower:
            session.close('R16(X1/3.0)')
        self.assertLess(lower.call_args_list[0].args[2].minimum,-0.3)
        with patch('direct_sympy_conversions.lower_finite_conversion',wraps=lower_finite_conversion) as lower:
            magnitude=session.close('Piecewise((R16(X1/3.0), U64And(Bits64(X1), 9223372036854775807) < 4548635623644200960), (R16(X1/3.0), True))')
        self.assertLess(max(abs(lower.call_args_list[0].args[2].minimum),abs(lower.call_args_list[0].args[2].maximum)),2**-14)
        self.assertLess(lower.call_args_list[1].args[2].minimum,-0.3)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'branches.cpp';binary=root/'branches'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double candidate(double X1){return '+cpp(syntax(result))+';}\n'+'double magnitude(double X1){return '+cpp(syntax(magnitude))+';}\n'+'''int main(){unsigned cases=0,mismatches=0;for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));if(x < -1 || x > 1)continue;double expected=double(_Float16(x/3.0));mismatches+=word<uint64_t>(candidate(x))!=word<uint64_t>(expected);mismatches+=word<uint64_t>(magnitude(x))!=word<uint64_t>(expected);cases++;}std::printf("Scoped conversion parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=30722 mismatches=0',run.stdout);print(run.stdout,end='')

    def test_magnitude_proofs_are_scoped_and_disconnected_complements_stay_conservative(self):
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(-1),F(1),-24,False)},input_dtype='f16')
        node=syntax('CASNumericRegion42()');key=session.key(node)
        session.completed[key]=FiniteSource(-1,1,-24)
        session.half_values.add(key)
        guard=syntax('U64And(Bits64(CASNumericRegion42()), 9223372036854775807) < 4544132024016830464')
        own=session.magnitude_guard_bounds(guard,True,{})
        self.assertEqual((own[key].minimum,own[key].maximum),(-math.nextafter(2**-14,0),math.nextafter(2**-14,0)))
        self.assertEqual(session.magnitude_guard_bounds(guard,False,{}),{})
        with self.assertRaisesRegex(RuntimeError,'scope'):
            with session.branch_context(session.domains,own,()):
                self.assertEqual(session.bounds(node).maximum,math.nextafter(2**-14,0))
                session.completed[session.key(syntax('X1/3.0'))]=FiniteSource(0,0,0)
                session.remember_closed_literal('0.0')
                raise RuntimeError('scope')
        self.assertEqual(session.bounds(node).maximum,1)
        self.assertNotIn('0.0',session.closed_literals)
        self.assertGreater(session.bounds(syntax('X1/3.0')).maximum,0.3)
        self.assertEqual(session.branch_depth,0)
        inclusive=syntax('U64And(Bits64(CASNumericRegion42()), 9223372036854775807) <= 4544132024016830464')
        for bounds,sign in ((FiniteSource(0,1,-24),1),(FiniteSource(-1,0,-24),-1)):
            session.completed[key]=bounds
            complement=session.magnitude_guard_bounds(inclusive,False,{})[key]
            endpoint=complement.minimum if sign==1 else -complement.maximum
            self.assertEqual(endpoint,math.nextafter(2**-14,math.inf))
        session.completed[key]=FiniteSource(-1,1,-24)
        self.assertEqual(session.magnitude_guard_bounds(syntax('U64And(Bits64(X1), 1) < 2'),True,{}),{})

    def test_branch_local_half_cell_cannot_eliminate_the_sibling_update(self):
        domains={'X1':Domain(F(-2),F(2),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        result=session.close('Piecewise((R16(X1 + 0.000000059604644775390625), X1 >= 1.0), (R16(X1 + 0.000000059604644775390625), True))')
        tree=syntax(result)
        self.assertEqual(ast.unparse(tree.args[0].elts[0]),'X1')
        self.assertNotEqual(ast.unparse(tree.args[1].elts[0]),'X1')
        self.assertFalse(session.half_update_is_invisible(syntax('X1'),syntax('0.000000059604644775390625')))
        self.assertGreater(session.arithmetic_eliminated,0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'update.cpp';binary=root/'update'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double candidate(double X1){return '+cpp(syntax(result))+';}\n'+'''int main(){unsigned cases=0,mismatches=0;for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));if(x < -2 || x > 2)continue;double expected=double(_Float16(x+0x1p-24));mismatches+=word<uint64_t>(candidate(x))!=word<uint64_t>(expected);cases++;}std::printf("Scoped Half update parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=32770 mismatches=0',run.stdout);print(run.stdout,end='')

    def test_tandem_sign_uses_only_a_proven_positive_factor_and_preserves_native_bits(self):
        domains={"X1":Domain(F(-65504),F(65504),-24,False),"X2":Domain(F(2)**-100,F(1024),-100,False)}
        def session():
            s=ConversionSession(StringCompiler(),domains,input_dtype="f16")
            s.half_values.discard(s.key(syntax("X2")))
            s.f32_values.add(s.key(syntax("X2")))
            return s
        optimized=session();control=session()
        expression="R16(R32(X1*X2))"
        result=optimized.close(expression)
        with patch.object(control,"same_sign_operand",side_effect=lambda node:node):general=control.close(expression)
        self.assertLess(result.count("X2"),general.count("X2"))
        self.assertEqual(ast.unparse(optimized.same_sign_operand(syntax("X1*X2"))),"X1")
        self.assertEqual(ast.unparse(optimized.same_sign_operand(syntax("X2*X1"))),"X1")
        self.assertEqual(ast.unparse(optimized.same_sign_operand(syntax("X1/X2"))),"X1")
        unknown=ConversionSession(StringCompiler(),{"X1":domains["X1"],"X2":Domain(F(0),F(1024),-100,False)},input_dtype="f16")
        self.assertEqual(ast.unparse(unknown.same_sign_operand(syntax("X1*X2"))),"X1 * X2")
        signed=ConversionSession(StringCompiler(),{"X1":domains["X1"],"X2":Domain(F(-1),F(1),-24,False)},input_dtype="f16")
        self.assertEqual(ast.unparse(signed.same_sign_operand(syntax("X1*X2"))),"X1 * X2")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"sign.cpp";binary=root/"sign"
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double candidate(double X1,double X2){return '+cpp(syntax(result))+';}\n'+'double control(double X1,double X2){return '+cpp(syntax(general))+';}\n'+'''int main(){unsigned cases=0,mismatches=0;for(uint32_t bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));for(float factor:{0x1p-100f,0x1p-24f,0x1p-14f,0.000001f,0.75f,1.0f,1.25f,1024.0f}){double expected=double(_Float16(float(x*double(factor))));uint64_t got=word<uint64_t>(candidate(x,double(factor)));mismatches+=got!=word<uint64_t>(expected);mismatches+=got!=word<uint64_t>(control(x,double(factor)));cases++;}}std::printf("Positive-factor sign parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=507904 mismatches=0',run.stdout);print(run.stdout,end='')

    @patch.object(ConversionSession,'close_frontier_candidates',lambda self,compact,baseline,*args:baseline)
    def test_closed_envelope_restores_exact_proofs_without_parsing_expanded_result(self):
        domains={"X1":Domain(F(-1),F(1),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype="f16")
        first=session.close("R16(R32(X1+X1/2.0))")
        with patch("direct_sympy_conversions.syntax",wraps=syntax) as parse:
            result=session.close("R16(R32(("+first+") * 0.75))")
        self.assertLess(max(len(call.args[0]) for call in parse.call_args_list),len(result))
        actual=syntax(result)
        self.assertEqual(session.closed_literal_keys[result],session.key(actual))
        self.assertEqual(session.closed_literals[result],(session.bounds(actual),"half",session.no_negative_zero(actual)))
        self.assertTrue(session.closed_literal_pure[result])
        self.assertIn(first,result)
        self.assertIn("X1",result)
        for placeholder in ("CASNumericRegion","CASRegion","R16(","R32("):
            self.assertNotIn(placeholder,result)

    def test_readonly_queries_compose_literal_keys_without_reopening_closed_subtrees(self):
        domains={"X1":Domain(F(-1),F(1),-24,False),"X2":Domain(F(-8),F(8),-24,False)}
        compiler=StringCompiler();session=ConversionSession(compiler,domains,input_dtype="f16")
        first=session.close("R16(R32(X1+X1/2.0))")
        second=session.close("R16(X2*0.5)")
        for text in (first,second):compiler.register_completed_region(text,domains,word_closed=True)
        expression="R32(("+first+") * 0.5)";original=syntax(expression)
        expected_key=session.key(original);expected_bounds=session.bounds(original)
        with patch("direct_sympy_conversions.syntax",wraps=syntax) as parse:
            query,keys=session.analyze_expression(expression)
        self.assertLess(max(len(c.args[0]) for c in parse.call_args_list),len(first))
        self.assertEqual(keys[query],expected_key)
        self.assertEqual(session.bounds(query),expected_bounds)
        self.assertEqual(session.value_kind(query),"f32")
        old,old_keys=session.analyze_expression(first)
        other,other_keys=session.analyze_expression(second)
        self.assertNotEqual(session.key(old),session.key(other))
        self.assertEqual(session.bounds(old),session.closed_literals[first][0])
        self.assertEqual(session.bounds(other),session.closed_literals[second][0])
        self.assertEqual(old_keys[old],session.key(syntax(first)))
        self.assertEqual(other_keys[other],session.key(syntax(second)))
        # A fundamental-input guard changes the context, so this query
        # deliberately reopens the literal instead of hiding its branches.
        guarded="Piecewise((("+first+"), X1 > 0), (0.0, True))"
        reopened,mapping=session.analyze_expression(guarded)
        self.assertEqual(mapping,{})
        self.assertNotIn("CASNumericRegion",ast.unparse(reopened))
        final=compiler.substitute("(X999999999 * 0.5)","X999999999",first,domains)
        self.assertNotIn("CASNumericRegion",final)

    def test_integer_offset_composition_preserves_half_cells_and_rejects_inexact_words(self):
        domains={"X1":Domain(F(-262144),F(262144),-126,False)}
        compiler=StringCompiler()
        expression=lower_tandem("X1",FiniteSource(-262144,262144,-126),compiler,domains,integer_word_exact=True)
        self.assertEqual(expression.count("X1"),6)  # Includes explicit overflow guards/arms.
        known=ConversionSession(StringCompiler(),{"X1":Domain(F(-65504),F(65504),-24,False),"X2":Domain(F(-2),F(2),-149,False)},input_dtype="f16")
        known.half_values.discard(known.key(syntax("X2")));known.f32_values.add(known.key(syntax("X2")))
        self.assertTrue(known.encoded_word_is_exact_integer(syntax("X1*X2")))
        self.assertTrue(known.encoded_word_is_exact_integer(syntax("X1+X1")))
        narrow=ConversionSession(StringCompiler(),{'X1':Domain(F(-3,128),F(3,128),-24,False)},input_dtype='f16')
        self.assertTrue(narrow.encoded_word_is_exact_integer(syntax('X1*(0.5+X1*0.25)')))
        untyped=ConversionSession(StringCompiler(),narrow.domains)
        self.assertFalse(untyped.encoded_word_is_exact_integer(syntax('X1*(0.5+X1*0.25)')))
        opaque=syntax('opaque(X1)*opaque(X1)')
        narrow.completed[narrow.key(opaque)]=FiniteSource(-1,1,-20)
        self.assertTrue(narrow.encoded_word_is_exact_integer(opaque))
        unknown=ConversionSession(StringCompiler(),domains)
        self.assertFalse(unknown.encoded_word_is_exact_integer(syntax("X1")))
        unknown.f32_values.add(unknown.key(syntax("X1")))
        self.assertTrue(unknown.encoded_word_is_exact_integer(syntax("X1")))
        # Precision admission is scoped to normal words. A signed F64
        # subnormal must retain its target-subnormal kernel, where interpreting
        # the signed raw word as a numeric F64 could lose low integer bits.
        tiny=2.0**-1074
        tiny_domains={"X1":Domain(F(-tiny),F(tiny),-1074,False)}
        tiny_kernel=lower_finite_conversion("X1","R32",FiniteSource(-tiny,tiny,-1074),StringCompiler(),tiny_domains,integer_word_exact=True)
        self.assertNotIn("F64FromU64",tiny_kernel)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"offset.work.expr";path.write_text(expression+"\n")
            source=Path(directory)/"offset.cpp";binary=Path(directory)/"offset"
            source.write_text("#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n"+"double compiled(double X1){return "+cpp(syntax(path.read_text()))+";}\n"+'''
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0,integerLoss=0;
auto check=[&](double x){uint64_t raw=word<uint64_t>(x);integerLoss+=uint64_t(double(raw))!=raw;double expected=static_cast<_Float16>(float(x));double actual=compiled(x);mismatches+=word<uint64_t>(actual)!=word<uint64_t>(expected);cases++;};
for(unsigned h=0;h<0x7c00;h++){double low=word<_Float16>(uint16_t(h)),high=h==0x7bff?65536:double(word<_Float16>(uint16_t(h+1)));double midpoint=(low+high)/2;
int e=std::ilogb(midpoint);double fuzz=std::ldexp(1.0,e-24);uint64_t center=word<uint64_t>(midpoint);
for(int side:{-1,0,1}){uint64_t edge=word<uint64_t>(midpoint+side*fuzz);for(int delta:{-4096,-2048,0,2048,4096}){double x=word<double>(uint64_t(int64_t(edge)+delta));check(x);check(-x);}}
}check(0.0);check(-0.0);uint64_t random=123456789;
for(unsigned j=0;j<100000;j++){random^=random<<13;random^=random>>7;random^=random<<17;unsigned exponent=963+(j%78);uint64_t raw=(uint64_t(exponent)<<52)|(random&UINT64_C(4503599627368448));double x=word<double>(raw);check(x);check(-x);}
std::printf("Integer offset tandem parity: cases=%u mismatches=%u integerCastLoss=%u\\n",cases,mismatches,integerLoss);return mismatches||integerLoss;}
''')
            built=subprocess.run(["clang++","-O3","-ffp-contract=off","-std=c++17",str(source),"-o",str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("mismatches=0 integerCastLoss=0",result.stdout);print(result.stdout,end="")

    def test_half_cell_elision_is_strict_typed_and_native_exact(self):
        domains={"X1":Domain(F(64),F(65504),-4,False),"X2":Domain(F(-1,128),F(1,128),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype="f16")
        for operation in ("+","-"):
            self.assertEqual(session.close("R16(R32(X1 "+operation+" X2))"),"X1")
        self.assertEqual(session.half_cell_radius(syntax("X1")),2**-6)
        session=ConversionSession(StringCompiler(),{"X1":domains["X1"],"X2":Domain(F(-1,64),F(1,64),-24,False)},input_dtype="f16")
        self.assertFalse(session.half_update_is_invisible(syntax("X1"),syntax("X2")))
        session=ConversionSession(StringCompiler(),{"X1":Domain(F(-64),F(64),-24,False),"X2":domains["X2"]},input_dtype="f16")
        self.assertIsNone(session.half_cell_radius(syntax("X1")))
        session=ConversionSession(StringCompiler(),domains,input_dtype="f64")
        self.assertFalse(session.half_update_is_invisible(syntax("X1"),syntax("X2")))
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"cell.cpp";binary=Path(directory)/"cell"
            source.write_text("#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n"+'''int main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));if(std::fabs(x)<64)continue;for(double d:{0.0,-0.0,0x1p-24,-0x1p-24,0x1p-7,-0x1p-7,0x1.ffcp-7,-0x1.ffcp-7}){double sum=static_cast<_Float16>(float(x+d));double difference=static_cast<_Float16>(float(x-d));mismatches+=word<uint64_t>(sum)!=word<uint64_t>(x);mismatches+=word<uint64_t>(difference)!=word<uint64_t>(x);cases+=2;}}std::printf("Half cell dependency parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            build=subprocess.run(["clang++","-O3","-ffp-contract=off","-std=c++17",str(source),"-o",str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn("mismatches=0",result.stdout);print(result.stdout,end="")

    def test_new_envelope_selectors_synchronize_before_literal_restoration(self):
        domains={name:Domain(F(-65504),F(65504),-24,False) for name in ('X1','X2')}
        compiler=StringCompiler();session=ConversionSession(compiler,domains,input_dtype='f16')
        producer=session.close('R32(1.0/(1.0+X1**2))')
        compiler.register_completed_region(producer,domains,word_closed=True)
        result=session.close(f'R16(R32(({producer})*X2)) + R16(R32(({producer})*X2))')
        self.assertEqual(result.count('Piecewise('),1)
        self.assertNotIn('CASNumericRegion',result)
        event=compiler.synchronization_events[-1]
        self.assertEqual(event[-1],'admitted');self.assertLess(event[2],event[0])
        self.assertTrue(all(e[2:4]==('factor','simplify') for e in compiler.events))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expression=root/'envelope.work.expr';expression.write_text(result+'\n')
            source=root/'envelope.cpp';binary=root/'envelope'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double compiled(double X1,double X2){return '+cpp(syntax(expression.read_text()))+';}\n'+'''int main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double X1=word<_Float16>(uint16_t(b));for(uint16_t other:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7bff),uint16_t(0xfbff)}){double X2=word<_Float16>(other);double r=static_cast<_Float16>(float(double(float(1.0/(1.0+X1*X1)))*X2));mismatches+=word<uint64_t>(compiled(X1,X2))!=word<uint64_t>(r+r);cases++;}}std::printf("Compact envelope selector parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            build=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=507904 mismatches=0',run.stdout);print(run.stdout,end='')

    @patch.object(ConversionSession,'close_frontier_candidates',lambda self,compact,baseline,*args:baseline)
    @patch('direct_sympy_silu.quadratic_tandem_source',return_value=None)
    def test_completed_rounding_frontiers_synchronize_inside_activation(self,_square):
        # A conservative grid keeps the F32 frontier observable here.
        # The tighter Half grid is now independently proved exact F32;
        # this test isolates selector synchronization, not grid elision.
        domains={'X1':Domain(-F(1,64),F(1,64),-30,False)}
        compiler=StringCompiler();session=ConversionSession(compiler,domains,input_dtype='f16')
        producer=session.close('R16(R32(X1+X1/2.0))')
        compiler.register_completed_region(producer,domains,word_closed=True)
        with patch.object(session,'selector_views',return_value={}):
            baseline=session.close('Silu16('+producer+')')
        result=session.close('Silu16('+producer+')')
        self.assertLess(len(result),len(baseline))
        self.assertLess(result.count('Piecewise('),baseline.count('Piecewise('))
        self.assertEqual(compiler.synchronization_events[-1][-1],'admitted')
        for name in ('CASNumericRegion','CASStableRegion','Silu16(','R16(','R32('):self.assertNotIn(name,result)
        self.assertEqual(session.value_kind(syntax(result)),'half')
        self.assertTrue(all(e[2:4]==('factor','simplify') for e in compiler.events))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'views.cpp';binary=root/'views'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U v){T r;std::memcpy(&r,&v,sizeof(r));return r;}\n'+
                'double baseline(double X1){return '+cpp(syntax(baseline))+';}\n'+
                'double compiled(double X1){return '+cpp(syntax(result))+';}\n'+
                '''int main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<=0x2400;b++)for(unsigned sign:{0u,0x8000u}){double x=word<_Float16>(uint16_t(b|sign));mismatches+=word<uint64_t>(compiled(x))!=word<uint64_t>(baseline(x));cases++;}std::printf("Completed selector activation parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            build=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=18434 mismatches=0',run.stdout);print(run.stdout,end='')
        leaf=session.close('R32(1.0/(1.0+X1**2))')
        compiler.register_completed_region(leaf,domains,word_closed=True)
        downstream=session.close('R16(R32(('+leaf+')*X1))')
        self.assertIn(leaf,session.selector_literals[downstream][1].values())
        session.closed_literal_keys.pop(leaf)
        protected={'CASNumericRegion900':downstream}
        self.assertEqual(session.selector_views(protected,['CASNumericRegion900']),{})
        self.assertEqual(protected,{'CASNumericRegion900':downstream})

    def test_numeric_envelope_preserves_enclosing_correlation_and_exact_keys(self):
        import direct_sympy_conversions as conversions
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        compiler=StringCompiler();session=ConversionSession(compiler,domains,input_dtype='f16')
        first=session.close('R16(R32(X1+X1/2.0))')
        compiler.register_completed_region(first,domains,word_closed=True)
        product=syntax('X1*('+first+')')
        # first has the sign of X1. Independent interval multiplication
        # cannot recover this valid nonnegative enclosure.
        session.completed[session.key(product)]=FiniteSource(0,1.5001,-48)
        with patch.object(conversions,'lower_finite_conversion',wraps=conversions.lower_finite_conversion) as lower:
            result=session.close('R16('+ast.unparse(product)+')')
        self.assertEqual(lower.call_args.args[2].minimum,0)
        self.assertNotIn('R16(',result);self.assertNotIn('R32(',result)
        self.assertEqual(session.pending,0)
        virtual=syntax('R16(CASNumericRegion999())')
        original=syntax('R16(-0.0)')
        replacements={session.key(virtual.args[0]):session.key(original.args[0])}
        ordinary=session.key(virtual)
        mapped=session.signatures.translated_keys(virtual,replacements)
        self.assertEqual(mapped[virtual],session.key(original))
        self.assertEqual(session.key(virtual),ordinary)
        self.assertNotEqual(mapped[virtual],session.key(syntax('R16(0.0)')))

    def test_synchronization_transfers_only_closed_whole_root_numeric_proofs(self):
        domains={"X1":Domain(F(-1),F(1),-24,False)}
        compiler=StringCompiler();session=ConversionSession(compiler,domains,input_dtype='f16')
        first=session.close('R16(R32(X1+X1/2.0))')
        # Deliberately retain the old opaque frontier to exercise explicit
        # whole-root proof transfer; normal closure now synchronizes it early.
        with patch.object(session,'selector_views',return_value={}):
            original=session.close('R16(R32(('+first+')+('+first+')))')
        replacement=compiler.synchronize(original,domains)
        self.assertNotEqual(original,replacement)
        before=len(session.half_values);bounds=session.bounds(syntax(original))
        session.propagate_closed_identity(original,replacement)
        self.assertEqual(session.value_kind(syntax(replacement)),'half')
        self.assertEqual(session.bounds(syntax(replacement)),bounds)
        self.assertIn(session.key(syntax(replacement)),session.converted_regions)
        self.assertIn(replacement,session.closed_literals)
        self.assertEqual(len(session.half_values),before+1)
        with self.assertRaisesRegex(ValueError,'closed numeric frontier'):
            session.propagate_closed_identity('R16(X1)',replacement)

    def test_synchronized_float_and_word_siblings_preserve_every_half_pattern(self):
        compiler=StringCompiler(dtype='f64');domains={"X1":Domain(F(-65504),F(65504),-24,False)}
        c="U64And(U64Shr(Bits64(X1),17),123456789)<1234567"
        value=f"Piecewise((X1,{c}),(-0.0,True))"
        sources=[value+'+'+value,
            f"Float64(U64Or(U64And(Bits64({value}),9223372036854775808),U64And(Bits64({value}),9223372036854775807)))"]
        outputs=[compiler.synchronize(source,domains) for source in sources]
        self.assertTrue(all(len(after)<len(compiler.stabilize(before,domains)) for before,after in zip(sources,outputs)))
        self.assertTrue(all(text.count('Piecewise(')==1 for text in outputs))
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'sync.cpp';binary=Path(directory)/'sync';functions=[];checks=[]
            for index,(before,after) in enumerate(zip(sources,outputs)):
                functions.extend([f'double before{index}(double X1){{return '+cpp(syntax(before))+';}',f'double after{index}(double X1){{return '+cpp(syntax(after))+';}'])
                checks.append(f'mismatches+=word<uint64_t>(before{index}(x))!=word<uint64_t>(after{index}(x));cases++;')
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'\n'.join(functions)+'\nint main(){unsigned cases=0,mismatches=0;for(unsigned bits=0;bits<65536;bits++){double x=word<_Float16>(uint16_t(bits));'+'\n'.join(checks)+'}\nstd::printf("Synchronized sibling native parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=131072 mismatches=0',result.stdout);print(result.stdout,end='')

    def test_branch_truth_propagation_preserves_all_half_words_and_nan_comparisons(self):
        compiler=StringCompiler();domains={"X1":Domain(F(-65504),F(65504),-24,False)}
        c="Bits64(X1)<9223372036854775808"
        sources=[f"Piecewise((Piecewise((Float64(Bits64(X1)),{c}),(777.0,True)),{c}),(Piecewise((999.0,{c}),(-X1,True)),True))",
            "Piecewise((Piecewise((1.0,X1>=0),(2.0,True)),Not(X1<0)),(3.0,True))",
            f"Piecewise((Piecewise(((-0.0),Not({c})),(X1,True)),Not({c})),(Piecewise((X1,{c}),(0.0,True)),True))"]
        outputs=[compiler.stabilize(source,domains) for source in sources]
        self.assertEqual([s.count("Piecewise(") for s in outputs],[1,2,1])
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"conditions.cpp";binary=Path(directory)/"conditions"
            functions=[];checks=[]
            for index,(before,after) in enumerate(zip(sources,outputs)):
                functions.extend([f"double before{index}(double X1){{return "+cpp(syntax(before))+";}",f"double after{index}(double X1){{return "+cpp(syntax(after))+";}"])
                checks.append(f"mismatches+=word<uint64_t>(before{index}(x))!=word<uint64_t>(after{index}(x));cases++;")
            source.write_text("#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n"+"\n".join(functions)+"\nint main(){unsigned cases=0,mismatches=0;for(unsigned bits=0;bits<65536;bits++){double x=word<_Float16>(uint16_t(bits));"+"\n".join(checks)+'}\nstd::printf("Exact branch facts native parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=196608 mismatches=0',result.stdout);print(result.stdout,end='')

    def test_compose_closed_matches_literal_closure_and_checks_budget_before_restoration(self):
        domains={"X1":Domain(F(-1),F(1),-24,False)}
        compiler=StringCompiler(max_characters=100000)
        session=ConversionSession(compiler,domains,input_dtype="f16")
        first=session.close("R16(R32(X1+X1/2.0))")
        second=session.close("R16(R32(X1*0.75))")
        for literal in (first,second):compiler.register_completed_region(literal,domains,word_closed=True)
        template="R16(R32(X999999998 * X999999999))"
        bindings={"X999999998":first,"X999999999":second}
        expected=session.close("R16(R32(("+first+") * ("+second+")))")
        before=len(compiler.substitution_events)
        actual=session.compose_closed(template,bindings)
        self.assertEqual(actual,expected)
        self.assertEqual([event[0] for event in compiler.substitution_events[before:] if event[0] in bindings],list(bindings))
        self.assertNotIn("CASNumericRegion",actual)
        self.assertNotIn("X99999999",actual)
        self.assertEqual(session.value_kind(syntax(actual)),"half")
        fresh=ConversionSession(compiler,domains,input_dtype="f16")
        with self.assertRaisesRegex(ValueError,"from this session"):fresh.compose_closed(template,bindings)
        with self.assertRaisesRegex(ValueError,"branch arms separately"):
            session.compose_closed("Piecewise((X999999998,X1>0),(0.0,True))",{"X999999998":first})
        compiler.max_characters=len(template)+100
        with self.assertRaisesRegex(ValueError,"budget before allocation"):
            session.compose_closed(template,bindings)
        self.assertGreater(session.numeric_envelopes[-1][3],compiler.max_characters)

    @patch.object(ConversionSession,'close_frontier_candidates',lambda self,compact,baseline,*args:baseline)
    def test_completed_numeric_envelopes_reuse_only_session_proofs_and_restore_literals(self):
        import direct_sympy_conversions as conversions
        domains={"X1":Domain(F(-1),F(1),-24,False)}
        compiler=StringCompiler(max_characters=100000)
        session=ConversionSession(compiler,domains,input_dtype="f16")
        first=session.close("R16(R32(X1+X1/2.0))")
        compiler.register_completed_region(first,domains,word_closed=True)
        expression="R16(R32(("+first+")*0.5))"
        with patch.object(conversions,"simplify_arithmetic",wraps=conversions.simplify_arithmetic) as arithmetic:
            result=session.close(expression)
        self.assertEqual(len(session.numeric_envelopes),1)
        self.assertLess(len(arithmetic.call_args_list[0].args[0]),100)
        self.assertNotIn("CASNumericRegion",result)
        self.assertIn(first,result)
        self.assertEqual(session.value_kind(syntax(result)),"half")
        # A different session has no authority to inherit numeric proofs.
        other=ConversionSession(compiler,domains,input_dtype="f16")
        other.close(expression);self.assertEqual(other.numeric_envelopes,[])
        before=len(session.numeric_envelopes)
        session.close("Piecewise(("+expression+",X1>0),(0.0,True))")
        self.assertEqual(len(session.numeric_envelopes),before)
        with self.assertRaisesRegex(ValueError,"Reserved compiler numeric"):
            session.close("CASNumericRegion1()")
        compiler.max_characters=len(expression)+100
        with self.assertRaisesRegex(ValueError,"budget before allocation"):
            session.close(expression)
        self.assertGreater(session.numeric_envelopes[-1][3],compiler.max_characters)

    def test_native_math_rationals_do_not_use_unsigned_integer_division(self):
        # Plain mathematical arithmetic is floating; unsigned operations have
        # their explicit U64* calls. SymPy may print an exact literal as 3/4.
        expressions=["0.75","0.125","1.5","3.5","7.75","-0.75","0.5+0.75"]
        compiler=StringCompiler();lines=[]
        for text in expressions:
            expected=float(eval(text,{"__builtins__":{}}))
            lowered=lower_finite_conversion(text,"R32",FiniteSource(expected,expected,-4),compiler,{})
            lines.append("mismatches += word<uint64_t>(double("+cpp(syntax(lowered))+")) != word<uint64_t>(double("+repr(expected)+"));")
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"literal.cpp";binary=Path(directory)/"literal"
            source.write_text("#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\nint main(){unsigned mismatches=0;\n"+"\n".join(lines)+"\nstd::printf(\"Literal math native certificate: cases=7 mismatches=%u\\n\",mismatches);return mismatches?1:0;}")
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=7 mismatches=0',result.stdout);print(result.stdout,end='')

    def test_common_inverse_cannot_be_factored_through_half_stores(self):
        compiler=StringCompiler();domains={"X1":Domain(F(-65504),F(65504),-24,False),"X2":Domain(F(-65504),F(65504),-24,False),"X3":Domain(F(1,65536),F(1024),-39,True)}
        original="R16(R32(R16(R32(X1*X3))*0.25 + R16(R32(X2*X3))*0.5))"
        stabilized=compiler.stabilize(original,domains)
        self.assertEqual(ast.dump(syntax(stabilized)),ast.dump(syntax(original)))
        def reference_cpp(node):
            if isinstance(node,ast.Call) and node.func.id in ("R16","R32"):
                value=reference_cpp(node.args[0])
                return "double(float("+value+"))" if node.func.id=="R32" else "double(static_cast<_Float16>("+value+"))"
            if isinstance(node,ast.BinOp):
                op={ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/"}[type(node.op)]
                return "("+reference_cpp(node.left)+op+reference_cpp(node.right)+")"
            return cpp(node)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"proof.cpp";binary=Path(directory)/"proof"
            source.write_text("""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1,double X2,double X3){return """+reference_cpp(syntax(stabilized))+""";}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned mismatches=0,unsafe=0,cases=0;
for(unsigned bits=1;bits<0x7c00;bits++)for(uint16_t other:{uint16_t(1),uint16_t(0x3c00),uint16_t(0x7bff),uint16_t(0xbc00)}){
 double x=word<_Float16>(uint16_t(bits)),y=word<_Float16>(other);
 float variance=(float(x*x)+float(y*y))/2.0f+1e-6f;
 double inverse=1.0f/std::sqrt(variance);
 double a=static_cast<_Float16>(float(x*inverse)),b=static_cast<_Float16>(float(y*inverse));
 double expected=static_cast<_Float16>(float(a*0.25+b*0.5));
 double factored=static_cast<_Float16>(float((x*0.25+y*0.5)*inverse));
 cases++;mismatches+=word<uint64_t>(candidate(x,y,inverse))!=word<uint64_t>(expected);
 unsafe+=word<uint64_t>(factored)!=word<uint64_t>(expected);
}std::printf("Half normalization factor barrier: cases=%u mismatches=%u unsafeRewrites=%u\\n",cases,mismatches,unsafe);
return mismatches||!unsafe;}
""")
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('mismatches=0',result.stdout);print(result.stdout,end='')

    def test_f32_sqrt_and_reciprocal_normal_cells_need_no_parity_copy(self):
        domains={"X1":Domain(F(2)**-149,F(3.4028234663852886e38),-149,True)}
        session=ConversionSession(StringCompiler(),domains)
        session.f32_values.add(session.key(syntax("X1")))
        self.assertTrue(session.no_odd_f32_ties(syntax("sqrt(X1)")))
        self.assertTrue(session.no_odd_f32_ties(syntax("1.0/X1")))
        unknown=ConversionSession(StringCompiler(),domains)
        self.assertFalse(unknown.no_odd_f32_ties(syntax("sqrt(X1)")))
        self.assertFalse(unknown.no_odd_f32_ties(syntax("1.0/X1")))
        self.assertFalse(session.no_odd_f32_ties(syntax("sqrt(-X1)")))
        self.assertFalse(session.no_odd_f32_ties(syntax("2.0/X1")))
        sqrt=lower_finite_conversion("sqrt(X1)","R32",session.bounds(syntax("sqrt(X1)")),session.compiler,domains,no_odd_f32_ties=True)
        reciprocal_domains={"X1":Domain(F(2)**-127,F(2)**125,-149,True)}
        reciprocal_session=ConversionSession(StringCompiler(),reciprocal_domains)
        reciprocal=lower_finite_conversion("1.0/X1","R32",reciprocal_session.bounds(syntax("1.0/X1")),reciprocal_session.compiler,reciprocal_domains,no_odd_f32_ties=True)
        self.assertEqual(sqrt.count("X1"),1);self.assertEqual(reciprocal.count("X1"),1)
        source=r'''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfenv>
#include <cfloat>
#include <cmath>
template<class T,class U>T word(U x){T y;static_assert(sizeof(x)==sizeof(y));std::memcpy(&y,&x,sizeof(y));return y;}
double root(double X1){return ROOT;}
double inverse(double X1){return INVERSE;}
int main(){static_assert(FLT_EVAL_METHOD==0);if(std::fesetround(FE_TONEAREST))return 2;
uint64_t roots=0,inverses=0,errors=0,ties=0;constexpr uint64_t low=(UINT64_C(1)<<29)-1,mid=UINT64_C(1)<<28;
for(uint32_t bits=0;bits<=UINT32_C(0x7f7fffff);bits++){
 float x=word<float>(bits);double value=std::sqrt(double(x));ties+=(word<uint64_t>(value)&low)==mid;
 errors+=word<uint64_t>(root(x))!=word<uint64_t>(double(std::sqrt(x)));roots++;
}
double negativeZero=-0.0;if(word<uint64_t>(root(negativeZero))!=word<uint64_t>(negativeZero))return 3;
for(uint32_t bits=UINT32_C(0x00400000);bits<=UINT32_C(0x7e000000);bits++){
 float x=word<float>(bits);double value=1.0/double(x);ties+=(word<uint64_t>(value)&low)==mid;
 errors+=word<uint64_t>(inverse(x))!=word<uint64_t>(double(1.0f/x));
 errors+=word<uint64_t>(inverse(-x))!=word<uint64_t>(double(1.0f/(-x)));inverses+=2;
}
std::printf("F32 normal-cell certificate: sqrt=%llu reciprocals=%llu mismatches=%llu ties=%llu\n",(unsigned long long)roots,(unsigned long long)inverses,(unsigned long long)errors,(unsigned long long)ties);
return errors||ties?1:0;}
'''.replace("ROOT",cpp(syntax(sqrt))).replace("INVERSE",cpp(syntax(reciprocal)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cells.cpp";path.write_text(source);executable=Path(directory)/"cells"
            subprocess.run(["clang++","-std=c++17","-O3","-ffp-contract=off",str(path),"-o",str(executable)],check=True,capture_output=True)
            output=subprocess.run([str(executable)],check=True,capture_output=True,text=True,timeout=90).stdout
            self.assertIn("sqrt=2139095040 reciprocals=4219469826 mismatches=0 ties=0",output);print(output.strip())

    def test_two_half_squares_remove_only_the_proved_odd_tie_copy(self):
        domains={name:Domain(F(-65504),F(65504),-24,False) for name in ("X1","X2")}
        session=ConversionSession(StringCompiler(),domains,input_dtype="f16")
        for source in ("X1**2+X2**2","X1*X1+X2*X2"):
            self.assertTrue(session.no_odd_f32_ties(syntax(source)))
        for source in ("X1+X2","X1**2+X2","X1**2+X2**3","X1**2+X2**2+X1**2"):
            self.assertFalse(session.no_odd_f32_ties(syntax(source)))
        unknown=ConversionSession(StringCompiler(),domains)
        self.assertFalse(unknown.no_odd_f32_ties(syntax("X1**2+X2**2")))
        self.assertEqual(session.value_kind(syntax("Silu16(X1)")),"half")
        self.assertEqual(session.value_kind(syntax("Silu16(X1)*X2")),"f32")
        self.assertIsNone(unknown.bounds(syntax("Silu16(X1)")))
        with self.assertRaises(ValueError):lower_finite_conversion("X1","R16",FiniteSource(-1,1,-24),session.compiler,domains,no_odd_f32_ties=True)
        result=session.close("R32(X1**2 + X2**2)")
        self.assertEqual(result.count("X1"),1);self.assertEqual(result.count("X2"),1)
        general=lower_finite_conversion("X1+X2","R32",FiniteSource(-131008,131008,-24),session.compiler,domains)
        self.assertEqual(general.count("X1"),2)
        source=r'''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <vector>
template<class T,class U>T word(U x){T y;static_assert(sizeof(x)==sizeof(y));std::memcpy(&y,&x,sizeof(y));return y;}
double candidate(double X1,double X2){return CANDIDATE;}
double general(double X1,double X2){return GENERAL;}
int main(){static_assert(FLT_EVAL_METHOD==0);if(std::fesetround(FE_TONEAREST))return 2;
std::vector<double> values(31744);for(unsigned i=0;i<31744;i++)values[i]=word<_Float16>(uint16_t(i));
uint64_t cases=0,errors=0,oddTies=0,evenTies=0;
constexpr uint64_t low=(UINT64_C(1)<<29)-1,mid=UINT64_C(1)<<28;
for(unsigned i=0;i<31744;i++)for(unsigned j=0;j<=i;j++){
 double x=values[i],y=values[j],sum=x*x+y*y;uint64_t raw=word<uint64_t>(sum);
 bool tie=(raw&low)==mid;oddTies+=tie&&((raw>>29)&1);evenTies+=tie&&!((raw>>29)&1);
 errors+=word<uint64_t>(candidate(x,y))!=word<uint64_t>(double(float(sum)));cases++;
}
// General Half addition has odd ties and must retain the parity correction.
double sum=1.0+3.0*std::ldexp(1.0,-24);uint64_t raw=word<uint64_t>(sum);
if(word<uint64_t>(general(1.0,3.0*std::ldexp(1.0,-24)))!=word<uint64_t>(double(float(sum))))return 3;
if(((raw+(mid-1))&~low)==word<uint64_t>(double(float(sum))))return 4;
std::printf("Two-Half-square F32 certificate: pairs=%llu mismatches=%llu oddTies=%llu evenTies=%llu\n",(unsigned long long)cases,(unsigned long long)errors,(unsigned long long)oddTies,(unsigned long long)evenTies);
return errors||oddTies?1:0;}
'''.replace("CANDIDATE",cpp(syntax(result))).replace("GENERAL",cpp(syntax(general)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"squares.cpp";path.write_text(source);executable=Path(directory)/"squares"
            subprocess.run(["clang++","-std=c++17","-O3","-ffp-contract=off",str(path),"-o",str(executable)],check=True,capture_output=True)
            output=subprocess.run([str(executable)],check=True,capture_output=True,text=True,timeout=90).stdout
            self.assertIn("pairs=503856640 mismatches=0 oddTies=0",output);print(output.strip())

    def test_half_square_compacts_one_operand_with_exhaustive_bit_parity(self):
        domains={"X1":Domain(F(-65504),F(65504),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype="f16")
        result=session.close("R32(X1 ** 2)")
        self.assertEqual(ast.dump(syntax(result)),ast.dump(syntax("X1 ** 2")))
        self.assertEqual(session.value_kind(syntax(result)),"f32")
        self.assertTrue(session.no_negative_zero(syntax(result)))
        square=session.bounds(syntax(result));product=session.bounds(syntax("X1*X1"))
        self.assertEqual(square.quantum,product.quantum)
        self.assertGreaterEqual(square.minimum,product.minimum)
        self.assertLessEqual(square.maximum,product.maximum)
        unknown=ConversionSession(StringCompiler(),domains)
        self.assertIsNone(unknown.bounds(syntax("X1 ** 2")))
        self.assertIn("R32",unknown.close("R32(X1 ** 2)"))
        self.assertIsNone(session.bounds(syntax("X1 ** 3")))
        count=0
        for bits in range(65536):
            x=struct.unpack("e",struct.pack("H",bits))[0]
            if not math.isfinite(x):continue
            expected=struct.pack("d",x*x)
            self.assertEqual(struct.pack("d",x**2),expected,bits)
            self.assertEqual(struct.pack("d",float(struct.unpack("f",struct.pack("f",x*x))[0])),expected,bits)
            count+=1
        source=r'''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
int main(){unsigned count=0;for(unsigned bits=0;bits<65536;++bits){
uint16_t raw=bits;_Float16 half;std::memcpy(&half,&raw,2);double x=half;
if(!std::isfinite(x))continue;double a=x*x,b=std::pow(x,2.0),c=float(x*x);
if(std::memcmp(&a,&b,8)||std::memcmp(&a,&c,8))return 1;++count;}
std::printf("Native Half squares: %u, mismatches=0\\n",count);}
'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"squares.cpp";path.write_text(source)
            executable=Path(directory)/"squares"
            subprocess.run(["clang++","-std=c++17","-O2","-ffp-contract=off",str(path),"-o",str(executable)],check=True,capture_output=True)
            output=subprocess.run([str(executable)],check=True,capture_output=True,text=True).stdout
            self.assertIn(str(count),output);print(output.strip())
        self.assertEqual(count,63488)

    def test_binade_proofs_preserve_predecessors_of_powers_of_two(self):
        for exponent in range(-1021,1023):
            power=2.0**exponent
            predecessor=math.nextafter(power,0)
            self.assertEqual(binary_exponent(predecessor),exponent-1)
            self.assertEqual(binary_exponent(power),exponent)
        x=math.nextafter(2.0**-126,0)
        session=ConversionSession(StringCompiler(),{"X1":Domain(F(x),F(x),-179,True)})
        self.assertEqual(session.bounds(syntax("X1+0")).quantum,-179)

    def test_normalization_certificate_rejects_unproved_geometry_and_epsilon(self):
        from direct_sympy_checkpoint import rms_half_bound
        for width,epsilon in ((0,1e-6),(1000001,1e-6),(2,0),(2,2**-127),(2,math.inf),(2,math.nan)):
            self.assertIsNone(rms_half_bound(width,epsilon))
        session=ConversionSession(StringCompiler(),{},input_dtype="f16")
        overflow=syntax("R16(65504.0*65504.0)")
        self.assertIsNone(session.bounds(overflow))
        self.assertIsNone(session.value_kind(overflow))
        for width in (1,2,1024,1000000):
            bound=rms_half_bound(width,1e-6)
            self.assertGreater(bound,math.sqrt(width))
            self.assertLess(bound,1.25*math.sqrt(width))

    def test_word_identities_are_typed_and_stabilized(self):
        compiler=StringCompiler();domains={"X1":Domain(F(-65504),F(65504),-24,False)}
        self.assertEqual(simplify_words("Bits64(Float64(U64And(Bits64(X1), 255)))",compiler,domains),"U64And(Bits64(X1), 255)")
        self.assertEqual(simplify_words("Float64(Bits64(X1))",compiler,domains),"X1")
        self.assertEqual(simplify_words("U64And(U64Shr(U64And(Bits64(X1), 18446744073172680704), 42), 1)",compiler,domains),"U64And(U64Shr(Bits64(X1), 42), 1)")
        self.assertIn("Float64",simplify_words("Bits64(Float64(unknown(X1)))",compiler,domains))
        self.assertIn("Bits64",simplify_words("Float64(Bits64(unknown(X1)))",compiler,domains))
        self.assertIn("Bits64",simplify_words("Float64(Bits64(not X1))",compiler,domains))
        self.assertTrue(all(event[2:4]==("factor","simplify") for event in compiler.events))
        masks=(1<<64)-1
        rules=["Bits64(Float64(U64And(Bits64(X1),255)))","Float64(Bits64(X1))",
            "U64And(U64And(Bits64(X1),255),15)","U64Add(Bits64(X1),0)",
            "U64Or(Bits64(X1),0)","U64And(Bits64(X1),Bits64(X1))",
            "U64Shr(U64And(Bits64(X1),255),4)","U64Add(18446744073709551615,1)"]
        functions={"Bits64":lambda x:struct.unpack("Q",struct.pack("d",x))[0],
            "Float64":lambda x:struct.unpack("d",struct.pack("Q",x))[0],
            "U64And":lambda a,b:a&b,"U64Or":lambda a,b:a|b,
            "U64Add":lambda a,b:(a+b)&masks,"U64Shr":lambda a,b:a>>b}
        rng=random.Random(72)
        words=[0,1,masks,1<<63,0x7ff0000000000000,0x7ff8000000000001]+[rng.getrandbits(64) for _ in range(2000)]
        comparisons=0
        for rule in rules:
            simplified=simplify_words(rule,compiler,domains)
            before=compile(ast.Expression(syntax(rule)),"before","eval")
            after=compile(ast.Expression(syntax(simplified)),"after","eval")
            for bits in words:
                env={"__builtins__":{},**functions,"X1":functions["Float64"](bits)}
                a,b=eval(before,env),eval(after,env)
                if type(a) is float:a=functions["Bits64"](a)
                if type(b) is float:b=functions["Bits64"](b)
                self.assertEqual(a,b,(rule,bits));comparisons+=1
        print(f"Word identity comparisons: {comparisons}, mismatches=0")

    def test_word_envelope_exposes_reinterpretation_without_rewalking_producers(self):
        from unittest.mock import patch
        import direct_sympy_words as words
        domains={"X1":Domain(F(-65504),F(65504),-24,False)}
        compiler=StringCompiler(max_characters=20000)
        region="Float64(U64And(U64Add(Bits64(X1), 268435455), 18446744073172680704))"
        region=simplify_words(region,compiler,domains)
        compiler.register_completed_region(region,domains,word_closed=True)
        expression="U64And("+"U64Add("*100+"Bits64("+region+")"+", 1)"*100+", 255)"
        reference=simplify_words(expression,StringCompiler(max_characters=20000),domains)
        with patch.object(words,"syntax",wraps=words.syntax) as parse:
            actual=simplify_words(expression,compiler,domains)
        self.assertEqual(actual,reference)
        self.assertLess(max(len(c.args[0]) for c in parse.call_args_list),len(expression))
        self.assertNotIn("CASWordPayload",actual);self.assertNotIn("CASStableRegion",actual)
        self.assertEqual(simplify_words("Bits64("+region+")",compiler,domains),region[8:-1])
        narrowed={"X1":Domain(F(1),F(2),-24,True)}
        self.assertEqual(simplify_words(expression,compiler,narrowed),simplify_words(expression,StringCompiler(max_characters=20000),narrowed))
        with self.assertRaisesRegex(ValueError,"Reserved"):
            simplify_words("Bits64(CASWordPayload0)",compiler,domains)
        compiler.max_characters=len(expression)-1
        with self.assertRaisesRegex(ValueError,"input budget"):
            simplify_words(expression,compiler,domains)
        compiler.max_characters=20000
        unproved=StringCompiler()
        pending="Float64(U64And(U64And(Bits64(X1),255),15))"
        unproved.register_completed_region(pending,domains)
        self.assertEqual(simplify_words(pending,unproved,domains),simplify_words(pending,StringCompiler(),domains))
        branch="Piecewise((Bits64("+region+"),X1>0),(Bits64("+region+"),True))"
        self.assertEqual(simplify_words(branch,compiler,domains),simplify_words(branch,StringCompiler(),domains))

    def test_structural_signature_preserves_bits_and_invalidates_mutation(self):
        signatures=StructuralSignatures()
        for expression in ("X1", "R16(X1*2.0)", "Bits64(X1)"):
            self.assertEqual(signatures.key(syntax(expression)),signatures.key(syntax(expression)))
        self.assertNotEqual(signatures.key(ast.Constant(value=0.0)),signatures.key(ast.Constant(value=-0.0)))
        self.assertNotEqual(signatures.key(ast.Constant(value=1)),signatures.key(ast.Constant(value=1.0)))
        self.assertNotEqual(signatures.key(syntax("X1")),signatures.key(syntax("X10")))
        tree=syntax("R16(X1+X2)");old=signatures.key(tree)
        tree.args[0].right=syntax("X3")
        signatures.invalidate(tree.args[0]);signatures.invalidate(tree)
        self.assertNotEqual(old,signatures.key(tree))
        self.assertEqual(signatures.key(tree),signatures.key(syntax("R16(X1+X3)")))
        # Pre-bound parent signatures must not leak after child lowering.
        session=ConversionSession(StringCompiler(),{"X1":Domain(F(-65504),F(65504),-24,False)},input_dtype="f16")
        expression=session.close("R16(R32(X1*1.0))")
        self.assertEqual(session.value_kind(syntax(expression)),"half")

    def test_converted_region_reuse_keeps_literal_copies_and_pending_boundaries(self):
        domains={name:Domain(F(-1),F(1),-24,False) for name in ("X1","X2")}
        sessions=[ConversionSession(StringCompiler(),domains,input_dtype="f16") for _ in range(2)]
        with self.assertRaises(TypeError):sessions[0].domains["X1"]=Domain(F(-2),F(2),-24,False)
        first=[session.close("R16(X1+X2)") for session in sessions]
        self.assertEqual(*first)
        sessions[1].converted_regions.clear()
        # Disable every completed-literal cache for the cold control. Merely
        # clearing converted keys now leaves certified root/literal reuse live.
        sessions[1].closed_literals.clear();sessions[1].closed_literal_keys.clear();sessions[1].closed_literal_pure.clear();sessions[1].closed_literal_characters=0
        sessions[1].compiler._regions.clear();sessions[1].compiler._region_roots.clear();sessions[1].compiler._region_characters=0
        visited=[session.visited_nodes for session in sessions]
        results=[session.close("R32("+expression+")") for session,expression in zip(sessions,first)]
        self.assertEqual(*results)
        self.assertGreater(sessions[0].reused_regions,0)
        self.assertLess(sessions[0].visited_nodes-visited[0],sessions[1].visited_nodes-visited[1])
        self.assertIn("X1",results[0]);self.assertIn("X2",results[0])
        unknown=ConversionSession(StringCompiler(),{})
        pending=unknown.close("R16(X3)")
        self.assertFalse(unknown.converted_regions)
        self.assertIn("R16",unknown.close(pending))
        self.assertEqual(unknown.reused_regions,0)

    def test_tandem_rejects_unproved_f32_source(self):
        for certificate in (None,FiniteSource(-1,1,None),FiniteSource(-1,1,-127),FiniteSource(-2**128,2**128,-24)):
            self.assertFalse(tandem_supported(certificate))
        self.assertTrue(tandem_supported(FiniteSource(-131008,131008,-126)))

    def test_arithmetic_identities_require_the_correct_zero_sign(self):
        from direct_sympy_arithmetic import simplify_arithmetic
        domains={"X1":Domain(F(-1),F(1),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype="f16")
        self.assertIn("+",simplify_arithmetic("X1+0.0",session))
        self.assertEqual(simplify_arithmetic("X1*1.0",session),"X1")
        self.assertEqual(simplify_arithmetic("X1/1.0",session),"X1")
        self.assertFalse(session.no_negative_zero(syntax("R16(X1*0.00000001)")))
        self.assertTrue(session.no_negative_zero(syntax("R32(0.0+X1)")))
        self.assertTrue(session.no_negative_zero(syntax("X1-X1")))
        self.assertFalse(session.no_negative_zero(syntax("-X1")))
        certified=ConversionSession(StringCompiler(),{"X1":Domain(F(-1),F(1),-24,True)},input_dtype="f16")
        self.assertEqual(simplify_arithmetic("X1+0.0",certified),"X1")
        result=simplify_arithmetic("R32(R32(0.0+X1))+0.0",session)
        self.assertEqual(ast.dump(syntax(result)),ast.dump(syntax("0.0+X1")))

    def test_constructor_rejects_uncertified_sources(self):
        with self.assertRaises(ValueError):FiniteSource(-math.inf,1)
        with self.assertRaises(ValueError):lower_finite_conversion("X1","R32",None,StringCompiler(),{})

    def test_exact_subnormal_grid_removes_only_the_unneeded_small_kernel(self):
        compiler=StringCompiler()
        for kind,q in (("R32",-149),("R16",-24)):
            exact=lower_finite_conversion("X1",kind,FiniteSource(-2,2,q),compiler,{})
            between=lower_finite_conversion("X1",kind,FiniteSource(-2,2,q-1),compiler,{})
            self.assertNotIn("Piecewise",exact)
            self.assertIn("Piecewise",between)
        self.assertNotIn("Piecewise",lower_tandem("X1",FiniteSource(-2,2,-24),compiler,{}))
        self.assertEqual(lower_tandem("X1",FiniteSource(-2**-15,2**-15,-24),compiler,{}),lower_tandem("X1",FiniteSource(-2,2,-24),compiler,{}))
        self.assertIn("Piecewise",lower_tandem("X1",FiniteSource(-2,2,-25),compiler,{}))

    def test_unknown_source_remains_a_barrier_in_the_session(self):
        session=ConversionSession(StringCompiler(),{})
        self.assertIn("R16",session.close("R16(X2)"))
        self.assertEqual(session.closed,0)
        self.assertEqual(session.pending,1)

    def test_actual_strings_match_native_ieee_conversion_cells_and_exponents(self):
        compiler=StringCompiler()
        r32=lower_finite_conversion("X1","R32",FiniteSource(-sys.float_info.max,sys.float_info.max),compiler,{})
        r16=lower_finite_conversion("X1","R16",FiniteSource(-sys.float_info.max,sys.float_info.max),compiler,{})
        pruned32=lower_finite_conversion("X1","R32",FiniteSource(-2,2,-126),compiler,{})
        pruned16=lower_finite_conversion("X1","R16",FiniteSource(-2**-15,2**-15),compiler,{})
        self.assertNotIn("Piecewise",pruned32+pruned16)
        session=ConversionSession(compiler,{"X1":Domain(F(-65504),F(65504),-1074,False)})
        composed=session.close("R16(R32(X1))")
        tandem=lower_tandem("X1",FiniteSource(-131008,131008,-126),compiler,{})
        self.assertEqual(session.closed,2)
        self.assertNotIn("R16",composed);self.assertNotIn("R32",composed)
        self.assertNotIn("R32",r32);self.assertNotIn("R16",r16)
        self.assertNotIn("X999999997",r32+r16)
        self.assertTrue(any(event[-1]=="branch-contexts" for event in compiler.events))
        word=simplify_words("U64And(U64Shr(U64And(Bits64(X1), 18446744073172680704), 42), 1)",compiler,{})
        typed=ConversionSession(compiler,{name:Domain(F(-65504),F(65504),-24,False) for name in ("X1","X2")},input_dtype="f16")
        product=typed.close("R32(X1*X2)")
        addition=typed.close("R16(R32(X1+X2))")
        zero=typed.close("R16(R32((0.0+0.0)+X1))")
        grid32=lower_finite_conversion("X1","R32",FiniteSource(-2,2,-149),compiler,{})
        grid16=lower_finite_conversion("X1","R16",FiniteSource(-2,2,-24),compiler,{})
        grid_tandem=lower_tandem("X1",FiniteSource(-2,2,-24),compiler,{})
        for expression in (product,addition,zero):
            self.assertNotIn("R32(",expression);self.assertNotIn("R16(",expression)
        self.assertGreaterEqual(typed.redundant+typed.arithmetic_eliminated,4)
        with tempfile.TemporaryDirectory(prefix="sympy-conversion-certificate-") as directory:
            root=Path(directory)
            source=root/"proof.cpp";binary=root/"proof"
            source.write_text("""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <cfloat>
#include <initializer_list>
#include <limits>
template<class T,class U>T word(U value){static_assert(sizeof(T)==sizeof(U));T result;std::memcpy(&result,&value,sizeof(result));return result;}
double r32(double X1){return """+cpp(syntax(r32))+""";}
double r16(double X1){return """+cpp(syntax(r16))+""";}
double tandem(double X1){return """+cpp(syntax(tandem))+""";}
double composed(double X1){return """+cpp(syntax(composed))+""";}
double pruned32(double X1){return """+cpp(syntax(pruned32))+""";}
double pruned16(double X1){return """+cpp(syntax(pruned16))+""";}
uint64_t wordParity(double X1){return """+cpp(syntax(word))+""";}
double typedProduct(double X1,double X2){return """+cpp(syntax(product))+""";}
double typedAddition(double X1,double X2){return """+cpp(syntax(addition))+""";}
double typedZero(double X1,double X2){return """+cpp(syntax(zero))+""";}
double grid32(double X1){return """+cpp(syntax(grid32))+""";}
double grid16(double X1){return """+cpp(syntax(grid16))+""";}
double gridTandem(double X1){return """+cpp(syntax(grid_tandem))+""";}
int main(){
 for(uint64_t bits:{UINT64_C(0),UINT64_C(1),UINT64_C(0x8000000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0xffffffffffffffff)})
  if(wordParity(word<double>(bits))!=((bits>>42)&1))return 4;
 static_assert(FLT_EVAL_METHOD==0);static_assert(std::numeric_limits<double>::is_iec559);
 if(std::fesetround(FE_TONEAREST))return 2;
 uint64_t grid32Cases=0,grid16Cases=0;
 for(uint32_t bits=0;bits<0x800000;bits++)for(uint32_t sign:{uint32_t(0),uint32_t(0x80000000)}){
  double x=word<float>(bits|sign);grid32Cases++;
  if(word<uint64_t>(grid32(x))!=word<uint64_t>(x))return 6;
 }
 for(uint16_t bits=0;bits<0x0400;bits++)for(uint16_t sign:{uint16_t(0),uint16_t(0x8000)}){
  double x=word<_Float16>(uint16_t(bits|sign));grid16Cases++;
  if(word<uint64_t>(grid16(x))!=word<uint64_t>(x))return 7;
 }
 std::printf("Exact subnormal-grid certificates: F32=%llu F16=%llu mismatches=0\\n",(unsigned long long)grid32Cases,(unsigned long long)grid16Cases);
 uint64_t gridTandemCases=0;
 for(uint32_t i=0;i<=33554432;i++)for(double sign:{1.,-1.}){
  double x=std::ldexp(double(i),-24)*sign;gridTandemCases++;
  double expected=static_cast<_Float16>(static_cast<float>(x));
  if(word<uint64_t>(gridTandem(x))!=word<uint64_t>(expected))return 8;
 }
 std::printf("Exact Half-grid tandem certificate: cases=%llu mismatches=0\\n",(unsigned long long)gridTandemCases);
 uint64_t cases32=0,cases16=0,composedCases=0,pruned32Cases=0,pruned16Cases=0,tandemCases=0,failures=0;
 auto check32=[&](double x){cases32++;double expected=static_cast<float>(x);if(word<uint64_t>(r32(x))!=word<uint64_t>(expected))failures++;};
 auto check16=[&](double x){cases16++;double expected=static_cast<_Float16>(x);if(word<uint64_t>(r16(x))!=word<uint64_t>(expected))failures++;};
 auto checkTandem=[&](double x){tandemCases++;double expected=static_cast<_Float16>(static_cast<float>(x));if(word<uint64_t>(tandem(x))!=word<uint64_t>(expected))failures++;};
 auto checkComposed=[&](double x){composedCases++;double expected=static_cast<_Float16>(static_cast<float>(x));if(word<uint64_t>(composed(x))!=word<uint64_t>(expected))failures++;checkTandem(x);};
 auto checkPruned32=[&](double x){pruned32Cases++;double expected=static_cast<float>(x);if(word<uint64_t>(pruned32(x))!=word<uint64_t>(expected))failures++;};
 auto checkPruned16=[&](double x){pruned16Cases++;double expected=static_cast<_Float16>(x);if(word<uint64_t>(pruned16(x))!=word<uint64_t>(expected))failures++;};
 for(unsigned exponent=126;exponent<=127;exponent++)for(uint32_t fraction=0;fraction<0x800000;fraction++){
  uint32_t bits=(exponent<<23)|fraction;
  double a=word<float>(bits),b=word<float>(bits+1),mid=(a+b)/2;
  for(double x:{std::nextafter(mid,-INFINITY),mid,std::nextafter(mid,INFINITY)})for(double sign:{1.,-1.}){check32(x*sign);checkPruned32(x*sign);}
 }
 for(uint16_t bits=0;bits<0x7bff;bits++){
  double a=word<_Float16>(bits),b=word<_Float16>(uint16_t(bits+1)),mid=(a+b)/2;
  for(double x:{std::nextafter(mid,-INFINITY),mid,std::nextafter(mid,INFINITY)})for(double sign:{1.,-1.}){check16(x*sign);checkComposed(x*sign);if(bits<512)checkPruned16(x*sign);}
 }
 for(uint16_t bits=0x0400;bits<0x7bff;bits++){
  double a=word<_Float16>(bits),b=word<_Float16>(uint16_t(bits+1)),mid=(a+b)/2;
  uint64_t wordMid=word<uint64_t>(mid);
  for(uint64_t boundary:{wordMid-(UINT64_C(1)<<28),wordMid+(UINT64_C(1)<<28)})
   for(int offset:{-1,0,1})for(double sign:{1.,-1.})checkComposed(word<double>(uint64_t(boundary+offset))*sign);
 }
 for(unsigned exponent=0;exponent<2047;exponent++)for(uint64_t fraction:{UINT64_C(0),UINT64_C(1),UINT64_C(0x7ffffffffffff),UINT64_C(0xfffffffffffff)})for(uint64_t sign:{UINT64_C(0),UINT64_C(0x8000000000000000)}){
  double x=word<double>(sign|(uint64_t(exponent)<<52)|fraction);check32(x);check16(x);
 }
 // Explicit target underflow, smallest-normal, binade and overflow cells.
 for(double mid:{std::ldexp(1.,-150),std::ldexp(1.,-126)-std::ldexp(1.,-150),std::ldexp(1.,128)-std::ldexp(1.,103),std::ldexp(1.,-25),std::ldexp(1.,-14)-std::ldexp(1.,-25),65520.})
  for(double x:{std::nextafter(mid,-INFINITY),mid,std::nextafter(mid,INFINITY)})for(double sign:{1.,-1.}){check32(x*sign);check16(x*sign);}
 for(double mid:{std::ldexp(1.,-25),std::ldexp(1.,-14)-std::ldexp(1.,-25),65520.}){
  uint64_t wordMid=word<uint64_t>(mid);
  for(uint64_t boundary:{wordMid-(UINT64_C(1)<<28),wordMid+(UINT64_C(1)<<28)})
   for(int offset:{-1,0,1})for(double sign:{1.,-1.})checkTandem(word<double>(uint64_t(boundary+offset))*sign);
 }
 for(double x:{0.,std::ldexp(1.,-126),131008.,65520.})for(double sign:{1.,-1.})checkTandem(x*sign);
 for(double x:{0.,std::ldexp(1.,-126)})for(double sign:{1.,-1.})checkPruned32(x*sign);
 uint64_t typedCases=0;
 for(uint32_t bits=0;bits<65536;bits++)if((bits&0x7c00)!=0x7c00)
  for(uint16_t second:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7bff),uint16_t(0xfbff)}){
   double x=word<_Float16>(uint16_t(bits)),y=word<_Float16>(second);
   double product=static_cast<float>(x*y),addition=static_cast<_Float16>(static_cast<float>(x+y)),zero=static_cast<_Float16>(static_cast<float>(0.0+x));
   typedCases++;
   if(word<uint64_t>(typedProduct(x,y))!=word<uint64_t>(product)||word<uint64_t>(typedAddition(x,y))!=word<uint64_t>(addition)||word<uint64_t>(typedZero(x,y))!=word<uint64_t>(zero))failures++;
  }
 if(typedCases!=507904)return 3;
 std::printf("%llu %llu %llu %llu %llu %llu %llu\\n",(unsigned long long)cases32,(unsigned long long)cases16,(unsigned long long)composedCases,(unsigned long long)pruned32Cases,(unsigned long long)pruned16Cases,(unsigned long long)tandemCases,(unsigned long long)failures);
 return failures?1:0;
}
""")
            subprocess.run(["clang++","-O3","-ffp-contract=off","-std=c++17",str(source),"-o",str(binary)],check=True,capture_output=True)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("Exact subnormal-grid certificates: F32=16777216 F16=2048 mismatches=0",result.stdout)
            print(result.stdout.splitlines()[0])
            self.assertIn("Exact Half-grid tandem certificate: cases=67108866 mismatches=0",result.stdout)
            print(result.stdout.splitlines()[1])
            count32,count16,count_composed,count_pruned32,count_pruned16,count_tandem,failures=map(int,result.stdout.splitlines()[-1].split())
            self.assertEqual(count32,100663296+16376+36)
            self.assertEqual(count16,31743*6+16376+36)
            self.assertEqual(count_composed,31743*6+30719*12)
            self.assertEqual(count_pruned32,100663300)
            self.assertEqual(count_pruned16,3072)
            self.assertEqual(count_tandem,559130)
            self.assertEqual(failures,0)
            print(f"Actual string IEEE certificate: F32={count32} F16={count16} composed={count_composed} prunedF32={count_pruned32} prunedF16={count_pruned16} mismatches={failures}; tandem={count_tandem}; typedPairs=507904")


if __name__=="__main__":unittest.main()

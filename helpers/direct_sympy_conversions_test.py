import ast
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
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
    if isinstance(node,ast.UnaryOp):return "(-double("+cpp(node.operand)+"))" if isinstance(node.op,ast.USub) else cpp(node.operand)
    if isinstance(node,ast.BinOp):
        if isinstance(node.op,ast.Pow):return "std::pow(double("+cpp(node.left)+"), double("+cpp(node.right)+"))"
        op={ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/"}[type(node.op)]
        return "("+cpp(node.left)+" "+op+" "+cpp(node.right)+")"
    if isinstance(node,ast.Compare):return "("+cpp(node.left)+" < "+cpp(node.comparators[0])+")"
    if isinstance(node,ast.Call):
        name=node.func.id
        if name=="Piecewise":
            result="UINT64_C(0)"
            for pair in reversed(node.args):result="("+cpp(pair.elts[1])+" ? "+cpp(pair.elts[0])+" : "+result+")"
            return result
        args=[cpp(x) for x in node.args]
        if name=="Bits64":return "word<uint64_t>(double("+args[0]+"))"
        if name=="Float64":return "word<double>(uint64_t("+args[0]+"))"
        if name=="sqrt":return "std::sqrt(double("+args[0]+"))"
        op={"U64And":"&","U64Or":"|","U64Shr":">>","U64Add":"+","U64Mul":"*"}[name]
        return "(uint64_t("+args[0]+") "+op+" uint64_t("+args[1]+"))"
    raise AssertionError(ast.dump(node))


class ConversionStringTests(unittest.TestCase):
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
            self.assertEqual(rms_half_bound(width,1e-6),2*math.sqrt(width))

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

    def test_constructor_rejects_uncertified_sources(self):
        with self.assertRaises(ValueError):FiniteSource(-math.inf,1)
        with self.assertRaises(ValueError):lower_finite_conversion("X1","R32",None,StringCompiler(),{})

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
        for expression in (product,addition,zero):
            self.assertNotIn("R32(",expression);self.assertNotIn("R16(",expression)
        self.assertGreaterEqual(typed.redundant,4)
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
int main(){
 for(uint64_t bits:{UINT64_C(0),UINT64_C(1),UINT64_C(0x8000000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0xffffffffffffffff)})
  if(wordParity(word<double>(bits))!=((bits>>42)&1))return 4;
 static_assert(FLT_EVAL_METHOD==0);static_assert(std::numeric_limits<double>::is_iec559);
 if(std::fesetround(FE_TONEAREST))return 2;
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
            count32,count16,count_composed,count_pruned32,count_pruned16,count_tandem,failures=map(int,result.stdout.split())
            self.assertEqual(count32,100663296+16376+36)
            self.assertEqual(count16,31743*6+16376+36)
            self.assertEqual(count_composed,31743*6+30719*12)
            self.assertEqual(count_pruned32,100663300)
            self.assertEqual(count_pruned16,3072)
            self.assertEqual(count_tandem,559130)
            self.assertEqual(failures,0)
            print(f"Actual string IEEE certificate: F32={count32} F16={count16} composed={count_composed} prunedF32={count_pruned32} prunedF16={count_pruned16} mismatches={failures}; tandem={count_tandem}; typedPairs=507904")


if __name__=="__main__":unittest.main()

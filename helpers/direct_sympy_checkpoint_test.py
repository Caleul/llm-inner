"""Parity of the working string, not admission of a final compiled artifact."""
import ast
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_checkpoint import CheckpointStrings,f32,rms_half_bound,write_expression
from direct_sympy_strings import StringCompiler,syntax


class CheckpointStringTests(unittest.TestCase):
    def test_streamed_expression_files_keep_literal_bytes_without_full_size_copy(self):
        import tracemalloc
        from direct_sympy_scan_backend import StreamingTextPath
        expression='X1'+(' '*20*1024*1024)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);measurements={}
            for name,write in (
                ('literal-copy',lambda path: path.write_text(expression+'\n',encoding='utf-8')),
                ('prior-scan-backend',lambda path: StreamingTextPath(path).write_text(expression+'\n')),
                ('chunked',lambda path: write_expression(path,expression))):
                tracemalloc.start()
                write(root/name)
                _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                measurements[name]=peak
            from direct_sympy_savepoints import digest_file
            self.assertEqual(digest_file(root/'literal-copy'),digest_file(root/'chunked'))
            self.assertEqual(digest_file(root/'prior-scan-backend'),digest_file(root/'chunked'))
            self.assertLess(measurements['chunked'],4*1024*1024)
            self.assertGreater(measurements['literal-copy'],len(expression))
            self.assertGreater(measurements['prior-scan-backend'],len(expression))
            for text in ('','-0.0','Piecewise((X1, X1 < 0), (X2, True))','X1\n + 2'):
                write_expression(root/'small',text)
                self.assertEqual((root/'small').read_bytes(),(text+'\n').encode('utf-8'))
            print('Expression file allocation parity: '+json.dumps({'characters':len(expression),'byteIdentical':True,'peakAllocatedBytes':measurements}))

    def test_half_products_drop_f32_before_substitution_and_keep_other_boundaries(self):
        from direct_sympy_conversions_test import cpp
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':2,'num_hidden_layers':1}))
            compiler=StringCompiler();builder=CheckpointStrings(root,compiler)
            result=builder.op('*','X1','X2')
            self.assertNotIn('R32',result)
            self.assertEqual(len(compiler.substitution_events),2)
            self.assertTrue(all(e[2:4]==('factor','simplify') for e in compiler.events))
            self.assertEqual(builder.conversions.value_kind(syntax(result)),'f32')
            key=builder.conversions.key(syntax('X2'))
            builder.conversions.half_values.discard(key)
            mixed=builder.op('*','X1','X2')
            self.assertIn('R32',mixed)
            x=1+2**-10;y=1+2**-23
            self.assertNotEqual(struct.pack('d',x*y),struct.pack('d',f32(x*y)))
            builder.conversions.f32_values.discard(key)
            self.assertIn('R32',builder.op('*','X1','X2'))
            self.assertIn('R32',builder.op('+','X1','X2'))
            unlowered=CheckpointStrings(root,StringCompiler(),lower_conversions=False)
            self.assertIn('R32',unlowered.op('*','X1','X2'))
            source=root/'products.cpp';binary=root/'products'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+'double compiled(double X1,double X2){return '+cpp(syntax(result))+';}\n'+'''int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));for(uint16_t o:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x03ff),uint16_t(0x83ff),uint16_t(0x0400),uint16_t(0x8400),uint16_t(0x3555),uint16_t(0xb555),uint16_t(0x3c01),uint16_t(0xbc01),uint16_t(0x7bff),uint16_t(0xfbff)}){double y=word<_Float16>(o);double expected=float(x*y);mismatches+=word<uint64_t>(compiled(x,y))!=word<uint64_t>(expected);cases++;}}std::printf("Early Half product parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=888832 mismatches=0',run.stdout)
            print(run.stdout,end='')

    @unittest.skipUnless(os.environ.get("LLM_INNER_DIRECT_JSON_CHECKPOINT"),"Checkpoint validation fixture not configured")
    def test_dependency_elision_requires_every_input_finite_and_a_strict_cell_bound(self):
        from fractions import Fraction as F
        from direct_sympy_strings import Domain
        from direct_sympy_conversions import ConversionSession
        from direct_sympy_layer_bounds import layer
        checkpoint=os.environ["LLM_INNER_DIRECT_JSON_CHECKPOINT"]
        for signs in ((1,1),(1,-1),(-1,1),(-1,-1)):
            with CheckpointStrings(checkpoint,StringCompiler()) as builder:
                builder.domains={f"X{i+1}":Domain(F(64 if sign>0 else -65504),F(65504 if sign>0 else -64),-4,False) for i,sign in enumerate(signs)}
                builder.conversions=ConversionSession(builder.compiler,builder.domains,input_dtype="f16")
                bounds=layer(builder,"model.layers.0.")
                self.assertTrue(all(0<x<2**-6 for x in bounds["attention"]+bounds["mlp"]))
                self.assertEqual(builder.hidden(0,0),"X1")
                self.assertEqual(builder.hidden(0,1),"X2")
                self.assertEqual(len(builder.elided_updates),4)
                self.assertEqual(len(builder.memo),4)
                self.assertTrue(all("hidden:" in k or "residual:" in k for k in builder.memo))
                self.assertTrue(all(e[2:4]==("factor","simplify") for e in builder.compiler.events))
                builder.layer_bound_cache["model.layers.0."]={"attention":[16.0,16.0],"mlp":[0.0,0.0]}
                self.assertFalse(builder.invisible_layer_update(0,"model.layers.0.","mlp",0,"X1"))
        with CheckpointStrings(checkpoint,StringCompiler()) as builder:
            self.assertFalse(builder.invisible_layer_update(0,"model.layers.0.","attention",0,"X1"))
            self.assertEqual(builder.elided_updates,[])
            builder.config["attention_bias"]=True
            self.assertIsNone(layer(builder,"model.layers.0."))

    @unittest.skipUnless(os.environ.get("LLM_INNER_DIRECT_JSON_CHECKPOINT"),"Checkpoint validation fixture not configured")
    def test_closed_rms_store_preserves_epsilon_and_double_rounding_counterexamples(self):
        from direct_sympy_conversions_test import cpp
        checkpoint=os.environ["LLM_INNER_DIRECT_JSON_CHECKPOINT"]
        with CheckpointStrings(checkpoint,StringCompiler(max_characters=1048576)) as builder:
            self.assertEqual(builder.width,2)
            expression=builder.norm("rounding:proof","model.layers.0.input_layernorm.weight",0,lambda i:f"X{i+1}")
            epsilon=repr(f32(builder.config["rms_norm_eps"]))
            gamma=builder.weight("model.layers.0.input_layernorm.weight",0)
            norm_bound=rms_half_bound(builder.width,f32(builder.config["rms_norm_eps"]))
            self.assertGreater(norm_bound,math.sqrt(builder.width))
            self.assertLess(norm_bound,1.416)
            for name in ("R16(","R32(","sqrt(","CASNumericRegion"):
                self.assertNotIn(name,expression)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"normalization.work.expr";path.write_text(expression+"\n")
            source=Path(directory)/"rms.cpp";binary=Path(directory)/"rms"
            source.write_text("#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n"+
                "double compiled(double X1,double X2){return "+cpp(syntax(path.read_text()))+";}\n"+'''
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0,unsafeEpsilon=0,unsafeHalf=0,unsafeProduct=0;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;
for(uint16_t other:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(3),uint16_t(0x0c01),uint16_t(0x1001),uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x3555),uint16_t(0x7bff),uint16_t(0xfbff),uint16_t(0x0854),uint16_t(0x0d35),uint16_t(0x0c08)}){
double x=word<_Float16>(uint16_t(bits)),y=word<_Float16>(other);
float sum=float(x*x)+float(y*y);double variance=double(sum/2.0f)+double(EPSILON);
uint64_t raw=word<uint64_t>(variance);double withoutEpsilonParity=word<double>((raw+UINT64_C(268435455))&UINT64_C(18446744073172680704));
unsafeEpsilon+=word<uint64_t>(withoutEpsilonParity)!=word<uint64_t>(double(float(variance)));
float inverse=1.0f/std::sqrt(float(variance));double product=x*double(inverse);
double normalized=static_cast<_Float16>(float(product));double expected=static_cast<_Float16>(float(normalized*GAMMA));
mismatches+=std::fabs(product)>NORMBOUND;mismatches+=std::fabs(normalized)>NORMBOUND;
unsafeHalf+=word<uint64_t>(normalized)!=word<uint64_t>(double(static_cast<_Float16>(product)));
uint64_t p=word<uint64_t>(product);double noProductParity=word<double>((p+UINT64_C(268435455))&UINT64_C(18446744073172680704));
unsafeProduct+=word<uint64_t>(normalized)!=word<uint64_t>(double(static_cast<_Float16>(noProductParity)));
mismatches+=word<uint64_t>(compiled(x,y))!=word<uint64_t>(expected);cases++;
}}std::printf("Closed RMS rounding parity: cases=%u mismatches=%u unsafeEpsilon=%u unsafeHalf=%u unsafeProduct=%u\\n",cases,mismatches,unsafeEpsilon,unsafeHalf,unsafeProduct);
return (mismatches||!unsafeEpsilon||!unsafeHalf||!unsafeProduct)?1:0;}
'''.replace("EPSILON",epsilon).replace("GAMMA",gamma).replace("NORMBOUND",repr(norm_bound)))
            built=subprocess.run(["clang++","-O3","-ffp-contract=off","-std=c++17",str(source),"-o",str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("cases=888832 mismatches=0",result.stdout);print(result.stdout,end="")

    def test_unclosed_numeric_producer_is_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':1,'num_hidden_layers':1}))
            builder=CheckpointStrings(directory,StringCompiler())
            publish=unittest.mock.Mock();builder.on_completed=publish
            for primitive in ('R16(X1)','R32(X1)','Silu16(X1)','sqrt(X1)'):
                with patch.object(builder.conversions,'close',return_value=primitive):
                    with self.assertRaisesRegex(ValueError,'closure incomplete.*not published'):
                        builder.producer('fixture:pending',lambda:primitive)
            self.assertEqual(builder.memo,{});self.assertEqual(builder.events,[])
            publish.assert_not_called()

    def test_producer_synchronizes_only_after_closure_and_keeps_whole_root_dtype(self):
        from fractions import Fraction as F
        from direct_sympy_strings import Domain
        from direct_sympy_conversions import ConversionSession
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':1,'num_hidden_layers':1}))
            builder=CheckpointStrings(directory,StringCompiler())
            builder.domains={'X1':Domain(F(-1),F(1),-24,False)}
            builder.conversions=ConversionSession(builder.compiler,builder.domains,input_dtype='f16')
            first=builder.producer('fixture:first',lambda:'R16(R32(X1+X1/2.0))')
            result=builder.producer('fixture:twice',lambda:'R16(R32(('+first+')+('+first+')))')
            self.assertTrue(any(e[-1]=='admitted' for e in builder.compiler.synchronization_events))
            self.assertNotIn('R16(',result);self.assertNotIn('R32(',result)
            self.assertEqual(result.count('Piecewise('),1)
            self.assertEqual(builder.conversions.value_kind(syntax(result)),'half')
            self.assertIn(builder.conversions.key(syntax(result)),builder.conversions.converted_regions)
            self.assertIn(result,builder.conversions.closed_literals)

    @unittest.skipUnless(os.environ.get("LLM_INNER_DIRECT_JSON_CHECKPOINT"),"Checkpoint validation fixture not configured")
    def test_closed_gate_up_projections_preserve_native_half_boundaries(self):
        from direct_sympy_conversions_test import cpp
        checkpoint=os.environ["LLM_INNER_DIRECT_JSON_CHECKPOINT"]
        with CheckpointStrings(checkpoint,StringCompiler(max_characters=1048576)) as builder:
            for projection in ('gate','up'):
                builder.producer('model.layers.0.'+projection+':0',lambda projection=projection:
                    builder.linear('model.layers.0.mlp.'+projection+'_proj.weight',0,lambda i:f'X{i+1}'))
            normalized=builder.norm('proof:test','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
            self.assertNotIn('R16(',normalized);self.assertNotIn('R32(',normalized)
            self.assertNotIn('sqrt(',normalized)
            self.assertGreater(builder.conversions.square_roots_closed,0)
            self.assertEqual(builder.conversions.value_kind(syntax(normalized)),'half')
            self.assertLess(builder.conversions.bounds(syntax(normalized)).maximum,3)
            functions=[];checks=[]
            for projection in ("gate","up"):
                expression=builder.memo["model.layers.0."+projection+":0"]
                with tempfile.TemporaryDirectory() as emitted:
                    path=Path(emitted)/(projection+".work.expr")
                    path.write_text(expression+"\n")
                    expression=path.read_text().strip()
                self.assertNotIn("R16(",expression);self.assertNotIn("R32(",expression)
                self.assertEqual(builder.conversions.value_kind(syntax(expression)),"half")
                functions.append("double "+projection+"(double X1,double X2){return "+cpp(syntax(expression))+";}")
                coefficients=[builder.weight("model.layers.0.mlp."+projection+"_proj.weight",0,i) for i in range(2)]
                # Independently retain the four-lane float reduction and Half
                # store. This compares projections, not a complete coordinate.
                checks.append("{float a=0.0f+float(X1*"+coefficients[0]+");float b=0.0f+float(X2*"+coefficients[1]+");double expected=static_cast<_Float16>((a+b)+(0.0f+0.0f));mismatches+=word<uint64_t>("+projection+"(X1,X2))!=word<uint64_t>(expected);cases++;}")
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"projections.cpp";binary=Path(directory)/"projections"
            source.write_text("""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}
"""+"\n".join(functions)+"""
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;
for(uint16_t other:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7bff),uint16_t(0xfbff)}){
double X1=word<_Float16>(uint16_t(bits)),X2=word<_Float16>(other);
"""+"\n".join(checks)+"""
}}std::printf("Closed gate/up projection parity: cases=%u mismatches=%u finalParity=false\\n",cases,mismatches);return mismatches?1:0;}
""")
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=1015808 mismatches=0',result.stdout);print(result.stdout,end='')

    def test_projection_closure_precedes_next_projection_and_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"config.json").write_text(json.dumps({"model_type":"llama","hidden_size":2,"num_hidden_layers":1}))
            compiler=StringCompiler();builder=CheckpointStrings(directory,compiler)
            order=[]
            def linear(name,row,input_value):
                order.append(("build",name));return "R16(X1)" if "gate" in name else "R16(X2)"
            def closed(expression):
                order.append(("close",expression));return "X1" if "X1" in expression else "X2"
            def product(operation,a,b):
                self.assertEqual(list(builder.memo),["model.layers.0.gate:0","model.layers.0.activation:0","model.layers.0.up:0"])
                self.assertEqual((operation,a,b),("*","X1","X2"))
                order.append(("compose",));return "R32(X1*X2)"
            with patch.object(builder,"linear",side_effect=linear),patch.object(builder.conversions,"close",side_effect=closed),patch.object(builder,"op",side_effect=product):
                builder.gated("model.layers.0.",0,lambda i:f"X{i+1}")
            self.assertEqual([x[0] for x in order],["build","close","close","build","close","compose"])
            self.assertEqual(order[2],('close','Silu16(X1)'))
            self.assertTrue(all(e[2:4]==("factor","simplify") for e in compiler.events))
            builder.memo.clear();order.clear()
            with patch.object(builder,"linear",side_effect=linear),patch.object(builder.conversions,"close",side_effect=ValueError("projection budget")):
                with self.assertRaisesRegex(ValueError,"projection budget"):builder.gated("model.layers.0.",0,lambda i:f"X{i+1}")
            self.assertEqual(len(order),1);self.assertEqual(builder.memo,{})
            order.clear()
            def activation_fails(expression):
                if 'Silu16' in expression:raise ValueError('activation budget')
                return closed(expression)
            with patch.object(builder,"linear",side_effect=linear) as projection,patch.object(builder.conversions,"close",side_effect=activation_fails):
                with self.assertRaisesRegex(ValueError,'activation budget'):builder.gated('model.layers.0.',0,lambda i:f'X{i+1}')
            self.assertEqual(projection.call_count,1)
            self.assertEqual(list(builder.memo),['model.layers.0.gate:0'])

    def test_square_substitutes_once_only_for_certified_finite_half(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"config.json").write_text(json.dumps({"model_type":"llama","hidden_size":2,"num_hidden_layers":1}))
            compiler=StringCompiler()
            builder=CheckpointStrings(directory,compiler)
            result=builder.op("*","X1","X1")
            self.assertEqual(ast.dump(syntax(result)),ast.dump(syntax("X1 ** 2")))
            self.assertEqual(builder.conversions.value_kind(syntax(result)),"f32")
            self.assertTrue(builder.conversions.no_negative_zero(syntax(result)))
            self.assertEqual(len(compiler.substitution_events),1)
            self.assertTrue(all(e[2:4]==("factor","simplify") for e in compiler.events))
            builder.conversions.half_values.clear();builder.conversions.f32_values.clear()
            before=len(compiler.substitution_events)
            result=builder.op("*","X1","X1")
            self.assertEqual(len(compiler.substitution_events)-before,2)
            self.assertIn("R32",result)

    @unittest.skipUnless(os.environ.get("LLM_INNER_DIRECT_JSON_CHECKPOINT"),"Checkpoint validation fixture not configured")
    def test_re_read_working_string_matches_fresh_reference_coordinate(self):
        import torch
        checkpoint=os.environ["LLM_INNER_DIRECT_JSON_CHECKPOINT"]
        compiler=StringCompiler(max_characters=1048576)
        with CheckpointStrings(checkpoint,compiler,lower_conversions=False) as builder:
            expression=builder.coordinate(2)
            self.assertTrue(all(e[2:4]==("factor","simplify") for e in compiler.events))
            self.assertNotIn("CASBoundary",expression)
            self.assertNotIn("X99999999",expression)
        with tempfile.TemporaryDirectory(prefix="direct-sympy-parity-") as directory:
            path=Path(directory)/"coordinate.work.expr"
            path.write_text(expression+"\n")
            # Read the actual string back from disk before executing it.
            tree=syntax(path.read_text())
            program=compile(ast.Expression(tree),str(path),"eval")
            reference=Path(directory)/"reference.json"
            subprocess.run([sys.executable,str(Path(__file__).with_name("capture_direct_json_reference.py")),
                checkpoint,str(reference)],check=True,capture_output=True)
            corpus=json.loads(reference.read_text())
            half=lambda x:struct.unpack("e",struct.pack("e",x))[0]
            silu=lambda x:float(torch.nn.functional.silu(torch.tensor(x,dtype=torch.float16)).item())
            functions={"R32":f32,"R16":half,"sqrt":math.sqrt,"Silu16":silu}
            compared=0
            for row in corpus["cases"]:
                values={f"X{i+1}":struct.unpack("e",struct.pack("H",bits))[0]
                    for i,bits in enumerate(row["inputBits"][0])}
                actual=eval(program,{"__builtins__":{},**functions},values)
                bits="0x"+struct.pack(">d",actual).hex()
                self.assertEqual(bits,row["logitF64Bits"][0][2],row["label"])
                compared+=1
            self.assertEqual(compared,60)
            print(f"Working string parity: {compared} exact cases, position=0 dimension=2; numerical primitives remain, finalParity=false")


if __name__=="__main__":unittest.main()

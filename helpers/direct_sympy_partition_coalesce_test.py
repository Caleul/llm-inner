"""Exact body sharing, complete coverage, signed zero and atomic emission."""
import ast
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import direct_sympy_partition_coalesce as coalesce
from direct_sympy_partition_run import encode
from direct_sympy_input_partitions import interval,split
from direct_sympy_savepoints import atomic,digest_file
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import syntax


def tree(root,depth,expression,directory):
    nodes={'':{'domains':encode(root),'status':'pending'}}
    for level in range(depth):
        for key,node in list(nodes.items()):
            if node['status']!='pending':continue
            axis=list(root)[level%len(root)]
            _,left,right=split(coalesce.decode(node['domains']),axis)
            node.update(status='split',axis=axis)
            for bit,d in (('0',left),('1',right)):nodes[key+bit]={'domains':encode(d),'status':'pending'}
    for key,node in nodes.items():
        if node['status']!='pending':continue
        text=expression(key);p=directory/(key+'.expr');atomic(p,text.encode())
        node.update(status='complete',artifact={'file':p.name,'sha256':digest_file(p),'characters':len(text)})
    return nodes


class CoalesceTests(unittest.TestCase):
    def test_identical_completed_bodies_merge_to_one_with_all_finite_half_native_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);body='0.0 + X1 * 0.5 + X2 * 0.25'
            domains={name:interval(-65504,65504) for name in ('X1','X2')}
            nodes=tree(domains,4,lambda _:body,root)
            _,_,summary=coalesce.groups(root,nodes)
            self.assertEqual(summary['completeRegions'],16)
            self.assertEqual(summary['coalescedRectangles'],1)
            self.assertEqual(summary['coveredInputPatterns'],63488**2)
            self.assertEqual(summary['copiedBodyCharacters'],16*len(body))
            with patch.object(coalesce.sympy,'factor',wraps=coalesce.sympy.factor) as factor,patch.object(coalesce.sympy,'simplify',wraps=coalesce.sympy.simplify) as simplify:
                result=coalesce.emit(root,nodes,root/'final.expr',max_characters=10000)
            self.assertGreaterEqual(factor.call_count,1);self.assertGreaterEqual(simplify.call_count,1)
            self.assertTrue(result['finalArtifactEmitted']);self.assertFalse(result['finalParity'])
            text=(root/'final.expr').read_text();self.assertEqual(text.count('X1'),1)
            self.assertEqual(text.count('X2'),1)
            source=root/'native.cpp';binary=root/'native'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double candidate(double X1,double X2){return '''+cpp(syntax(text))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;
for(unsigned z=0;z<2;z++){double x=word<_Float16>(uint16_t(bits)),y=word<double>(uint64_t(z)<<63);
double expected=(0.0+x*0.5)+y*0.25;cases++;
if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected))return 1;}}
std::printf("Coalesced native parity: cases=%u mismatches=0\\n",cases);}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stderr)
            self.assertIn('cases=126976 mismatches=0',tested.stdout);print(tested.stdout,end='')

    def test_signed_zeros_different_bodies_and_nonadjacent_regions_keep_their_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);domains={'X1':interval(-65504,65504)}
            nodes=tree(domains,1,lambda key:'-0.0' if key=='0' else '0.0',root)
            _,_,summary=coalesce.groups(root,nodes)
            self.assertEqual(summary['distinctExpressions'],2);self.assertEqual(summary['coalescedRectangles'],2)
            coalesce.emit(root,nodes,root/'zeros.expr',max_characters=10000)
            class Lazy(ast.NodeTransformer):
                def visit_Call(self,node):
                    node=self.generic_visit(node)
                    if node.func.id=='And':return ast.BoolOp(op=ast.And(),values=node.args)
                    if node.func.id=='Or':return ast.BoolOp(op=ast.Or(),values=node.args)
                    if node.func.id=='Piecewise':
                        tail=ast.Constant(value=0.0)
                        for pair in reversed(node.args):tail=ast.IfExp(test=pair.elts[1],body=pair.elts[0],orelse=tail)
                        return tail
                    return node
            def evaluate(path,x):
                program=compile(ast.fix_missing_locations(ast.Expression(Lazy().visit(syntax(path.read_text())))),'<coalesced>','eval')
                return eval(program,{'__builtins__':{}},{'X1':x})
            for x in (-65504,-1,-0.0,0.0,1,65504):
                self.assertEqual(struct.pack('>d',evaluate(root/'zeros.expr',x)),struct.pack('>d',-0.0 if x<=0 else 0.0))
            nodes=tree(domains,2,lambda key:'1.0' if key in ('00','11') else '2.0',root)
            _,_,summary=coalesce.groups(root,nodes)
            self.assertEqual(summary['coalescedRectangles'],3)
            coalesce.emit(root,nodes,root/'gaps.expr',max_characters=10000)
            self.assertIn('Or(', (root/'gaps.expr').read_text())
            self.assertEqual(evaluate(root/'gaps.expr',0),2)
            self.assertEqual(evaluate(root/'gaps.expr',65504),1)
            nodes=tree(domains,1,lambda key:'0.0' if key=='0' else 'Piecewise((X1, 1.0 / X1 > 0.0), (-X1, True))',root)
            coalesce.emit(root,nodes,root/'lazy.expr',max_characters=10000)
            self.assertEqual(evaluate(root/'lazy.expr',0),0)
            self.assertEqual(evaluate(root/'lazy.expr',1),1)

    def test_pending_corrupt_and_budgeted_artifacts_never_replace_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);domains={'X1':interval(-65504,65504)}
            nodes=tree(domains,1,lambda _:'X1',root);final=root/'final.expr';final.write_text('existing')
            with self.assertRaisesRegex(ValueError,'budget'):coalesce.emit(root,nodes,final,max_characters=1)
            self.assertEqual(final.read_text(),'existing')
            nodes['0']['status']='pending'
            with self.assertRaisesRegex(ValueError,'Unfinished'):coalesce.emit(root,nodes,final,max_characters=10000)
            self.assertEqual(final.read_text(),'existing')
            nodes['0']['status']='complete';(root/'0.expr').write_text('X1 + 1.0')
            with self.assertRaisesRegex(ValueError,'integrity'):coalesce.groups(root,nodes)
            self.assertEqual(final.read_text(),'existing')


if __name__=='__main__':unittest.main()

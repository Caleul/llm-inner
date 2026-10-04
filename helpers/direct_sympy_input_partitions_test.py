"""Complete-domain coverage, strict state identity and real emitted region parity."""
from fractions import Fraction as F
import argparse
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from direct_sympy_strings import Domain,syntax
from direct_sympy_conversions_test import cpp
from direct_sympy_input_partitions import MAX_RANK,rank,value,interval,split,validate_domains
import direct_sympy_partition_run as runner
from direct_sympy_savepoints import atomic,digest_file


class InputPartitionTests(unittest.TestCase):
    def test_checkpoint_update_cuts_preserve_complete_coverage_and_bound_geometric_fanout(self):
        root=runner.encode({name:interval(-65504,65504) for name in ('X1','X2','X3')})
        model=argparse.Namespace(layers=2)
        bounds={'attention':[0.0068,0.0024,0.0],'mlp':[0.0003,0.0003,0.0]}
        with patch('direct_sympy_layer_bounds.layer',return_value=bounds):
            tree,geometry=runner.seed_update_cells(model,root,9)
        self.assertEqual(geometry['thresholdRanks'],{'X1':rank(32),'X2':rank(16),'X3':1})
        self.assertEqual(geometry['seededAxes'],['X1','X2'])
        self.assertEqual(geometry['leafRegions'],9)
        self.assertEqual(runner.audit_tree(tree,root),(0,63488**3))
        for name in ('X1','X2'):
            ranges=sorted({tuple(node['domains'][name]) for node in tree.values() if node['status']=='pending'})
            for index in range(-MAX_RANK,MAX_RANK+1):self.assertEqual(sum(a<=index<=b for a,b in ranges),1)
            self.assertEqual(sum(a<=0<=b for a,b in ranges),1)
        with patch('direct_sympy_layer_bounds.layer',return_value=None):
            unchanged,geometry=runner.seed_update_cells(model,root,9)
        self.assertEqual(len(unchanged),1);self.assertEqual(geometry['seededAxes'],[])
        for bound in (None,-1,float('inf'),8):self.assertIsNone(runner.update_threshold(bound))
        self.assertEqual(runner.update_threshold(2**-25),rank(2**-12))
        source=runner.decode(root)
        for cut in (True,-MAX_RANK-1,MAX_RANK,1.5):
            with self.assertRaises(ValueError):split(source,'X1',cut)
        broken=json.loads(json.dumps(tree));broken['']['cut']+=1
        with self.assertRaisesRegex(ValueError,'coverage'):runner.audit_tree(broken,root)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture required')
    def test_seeded_controller_emits_all_four_external_boxes_before_central_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            args=argparse.Namespace(checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],state=directory,
                resume=False,seed_update_cells=True,seed_max_regions=9,dimension=2,max_attempts=4,
                max_characters=1048576,cas_characters=8388608,max_paths=64,region_seconds=30,
                min_values=32,total_characters=100000)
            self.assertEqual(runner.run(args),1)
            state=json.loads((Path(directory)/'frontier.json').read_text())
            self.assertEqual(state['updateCellGeometry']['thresholdRanks'],{'X1':rank(32),'X2':rank(16)})
            self.assertEqual([attempt['region'] for attempt in state['attempts']],['00','011','110','1111'])
            self.assertTrue(all(attempt['complete'] and attempt['elidedUpdates']==4 for attempt in state['attempts']))
            covered,pending=runner.audit_tree(state['tree'],state['root'])
            self.assertEqual(covered,4*(MAX_RANK-rank(32)+1)*(MAX_RANK-rank(16)+1))
            self.assertEqual(covered+pending,63488**2)
            self.assertFalse(state['finalArtifactEmitted']);self.assertFalse(state['finalParity'])
            self.assertFalse((Path(directory)/'coordinate.expr').exists())
            # Scheduling can change without weakening numerical identity.
            args.resume=True;args.seed_update_cells=False;args.frontier_order='lexical';args.max_attempts=1
            with patch.object(runner,'compile_region',return_value={'complete':False,'stop':'test budget','seconds':0}):
                self.assertEqual(runner.run(args),1)
            resumed=json.loads((Path(directory)/'frontier.json').read_text())
            self.assertEqual(resumed['coveredInputPatterns'],covered)
            self.assertEqual(resumed['attempts'][-1]['region'],'010')
            print(f'Checkpoint update-cell geometry: regions=4 covered={covered} lost=0 overlaps=0 finalParity=false')

    def test_frontier_visits_largest_pending_domain_with_stable_ties_and_signed_zeros(self):
        tree={'deep':{'status':'pending','domains':{'X1':[10,11]}},
              'large':{'status':'pending','domains':{'X1':[-10,10]}},
              'finished':{'status':'complete','domains':{'X1':[-100,100]}},
              'split':{'status':'split','domains':{'X1':[-1000,1000]}}}
        before=json.loads(json.dumps(tree))
        self.assertEqual(runner.next_region(tree),'large')
        self.assertEqual(runner.next_region(tree,'lexical'),'deep')
        self.assertEqual(tree,before)
        tree={'b':{'status':'pending','domains':{'X1':[-1,1]}},'a':{'status':'pending','domains':{'X1':[1,4]}}}
        self.assertEqual(runner.cardinality(tree['b']['domains']),4)
        self.assertEqual(runner.next_region(tree),'a')
        tree['a']['status']='complete';self.assertEqual(runner.next_region(tree),'b')
        tree['b']['status']='complete';self.assertIsNone(runner.next_region(tree))
        with self.assertRaisesRegex(ValueError,'Unknown frontier order'):runner.next_region(tree,'bad')

    def test_all_finite_half_patterns_have_exact_disjoint_coverage_including_both_zeros(self):
        domains={'X1':interval(-65504,65504),'X2':interval(-65504,65504)}
        tree={'':{'domains':runner.encode(domains),'status':'pending'}}
        for depth in range(6):
            for key,node in list(tree.items()):
                if node['status']!='pending':continue
                axis='X'+str(depth%2+1);guard,left,right=split(runner.decode(node['domains']),axis)
                node.update(status='split',axis=axis)
                self.assertIn('<=',guard)
                tree[key+'0']={'domains':runner.encode(left),'status':'pending'}
                tree[key+'1']={'domains':runner.encode(right),'status':'pending'}
        root=runner.encode(domains);covered,pending=runner.audit_tree(tree,root)
        self.assertEqual((covered,pending),(0,63488**2))
        ranges=[node['domains']['X1'] for node in tree.values() if node['status']=='pending']
        ranges=sorted(set(map(tuple,ranges)))
        for bits in range(65536):
            if bits&0x7c00==0x7c00:continue
            x=struct.unpack('>e',struct.pack('>H',bits))[0]
            r=rank(x)
            self.assertEqual(F(x),value(r))
            self.assertEqual(sum(a<=r<=b for a,b in ranges),1)
        for node in tree.values():
            if node['status']=='pending':node['status']='complete'
        self.assertEqual(runner.audit_tree(tree,root),(63488**2,0))
        leaf=next(node for node in tree.values() if node['status']=='complete')
        leaf['domains']['X1'][0]+=1
        with self.assertRaisesRegex(ValueError,'coverage'):runner.audit_tree(tree,root)
        print(f'Partition domain proof: HalfPatterns=63488 vectorPatterns={63488**2} lost=0 overlaps=0')

    def test_invalid_precision_and_zero_certificates_are_rejected(self):
        original={'X1':interval(-65504,65504)}
        for domain in (Domain(F(-1),F(1),-10),Domain(F(0),F(1),-24,True),Domain(F(-65505),F(1),-24),Domain(F(1),F(1)+F(2)**-100,-24)):
            with self.assertRaises(ValueError):validate_domains({'X1':domain},original)
        with self.assertRaises(ValueError):validate_domains({},original)
        self.assertEqual(interval(32,65504).quantum,-5)
        self.assertEqual(rank(-0.0),rank(0.0))
        with self.assertRaises(ValueError):value(MAX_RANK+1)

    def test_incomplete_dispatch_and_corrupt_resume_never_replace_final_output(self):
        checkpoint=os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT')
        if not checkpoint:self.skipTest('Checkpoint fixture required')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);final=root/'coordinate.expr';final.write_text('keep-existing-final')
            domains={'X1':interval(-65504,65504),'X2':interval(-65504,65504)}
            tree={'':{'domains':runner.encode(domains),'status':'pending'}}
            with self.assertRaisesRegex(ValueError,'Unfinished'):runner.combine(root,tree,final,100000)
            self.assertEqual(final.read_text(),'keep-existing-final')
            args=argparse.Namespace(checkpoint=checkpoint,state=str(root/'state'),resume=False,dimension=2,max_attempts=1,
                max_characters=1000,cas_characters=1000,max_paths=2,region_seconds=1,min_values=32,total_characters=100000)
            with patch.object(runner,'compile_region',return_value={'complete':False,'stop':'test budget','seconds':0}):
                self.assertEqual(runner.run(args),1)
            manifest=Path(args.state)/'frontier.json';saved=manifest.read_bytes();state=json.loads(saved)
            self.assertEqual(state['coveredInputPatterns'],0)
            self.assertEqual(state['unfinishedInputPatterns'],63488**2)
            args.resume=True;args.dimension=1
            with self.assertRaisesRegex(ValueError,'Incompatible'):runner.run(args)
            self.assertEqual(manifest.read_bytes(),saved)
            # Test final readback/integrity and atomic character budget with
            # a synthetic complete cover, independently of model parity.
            left,right=split(domains,'X1')[1:]
            tree={'':{'domains':runner.encode(domains),'status':'split','axis':'X1'}}
            for key,d in (('0',left),('1',right)):
                p=root/(key+'.expr');atomic(p,b'X1 + X2')
                tree[key]={'domains':runner.encode(d),'status':'complete',
                    'artifact':{'file':p.name,'sha256':digest_file(p)}}
            with self.assertRaisesRegex(ValueError,'budget'):runner.combine(root,tree,final,1)
            self.assertEqual(final.read_text(),'keep-existing-final')
            result=runner.combine(root,tree,final,100000)
            self.assertEqual(result['coveredInputPatterns'],63488**2)
            self.assertNotIn('CompileValue',final.read_text())
            self.assertFalse(result['parityVerified'])
            (root/'0.expr').write_text('X1 - X2')
            with self.assertRaisesRegex(ValueError,'integrity'):runner.combine(root,tree,final,100000)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture required')
    def test_external_worker_timeout_is_bounded_and_does_not_publish_partial_state(self):
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        domains={name:interval(-65504,65504) for name in ('X1','X2')}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'coordinate.expr';path.write_text('existing-complete-artifact')
            started=time.monotonic()
            result=runner.compile_region(checkpoint,2,domains,path,max_characters=1048576,
                cas_characters=8388608,max_paths=64,max_seconds=1)
            self.assertFalse(result['complete']);self.assertIn('worker wall-clock budget',result['stop'])
            self.assertLess(time.monotonic()-started,10)
            self.assertEqual(path.read_text(),'existing-complete-artifact')
            with patch.object(runner.subprocess,'run',return_value=argparse.Namespace(returncode=1,stderr='semantic failure',stdout='')):
                with self.assertRaisesRegex(ValueError,'semantic failure'):
                    runner.compile_region(checkpoint,2,domains,path,max_characters=1048576,
                        cas_characters=8388608,max_paths=64,max_seconds=1)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture required')
    def test_complete_emitted_large_region_matches_fresh_model_coordinate(self):
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(1)
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'coordinate.expr'
            domains={name:interval(32,65504) for name in ('X1','X2')}
            report=runner.compile_region(checkpoint,2,domains,path,max_characters=1048576,
                cas_characters=8388608,max_paths=64,max_seconds=60)
            self.assertTrue(report['complete']);self.assertEqual(report['elidedUpdates'],4)
            text=path.read_text()
            for name in ('CompileValue','R16(','R32(','Silu16(','sqrt('):self.assertNotIn(name,text)
            rng=random.Random(192)
            ranks=[rank(32),rank(32)+1,rank(64)-1,rank(64),rank(64)+1,MAX_RANK-1,MAX_RANK]
            ranks.extend(rng.randrange(rank(32),MAX_RANK+1) for _ in range(17))
            samples=[(a,b) for a in ranks for b in ranks]
            reference=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float16,attn_implementation='eager').eval()
            matrix=torch.tensor([[float(value(a)),float(value(b))] for a,b in samples],dtype=torch.float16).unsqueeze(1)
            with torch.inference_mode():expected=reference(inputs_embeds=matrix,use_cache=False).logits[:,0,2].double()
            source=root/'native.cpp';binary=root/'native'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfenv>
#include <cmath>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double candidate(double X1,double X2){return '''+cpp(syntax(text))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned a,b;
while(std::scanf("%u %u",&a,&b)==2){auto result=candidate(word<_Float16>(uint16_t(a)),word<_Float16>(uint16_t(b)));
std::printf("%016llx\\n",(unsigned long long)word<uint64_t>(result));}}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            output=subprocess.run([str(binary)],input=''.join(f'{a} {b}\n' for a,b in samples),check=True,capture_output=True,text=True)
            actual=output.stdout.splitlines()
            self.assertEqual(len(actual),len(samples))
            for i,v in enumerate(expected.tolist()):self.assertEqual(actual[i],struct.pack('>d',v).hex(),samples[i])
            durable=os.environ.get('LLM_INNER_DIRECT_REGION_OUTPUT')
            if durable:
                target=Path(durable);target.parent.mkdir(parents=True,exist_ok=True)
                atomic(target,path.read_bytes());atomic(str(target)+'.report.json',json.dumps(report,indent=2).encode())
            print(f'Emitted full region coordinate: cases={len(samples)} mismatches=0 characters={len(text)} position=0 dimension=2 fullInputCoverage=false')


if __name__=='__main__':unittest.main()

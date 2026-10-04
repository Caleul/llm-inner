import copy,json,os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import argparse
from direct_sympy_cover_regions import cover,live_identity,promote_tree
from direct_sympy_partition_run import audit_tree,encode,run
from direct_sympy_input_partitions import interval,value,split
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_savepoints import atomic,canonical


class CoverRegionTests(unittest.TestCase):
    def test_intersections_preserve_old_leaves_and_both_zeros_without_duplicate_coverage(self):
        domains={name:interval(value(-8),value(8)) for name in ('X1','X2')};root=encode(domains)
        _,left,right=split(domains,'X1',-1)
        tree={'':{'domains':root,'status':'split','axis':'X1','cut':-1},
            '0':{'domains':encode(left),'status':'complete','artifact':{'file':'old.expr','sha256':'old'}},
            '1':{'domains':encode(right),'status':'pending'}}
        original=copy.deepcopy(tree);rectangle={'X1':[-2,4],'X2':[-3,3]}
        promoted,keys,added=promote_tree(tree,root,rectangle,{'file':'new.expr','sha256':'new'})
        self.assertEqual(tree,original);self.assertEqual(promoted['0'],original['0'])
        self.assertEqual(added,48);self.assertEqual(audit_tree(promoted,root),(192,132))
        inputs=list(range(-8,9))+[0] # two distinct IEEE zeros
        for x in inputs:
            for y in inputs:
                matches=[n for n in promoted.values() if n['status']!='split' and n['domains']['X1'][0]<=x<=n['domains']['X1'][1] and n['domains']['X2'][0]<=y<=n['domains']['X2'][1]]
                self.assertEqual(len(matches),1)
                expected=x<0 or (-2<=x<=4 and -3<=y<=3)
                self.assertEqual(matches[0]['status']=='complete',expected)
        repeated,keys,added=promote_tree(promoted,root,rectangle,{'file':'unused.expr'})
        self.assertEqual((keys,added),([],0));self.assertEqual(repeated,promoted)
        zero,_,added=promote_tree(tree,root,{'X1':[0,0],'X2':[0,0]},{'file':'zero.expr'})
        self.assertEqual(added,4);self.assertEqual(audit_tree(zero,root),(148,176))
        for rectangle in ({'X1':[-9,4],'X2':[-3,3]}, {'X1':[0,0]}, {'X1':[4,-2],'X2':[-3,3]}):
            with self.assertRaises(ValueError):promote_tree(tree,root,rectangle,{})
        print('Cover intersection proof: vectorPatterns=324 added=48 lost=0 overlaps=0 signedZeroPatterns=4')

    def setup_state(self,directory):
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        root=encode(CheckpointStrings(checkpoint,StringCompiler()).domains)
        state={'identity':live_identity(checkpoint,2),'root':root,
            'tree':{'':{'domains':root,'status':'pending'}},'attempts':[],
            'coveredInputPatterns':0,'unfinishedInputPatterns':63488**2,'totalInputPatterns':63488**2,
            'finalArtifactEmitted':False,'finalParity':False}
        manifest=Path(directory)/'frontier.json';atomic(manifest,canonical(state))
        return checkpoint,manifest,state

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_incompatible_budget_failed_and_parity_failed_candidates_never_mutate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint,manifest,state=self.setup_state(directory)
            rectangle={'X1':[-31743,-20480],'X2':[-8192,8192]}
            state['identity']['sources'][next(iter(state['identity']['sources']))]='changed'
            atomic(manifest,canonical(state));before=manifest.read_bytes()
            with patch('direct_sympy_cover_regions.compile_region',side_effect=AssertionError('must reject before compiling')):
                with self.assertRaisesRegex(ValueError,'Incompatible'):cover(checkpoint,directory,rectangle)
            self.assertEqual(manifest.read_bytes(),before)
            checkpoint,manifest,state=self.setup_state(directory);before=manifest.read_bytes()
            with patch('direct_sympy_cover_regions.compile_region',return_value={'complete':False,'stop':'test budget'}):
                result=cover(checkpoint,directory,rectangle)
                self.assertEqual(result['addedInputPatterns'],0)
            self.assertEqual(manifest.read_bytes(),before)
            def fake_compile(checkpoint,dimension,domains,path,**_):
                import hashlib
                path.write_text('0.0')
                return {'complete':True,'artifact':{'sha256':hashlib.sha256(b'0.0').hexdigest(),'characters':3}}
            with patch('direct_sympy_cover_regions.compile_region',side_effect=fake_compile),patch('direct_sympy_cover_regions.verify_region',return_value={'mismatches':1}):
                with self.assertRaisesRegex(ValueError,'parity failed'):cover(checkpoint,directory,rectangle)
            self.assertEqual(manifest.read_bytes(),before)
            self.assertEqual(list(Path(directory).glob('cover-*.expr')),[])

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_real_central_region_is_compiled_validated_admitted_and_resumes_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint,manifest,state=self.setup_state(directory)
            rectangle={'X1':[-31743,-20480],'X2':[-8192,8192]}
            result=cover(checkpoint,directory,rectangle,random_cases=64)
            self.assertEqual(result['addedInputPatterns'],184571904)
            self.assertEqual(result['parity']['mismatches'],0)
            self.assertEqual(result['sharedArtifact']['characters'],11)
            saved=json.loads(manifest.read_text())
            self.assertEqual(audit_tree(saved['tree'],saved['root']),(184571904,3846154240))
            self.assertFalse(saved['finalArtifactEmitted']);self.assertFalse(saved['finalParity'])
            before=manifest.read_bytes();again=cover(checkpoint,directory,rectangle,random_cases=0)
            self.assertEqual(again['addedInputPatterns'],0);self.assertEqual(manifest.read_bytes(),before)
            args=argparse.Namespace(checkpoint=checkpoint,state=directory,resume=True,dimension=2,
                max_attempts=1,max_characters=1000,cas_characters=1000,max_paths=2,region_seconds=1,
                min_values=32,total_characters=100000)
            with patch('direct_sympy_partition_run.compile_region',return_value={'complete':False,'stop':'test budget','seconds':0}):
                self.assertEqual(run(args),1)
            self.assertEqual(json.loads(manifest.read_text())['coveredInputPatterns'],184571904)
            print('Fresh central cover parity: patterns=184571904 mismatches=0 compatibleResume=true finalParity=false')


if __name__=='__main__':unittest.main()

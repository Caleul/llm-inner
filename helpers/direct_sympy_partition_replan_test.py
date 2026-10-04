"""Geometry can survive source changes; compiled evidence cannot."""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch
from direct_sympy_partition_run import replan,audit_tree,encode,cardinality
from direct_sympy_input_partitions import split,interval

class GeometryReplanTests(unittest.TestCase):
    def fixture(self,cut=None):
        domains={'X1':interval(-F(1),F(1)),'X2':interval(-F(1),F(1))}
        root=encode(domains);_,left,right=split(domains,'X1',cut)
        identity={'version':1,'dimension':2,'position':0,'checkpoint':{'config.json':'config','model.safetensors':'weights'},'sources':{'compiler.py':'old'},'referenceBackend':{'name':'old CPU'}}
        previous={'identity':identity,'root':root,'tree':{'':{'domains':root,'status':'split','axis':'X1'},'0':{'domains':encode(left),'status':'complete','artifact':{'file':'NEVER_READ.expr','sha256':'untrusted'}},'1':{'domains':encode(right),'status':'pending','proofs':'NEVER_REUSE'}},'attempts':[{'complete':True,'expression':'NEVER_REUSE'}],'finalArtifactEmitted':True,'finalParity':True}
        if cut is not None:previous['tree']['']['cut']=cut
        current=copy.deepcopy(identity);current['sources']={'compiler.py':'new'};current['referenceBackend']={'name':'new CPU'}
        return previous,current,root

    def test_old_proofs_artifacts_and_success_are_removed_but_geometry_is_retained(self):
        previous,current,root=self.fixture();before=copy.deepcopy(previous)
        with patch('direct_sympy_partition_run.digest_file',side_effect=AssertionError('Old artifact must never be read')):
            result=replan(previous,current,root)
        self.assertEqual(previous,before)
        self.assertEqual(result['identity'],current)
        self.assertEqual(result['attempts'],[])
        self.assertFalse(result['finalArtifactEmitted']);self.assertFalse(result['finalParity'])
        self.assertEqual(audit_tree(result['tree'],root),(0,cardinality(root)))
        self.assertEqual(result['tree']['']['axis'],'X1')
        for key in ('0','1'):
            self.assertEqual(result['tree'][key]['status'],'pending')
            self.assertEqual(set(result['tree'][key]),{'status','domains'})
        result['tree']['0']['domains']['X1'][0]+=1
        self.assertEqual(previous,before)
        custom,current,root=self.fixture(-14336)
        with patch('direct_sympy_partition_run.digest_file',side_effect=AssertionError('Old artifact must never be read')):
            replanned=replan(custom,current,root)
        self.assertEqual(replanned['tree']['']['cut'],-14336)
        self.assertEqual(audit_tree(replanned['tree'],root),(0,cardinality(root)))
        self.assertNotIn('artifact',replanned['tree']['0'])
        print('Geometry replan: numericalResultsReused=false lost=0 overlaps=0')

    def test_checkpoint_coordinate_domain_and_corrupt_geometry_are_rejected(self):
        previous,current,root=self.fixture()
        for field,value in [('version',2),('dimension',1),('position',1),('checkpoint',{'config.json':'changed'})]:
            changed=copy.deepcopy(current);changed[field]=value
            with self.assertRaises(ValueError):replan(previous,changed,root)
        changed=copy.deepcopy(root);changed['X1'][0]+=1
        with self.assertRaises(ValueError):replan(previous,current,changed)
        for mutate in (lambda tree:tree['0']['domains']['X1'].__setitem__(0,0),lambda tree:tree.update({'junk':tree['1']}),lambda tree:tree.pop('1')):
            broken=copy.deepcopy(previous);mutate(broken['tree'])
            with self.assertRaises(ValueError):replan(broken,current,root)

if __name__=='__main__':unittest.main()

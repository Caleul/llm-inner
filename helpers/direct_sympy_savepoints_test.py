import json
import hashlib
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from fractions import Fraction as F

import direct_sympy_savepoints as saves
from direct_sympy_savepoints import ProducerSavepoints
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import Domain,StringCompiler
from direct_sympy_conversions import ConversionSession
from direct_sympy_strings import syntax
from direct_sympy_conversions_test import cpp


class SavepointTests(unittest.TestCase):
    def model(self,root):
        checkpoint=root/'checkpoint';checkpoint.mkdir(exist_ok=True)
        if not (checkpoint/'config.json').exists():
            (checkpoint/'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':1,'num_hidden_layers':1}))
            (checkpoint/'weights.safetensors').write_bytes(b'identity-only fixture')
        model=CheckpointStrings(checkpoint,StringCompiler())
        model.domains={'X1':Domain(F(-1),F(1),-24,False)}
        model.conversions=ConversionSession(model.compiler,model.domains,input_dtype='f16')
        return model

    def first(self,model):return model.producer('fixture:root',lambda:'R16(R32(X1+X1/2.0))')
    def second(self,model):return model.producer('fixture:next',lambda:'R16(R32(('+self.first(model)+')*0.5))')

    def test_continuous_and_resumed_rounding_strings_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=self.model(root)
            with ProducerSavepoints(root/'state',original,0) as store:
                original.on_completed=store.save
                with patch.object(saves,'syntax',side_effect=AssertionError('reparsed new certified producer')):
                    first=self.first(original)
            with self.assertRaisesRegex(ValueError,'writer is closed'):store.save(original)
            original.on_completed=None
            resumed=self.model(root)
            with ProducerSavepoints(root/'state',resumed,0) as store:
                self.assertEqual(store.restore(resumed),1)
                self.assertIn(first,resumed.conversions.closed_literals)
                self.assertEqual(self.first(resumed),first)
                final=self.second(resumed)
                self.assertEqual(final,self.second(original))
                self.assertTrue(resumed.conversions.numeric_envelopes)
                self.assertTrue(original.conversions.numeric_envelopes)
                self.assertNotIn('CASNumericRegion',final)
                self.assertEqual(len(resumed.events),2)
            source=root/'proof.cpp';binary=root/'proof'
            source.write_text('''
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfenv>
#include <cstdio>
#include <initializer_list>
template<class T,class U>T word(U value){T result;static_assert(sizeof(T)==sizeof(U));std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1){return '''+cpp(syntax(final))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned bits=0;bits<=0x3c00;bits++)for(unsigned sign:{0u,0x8000u}){
double x=word<_Float16>(uint16_t(bits|sign));
double first=static_cast<_Float16>(float(float(x)+float(x/2.0)));
double expected=static_cast<_Float16>(float(first*0.5));cases++;
if(word<uint64_t>(candidate(x))!=word<uint64_t>(expected))return 1;
}std::printf("Native resumed rounding: cases=%u mismatches=0\\n",cases);}
''')
            compiled=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=30722 mismatches=0',result.stdout)
            print(result.stdout,end='')
            payload=json.loads((root/'state/frontier.json').read_text())['payload']
            self.assertNotIn('expression',payload['records'][0])
            self.assertIsNotNone(payload['records'][0]['bounds'])
            self.assertEqual(payload['records'][0]['kind'],'half')

    def test_restore_compacts_validated_dependencies_and_keeps_exact_structural_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=self.model(root)
            with ProducerSavepoints(root/'state',original,0) as store:
                original.on_completed=store.save
                first=self.first(original);second=self.second(original)
                original.producer('fixture:alias',lambda:first)
            original.on_completed=None
            resumed=self.model(root)
            with ProducerSavepoints(root/'state',resumed,0) as store:
                with patch.object(saves,'syntax',wraps=saves.syntax) as parse:
                    self.assertEqual(store.restore(resumed),3)
                self.assertLess(max(len(c.args[0]) for c in parse.call_args_list),len(second))
                self.assertEqual(resumed.conversions.closed_literal_characters,sum(map(len,resumed.conversions.closed_literals)))
                for expression in (first,second):
                    self.assertEqual(resumed.conversions.closed_literal_keys[expression],resumed.conversions.key(syntax(expression)))
                self.assertEqual(self.second(resumed),second)
                expected=original.producer('fixture:third',lambda:'R16(R32(('+second+')*0.5))')
                actual=resumed.producer('fixture:third',lambda:'R16(R32(('+second+')*0.5))')
                self.assertEqual(actual,expected)
            # A budget eviction of session literal caches cannot invalidate
            # keys composed from earlier validated records during restoration.
            evicted=self.model(root)
            restore=ConversionSession.restore_compact_literal
            def drop_cached_literals(session,*args):
                result=restore(session,*args)
                session.closed_literals.clear();session.closed_literal_keys.clear();session.closed_literal_pure.clear();session.closed_literal_characters=0
                return result
            with ProducerSavepoints(root/'state',evicted,0) as store:
                with patch.object(ConversionSession,'restore_compact_literal',drop_cached_literals):self.assertEqual(store.restore(evicted),3)
            self.assertEqual(evicted.conversions.value_kind(syntax(second)),'half')
            # Recompute the manifest checksum to test validation itself,
            # rather than merely testing the outer integrity checksum.
            path=root/'state/frontier.json';envelope=json.loads(path.read_text());record=envelope['payload']['records'][1]
            bad='('+second+') + X2';digest=hashlib.sha256(bad.encode()).hexdigest()
            (root/'state/objects'/(digest+'.expr')).write_text(bad)
            record['digest']=digest;record['characters']=len(bad)
            envelope['integrity']=hashlib.sha256(saves.canonical(envelope['payload'])).hexdigest();path.write_bytes(saves.canonical(envelope))
            fresh=self.model(root)
            with ProducerSavepoints(root/'state',fresh,0) as store:
                with self.assertRaisesRegex(ValueError,'non-input dependency'):store.restore(fresh)
            self.assertFalse(fresh.memo);self.assertFalse(fresh.conversions.closed_literals)

    def test_published_records_reuse_only_the_identical_literal_and_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=self.model(root)
            with ProducerSavepoints(root/'state',model,0) as store:
                model.on_completed=store.save;self.first(model)
                before=(root/'state/frontier.json').read_bytes()
                with patch.object(saves,'syntax',side_effect=AssertionError('reparsed completed producer')),patch.object(saves,'save_expression',side_effect=AssertionError('rewrote identical producer')):
                    store.save(model)
                self.assertEqual((root/'state/frontier.json').read_bytes(),before)
                first_record=json.loads(before)['payload']['records'][0]
                object_path=root/'state/objects'/(first_record['digest']+'.expr')
                object_path.unlink()
                store.save(model)
                self.assertEqual(saves.digest_file(object_path),first_record['digest'])
                # A changed expression under the same name must be re-certified.
                model.memo['fixture:root']='-0.0'
                with patch.object(saves,'syntax',wraps=saves.syntax) as parse:store.save(model)
                self.assertEqual(parse.call_count,1)
                record=json.loads((root/'state/frontier.json').read_text())['payload']['records'][0]
                self.assertNotEqual((root/'state/frontier.json').read_bytes(),before)
                self.assertFalse(record['noNegativeZero'])
                # A new session cannot inherit the old dtype/domain certificates.
                model.conversions=ConversionSession(model.compiler,model.domains)
                with patch.object(saves,'syntax',wraps=saves.syntax) as parse:store.save(model)
                self.assertEqual(parse.call_count,1)
            restored=self.model(root)
            with ProducerSavepoints(root/'state',restored,0) as store:
                self.assertEqual(store.restore(restored),1)
                with patch.object(saves,'syntax',side_effect=AssertionError('reparsed restored producer')):store.save(restored)

    def test_corruption_and_checkpoint_changes_are_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=self.model(root)
            with ProducerSavepoints(root/'state',original,0) as store:
                original.on_completed=store.save;self.first(original)
            fresh=self.model(root)
            with ProducerSavepoints(root/'state',fresh,1) as store:
                with self.assertRaisesRegex(ValueError,'Incompatible'):store.restore(fresh)
                self.assertFalse(fresh.memo)
            object_path=next((root/'state/objects').glob('*.expr'))
            data=object_path.read_bytes();object_path.write_bytes(data+b'0')
            with ProducerSavepoints(root/'state',fresh,0) as store:
                with self.assertRaisesRegex(ValueError,'integrity'):store.restore(fresh)
                self.assertFalse(fresh.memo)
            object_path.write_bytes(data)
            (root/'checkpoint/weights.safetensors').write_bytes(b'changed checkpoint')
            with ProducerSavepoints(root/'state',fresh,0) as store:
                with self.assertRaisesRegex(ValueError,'Incompatible'):store.restore(fresh)
                self.assertFalse(fresh.memo)

    def test_failed_manifest_publication_preserves_previous_frontier_and_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=self.model(root)
            with ProducerSavepoints(root/'state',model,0) as store:
                model.on_completed=store.save;self.first(model)
                with self.assertRaises(BlockingIOError):ProducerSavepoints(root/'state',model,0)
                original_atomic=saves.atomic
                def fail_manifest(path,data):
                    if Path(path).name=='frontier.json':raise OSError('simulated publication failure')
                    return original_atomic(path,data)
                with patch.object(saves,'atomic',side_effect=fail_manifest):
                    with self.assertRaises(OSError):self.second(model)
            fresh=self.model(root)
            with ProducerSavepoints(root/'state',fresh,0) as store:
                self.assertEqual(store.restore(fresh),1)
                self.assertNotIn('fixture:next',fresh.memo)


if __name__=='__main__':unittest.main()

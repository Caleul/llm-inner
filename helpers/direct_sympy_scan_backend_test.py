import itertools
import json
from pathlib import Path
import random
import re
import tempfile
import time
import unittest
from direct_sympy_scan_backend import EquivalentScans,NECESSARY,StreamingTextPath,install
import direct_sympy_savepoints_test as fixtures
from direct_sympy_savepoints import ProducerSavepoints


class EquivalentScanTests(unittest.TestCase):
    def test_producer_releases_source_before_numeric_closure_without_changing_events(self):
        import weakref
        class TrackedString(str):pass
        fixture=fixtures.SavepointTests();results=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for optimized in (False,True):
                model=fixture.model(root);observed=[];source=[]
                close=model.conversions.close
                def build():
                    text=TrackedString('R16(R32(X1 + 0.5))');source.append(weakref.ref(text));return text
                def tracked_close(text):
                    observed.append(source[0]() is not None);return close(text)
                model.conversions.close=tracked_close
                if optimized:
                    with install():result=model.producer('fixture:lifetime',build)
                else:result=model.producer('fixture:lifetime',build)
                self.assertEqual(observed,[not optimized])
                self.assertIsNone(source[0]())
                results.append((result,model.events,model.compiler.events))
        self.assertEqual(*results)

    def test_parenthesized_producer_compacts_before_allocating_a_second_giant_wrapper(self):
        import tracemalloc
        from fractions import Fraction as F
        from direct_sympy_strings import StringCompiler,Domain,syntax
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        leaf='Float64('+(' '*16*1024**2)+'Bits64(X1))'
        expression='R16('+leaf+' + 0.0)'
        results={};peaks={}
        for mode in ('literal-wrapper','compact-wrapper'):
            compiler=StringCompiler(max_characters=64*1024**2)
            compiler.register_completed_region(leaf,domains,syntax('Float64(Bits64(X1))'),word_closed=True)
            backend=EquivalentScans()
            tracemalloc.start()
            results[mode]=compiler.stabilize('('+expression+')',domains) if mode=='literal-wrapper' else backend.stabilize_enveloped(compiler,expression,domains)
            peaks[mode]=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
            self.assertTrue(all(e[2:4]==('factor','simplify') for e in compiler.events))
            self.assertNotIn('CASStableRegion',results[mode])
        self.assertEqual(results['literal-wrapper'],results['compact-wrapper'])
        self.assertLess(peaks['compact-wrapper'],peaks['literal-wrapper']*0.65)
        print('Producer wrapper allocation parity: '+json.dumps({'characters':len(expression),'byteIdentical':True,'peakAllocatedBytes':peaks}))

    def test_envelope_compaction_keeps_input_refining_barriers_limits_and_producer_restoration(self):
        from fractions import Fraction as F
        from direct_sympy_strings import StringCompiler,Domain,syntax
        from direct_sympy_checkpoint import CheckpointStrings
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        leaf='Float64('+(' '*1048576)+'Bits64(X1))'
        for expression in ('R16('+leaf+' * 0.5)',
                           'Piecewise((R16('+leaf+'),X1>0),(R16('+leaf+'),True))',
                           'U64And(Bits64('+leaf+'),9223372036854775808)'):
            outputs=[]
            for optimized in (False,True):
                compiler=StringCompiler(max_characters=8*1048576)
                compiler.register_completed_region(leaf,domains,syntax('Float64(Bits64(X1))'),word_closed=True)
                backend=EquivalentScans()
                outputs.append(backend.stabilize_enveloped(compiler,expression,domains) if optimized else compiler.stabilize('('+expression+')',domains))
                if 'Piecewise' in expression:self.assertEqual(backend.producer_envelopes,[])
            self.assertEqual(*outputs)
        compiler=StringCompiler(max_characters=len(leaf))
        with self.assertRaisesRegex(ValueError,'String expression budget exceeded'):
            EquivalentScans().stabilize_enveloped(compiler,leaf,domains)
        original=CheckpointStrings.producer
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with install():
                outer=CheckpointStrings.producer
                with install():self.assertIsNot(CheckpointStrings.producer,original)
                self.assertIs(CheckpointStrings.producer,outer)
                raise RuntimeError('fixture')
        self.assertIs(CheckpointStrings.producer,original)

    def test_large_region_search_preserves_first_offsets_near_matches_and_unicode(self):
        backend=EquivalentScans();size=1048576
        regions=['a'*(size-1)+'b','字'*(size-1)+'末',('ab'*size)+'END']
        for region in regions:
            near=region[:-1]+'!'
            for text in (region,near,near+region,'x'+region+region,region+'x'+near):
                for start in (-1,0,1,len(text)//2,len(text),len(text)+1):
                    self.assertEqual(backend.literal_find(text,region,start),text.find(region,start))
        self.assertGreater(backend.literal_region_searches,0)
        self.assertGreater(backend.literal_region_candidates,0)

    def test_large_compaction_preserves_identifier_boundaries_and_restores_method(self):
        from fractions import Fraction as F
        from direct_sympy_strings import StringCompiler,Domain,syntax
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        region='Float64('+(' '*1048576)+'Bits64(X1))'
        compiler=StringCompiler(max_characters=8*1048576);compiler.register_completed_region(region,domains,syntax(region))
        expression='other'+region+' + '+region+' + '+region+' (X1)'
        original=StringCompiler.compact_regions
        expected=compiler.compact_regions(expression,compiler.context(domains))
        with install() as backend:
            self.assertEqual(compiler.compact_regions(expression,compiler.context(domains)),expected)
            self.assertGreater(backend.literal_region_searches,0)
            with install():self.assertEqual(compiler.compact_regions(expression,compiler.context(domains)),expected)
        self.assertIs(StringCompiler.compact_regions,original)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with install():raise RuntimeError('fixture')
        self.assertIs(StringCompiler.compact_regions,original)

    def test_streaming_ascii_diagnostics_preserve_bytes_and_character_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            text=('Float64(X1)\n'*100000)+'\n';root=Path(directory)
            for encoding in ('utf-8','utf-16'):
                for newline in (None,'\r\n'):
                    expected=root/'normal';actual=StreamingTextPath(root/'streamed')
                    self.assertEqual(actual.write_text(text,encoding=encoding,newline=newline),expected.write_text(text,encoding=encoding,newline=newline))
                    self.assertEqual(actual.read_bytes(),expected.read_bytes())
            text='字X1';self.assertEqual(actual.write_text(text),len(text));self.assertEqual(actual.read_text(),text)

    def test_search_predicates_preserve_matches_flags_errors_and_unicode(self):
        backend=EquivalentScans()
        words=['CASStableRegion1','CASNumericRegion7','CASBoundary3','Piecewise(','R16 (','R32(','sqrt(','Silu16(']
        surroundings=['',' ','_','a','字','\u0301','²','\n','(',')']
        for pattern in NECESSARY:
            for word,left,right in itertools.product(words,surroundings,surroundings):
                text=left+word+right
                for flags in (0,re.I,re.ASCII):
                    expected=re.search(pattern,text,flags);actual=backend.search(pattern,text,flags)
                    self.assertEqual(None if expected is None else (expected.span(),expected.group(),expected.pos),None if actual is None else (actual.span(),actual.group(),actual.pos))
            for flags in (None,0.0):
                def outcome(call):
                    try:return call(pattern,'absent',flags)
                    except Exception as error:return type(error)
                self.assertEqual(outcome(backend.search),outcome(re.search))
        self.assertIsNone(backend.search(r'\bCASStableRegion[0-9]+\b','Float64(X1)'))
        self.assertGreater(backend.negative_searches,0)

    def test_identifier_findall_preserves_unicode_and_overlapping_candidates(self):
        backend=EquivalentScans();random.seed(513)
        words=['X1','X11','_x','CASNumericRegion1()','abc','字X1','X1字','aX1','X1_','()','\u0301','²',' ']
        texts=['ababa','X1X1','X1()X1()','_x_x',*(''.join(random.choices(words,k=20)) for _ in range(400))]
        patterns=[r'\bX1\b',r'\bX11\b',r'\b_x\b',r'\babc\b',r'\bX1\(\)',r'\bCASNumericRegion1\(\)',r'(X[0-9]+)',r'X.*1']
        for pattern,text,flags in itertools.product(patterns,texts,(0,re.I,re.ASCII)):
            self.assertEqual(backend.findall(pattern,text,flags),re.findall(pattern,text,flags))
        self.assertEqual(backend.findall(re.compile(r'\bX1\b'),'X1'),['X1'])
        self.assertGreater(backend.identifier_scans,0)

    def test_compilation_and_old_savepoint_are_byte_identical_without_identity_changes(self):
        fixture=fixtures.SavepointTests()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=fixture.model(root)
            with ProducerSavepoints(root/'state',original,0) as store:
                original.on_completed=store.save;first=fixture.first(original)
            baseline=(root/'state/frontier.json').read_bytes()
            original.on_completed=None
            expected=fixture.second(original)
            with install():
                restored=fixture.model(root)
                with ProducerSavepoints(root/'state',restored,0) as store:
                    self.assertEqual(store.restore(restored),1)
                    self.assertEqual((root/'state/frontier.json').read_bytes(),baseline)
                    self.assertEqual(fixture.second(restored),expected)
                    self.assertEqual(restored.memo['fixture:root'],first)
            self.assertIs(__import__('direct_sympy_strings').re,re)

    def test_large_negative_and_identifier_scans_have_identical_results(self):
        backend=EquivalentScans();text=('Float64(U64And(Bits64(X1), 9223372036854775807)) + '*400000)+'X999999999'
        pattern=r'\bX999999999\b';started=time.monotonic();expected=re.findall(pattern,text);normal=time.monotonic()-started
        started=time.monotonic();actual=backend.findall(pattern,text);fast=time.monotonic()-started
        self.assertEqual(actual,expected)
        self.assertEqual(backend.search(r'\bCASStableRegion[0-9]+\b',text),re.search(r'\bCASStableRegion[0-9]+\b',text))
        print(json.dumps({'characters':len(text),'normalIdentifierSeconds':normal,'equivalentIdentifierSeconds':fast,'identical':True}))


if __name__=='__main__':unittest.main()

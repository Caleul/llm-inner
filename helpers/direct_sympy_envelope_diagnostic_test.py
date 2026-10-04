import sys
from fractions import Fraction as F
from types import SimpleNamespace
import unittest

from direct_sympy_conversions import ConversionSession
from direct_sympy_envelope_diagnostic import capture_envelopes
from direct_sympy_strings import Domain,StringCompiler


class EnvelopeDiagnosticTests(unittest.TestCase):
    def test_rejected_accounting_matches_actual_restoration_without_changing_compilation(self):
        session=ConversionSession(StringCompiler(),{n:Domain(-F(1,64),F(1,64),-24,False) for n in ('X1','X2')},input_dtype='f16')
        activation=session.close('Silu16(X1)');up=session.close('R16(X2/3.0)')
        gated=session.compose_closed('R16(R32(X999999998 * X999999999))',{'X999999998':activation,'X999999999':up})
        template='R16(R32(X999999998 / 3.0))';bindings={'X999999998':gated}
        session.compiler.max_characters=3000
        reports=[];before=dict(session.closed_literals);previous=sys.gettrace()
        with capture_envelopes(SimpleNamespace(memo={'gated':gated}),reports):
            with self.assertRaisesRegex(ValueError,'Closed numeric envelope exceeds'):
                session.compose_closed(template,bindings)
        self.assertIs(sys.gettrace(),previous)
        self.assertEqual(session.closed_literals,before)
        self.assertEqual(len(reports),1)
        report=reports[0]
        gated_alias=next(alias for alias in report['aliases'] if alias['producer']=='gated')
        self.assertEqual(gated_alias['characters'],len(gated))
        self.assertEqual(gated_alias['occurrences'],4)
        self.assertGreater(report['expandedCharacters'],3000)
        self.assertLess(report['compactCharacters'],3000)
        session.compiler.max_characters=1000000
        observed=[]
        with capture_envelopes(SimpleNamespace(memo={'gated':gated}),observed):
            result=session.compose_closed(template,bindings)
        self.assertEqual(observed,[])
        self.assertEqual(len(result),report['expandedCharacters'])
        self.assertNotRegex(result,r'CASNumericRegion|R16\(|R32\(')
        self.assertEqual(result,session.compose_closed(template,bindings))

    def test_unrelated_exception_propagates_and_existing_trace_is_restored(self):
        previous=sys.gettrace()
        def old_trace(*_):return None
        reports=[]
        sys.settrace(old_trace)
        try:
            with self.assertRaisesRegex(RuntimeError,'unrelated'):
                with capture_envelopes(SimpleNamespace(memo={}),reports):raise RuntimeError('unrelated')
            self.assertIs(sys.gettrace(),old_trace)
            self.assertEqual(reports,[])
        finally:sys.settrace(previous)


if __name__=='__main__':unittest.main()

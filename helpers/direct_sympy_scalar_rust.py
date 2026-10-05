"""Straight-line Rust with incorporated weights and fully defined numeric bodies.

Scalar locals are executable producers, never runtime graph descriptors. Only
last-position outputs are emitted. Candidate admission requires native bit parity.
"""
import ast
from fractions import Fraction as F
import os
import tempfile
from pathlib import Path
import sys

from direct_sympy_rust import RustExpression,emit_scalar
from direct_sympy_strings import StringCompiler,Domain,syntax
from direct_sympy_conversions import ConversionSession,FiniteSource,lower_finite_conversion


class ScalarRustExpression(RustExpression):
    def __init__(self,inputs,locals):
        super().__init__(len(inputs));self.bindings={name:f'input[{i}]' for i,name in enumerate(inputs)}
        self.bindings.update({name:'v'+name[1:] for name in locals})

    def emit(self,node,expected=None):
        if isinstance(node,ast.Name):
            if node.id not in self.bindings:raise ValueError('Unresolved scalar: '+node.id)
            if expected not in (None,'f64'):raise ValueError('Scalar word/type mismatch')
            return 'f64',self.bindings[node.id]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('R16','R32','sqrt','Silu16','Exp32'):
            if len(node.args)!=1 or expected not in (None,'f64'):raise ValueError('Invalid numerical scalar call')
            if node.func.id=='sqrt':raise ValueError('Root requires its original immediate F32 boundary')
            operand=node.args[0]
            if node.func.id=='R32' and isinstance(operand,ast.Call) and isinstance(operand.func,ast.Name) and operand.func.id=='sqrt':
                if len(operand.args)!=1:raise ValueError('Unary root required')
                body='root_binary32('+self.emit(operand.args[0],'f64')[1]+')'
            else:body=self.emit(operand,'f64')[1]
            name={'R16':'round_binary16','R32':'round_binary32','sqrt':'root_binary32',
                'Silu16':'activation_half','Exp32':'exponential_small_negative'}[node.func.id]
            return 'f64',f'{name}({body})'
        return super().emit(node,expected)


def numerical_bodies(compiler):
    # Generic finite-F64 rounding includes subnormal, signed-zero and overflow
    # arms. No hardware cast, rounding-mode abstraction or response table.
    domain={'X1':Domain(-F(sys.float_info.max),F(sys.float_info.max),-1074,False)}
    bodies=[]
    for kind,name in [('R16','round_binary16'),('R32','round_binary32')]:
        text=lower_finite_conversion('X1',kind,FiniteSource(-sys.float_info.max,sys.float_info.max),compiler,domain)
        bodies.append(f'fn {name}(input: f64) -> f64 {{ let input = [input]; '+emit_scalar(text,1)+' }')
    # Existing all-Half certificate: final Half bits agree throughout ±1/16.
    session=ConversionSession(compiler,{'X1':Domain(-F(1,16),F(1,16),-24,False)},input_dtype='f16')
    closed=session.close('Silu16(X1)')
    bodies.append('fn activation_half(input: f64) -> f64 { let input = [input]; '+emit_scalar(closed,1)+' }')
    # Universal ordered 5/5 root certificate covers all positive finite F32
    # mantissa/parity pairs. Exact power-of-two scaling covers all exponents.
    from direct_sympy_sqrt import CONSTANT_TERM,PARTIAL_FRACTIONS
    lines=['fn root_binary32(value: f64) -> f64 {',
        ' let word=value.to_bits();',
        ' let exponent=(word >> 52) & 2047u64;',
        ' let mantissa=f64::from_bits((word & 4503599627370495u64) | 4607182418800017408u64);',
        ' let z=mantissa - 1.5;',f' let p={CONSTANT_TERM!r}f64;']
    for residue,pole in PARTIAL_FRACTIONS:lines.append(f' let p=p + ({residue!r}f64 / (z + {pole!r}f64));')
    lines+=[' let p=p * (1.0 + (((exponent + 1u64) & 1u64) as f64) * 0.4142135623730951);',
        ' let rounded=(p + 536870912.0) - 536870912.0;',
        ' let target=(exponent + 1023u64) >> 1;',
        ' let adjustment=target.wrapping_add(18446744073709550593u64).wrapping_mul(4503599627370496u64);',
        ' f64::from_bits(rounded.to_bits().wrapping_add(adjustment))','}']
    bodies.append('\n'.join(lines))
    # Ordered CPU range-reduction polynomial on [-.34,0]: exponent is zero.
    # All intermediate F32 boundaries are implemented by the closed word body.
    from direct_sympy_checkpoint import f32
    coefficients=[0.000198527617612853646278381,0.00139304355252534151077271,
        0.00833336077630519866943359,0.0416664853692054748535156,0.166666671633720397949219,0.5]
    lines=['fn exponential_small_negative(x: f64) -> f64 {',f' let p={f32(coefficients[0])!r}f64;']
    for c in coefficients[1:]:lines.append(f' let p=round_binary32(p * x + {f32(c)!r}f64);')
    lines+=[' let squared=round_binary32(x * x);',' let tail=round_binary32(squared * p + x);',
        ' if x < -0.0000000298023223876953125 { round_binary32(1.0 + tail) } else { 1.0 }','}']
    bodies.append('\n'.join(lines))
    return '\n\n'.join(bodies)+'\n'


def write_scalar_rust_candidate(path,programs,certificates,*,width,vocab,context,compiler,max_bytes=2*1024**3):
    if set(programs)!=set(range(1,context+1)) or set(certificates)!=set(programs):raise ValueError('Incomplete variable-length architecture')
    for length,program in programs.items():
        program.validate()
        if len(program.inputs)!=length*width or len(program.outputs)!=vocab:raise ValueError('Invalid direct-vector shape')
        certificate=certificates[length]
        if certificate.get('inputDomain')!='All finite Half values, including both signed zeros':raise ValueError('Finite architecture proof required')
        for layer in certificate['layers']:
            if layer.get('gateAbsMaximum',float('inf'))>1/16:raise ValueError('Activation domain exceeds elementary certificate')
            if 2*layer['scoreAbsMaximum']>.34:raise ValueError('Exponential domain exceeds elementary certificate')
    source_bytes=sum(len(text.encode()) for program in programs.values()
        for text in [*(value for _,value in program.definitions),*program.outputs])
    if source_bytes>max_bytes:raise ValueError('Accumulated source expression and condition budget exceeded')
    numeric=numerical_bodies(compiler)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    total=0;total_producers=0
    try:
        with os.fdopen(descriptor,'w') as stream:
            def write(text):
                nonlocal total
                total+=len(text.encode())
                if total>max_bytes:raise ValueError('Accumulated Rust expression and condition budget exceeded')
                stream.write(text)
            write(numeric)
            write('\npub fn predict_next_token(input: &[f64]) -> [f64; '+str(vocab)+'] {\n match input.len() {\n')
            for length,program in sorted(programs.items()):
                emitter=ScalarRustExpression(program.inputs,[name for name,_ in program.definitions])
                write(f'  {length*width} => {{\n')
                for name,text in program.definitions:
                    write('   let v'+name[1:]+': f64 = '+emitter.emit(syntax(text),'f64')[1]+';\n')
                    total_producers+=1
                write('   ['+', '.join(emitter.emit(syntax(text),'f64')[1] for text in program.outputs)+']\n  },\n')
            write('  _ => panic!("Unsupported input length"),\n }\n}\n')
        os.replace(temporary,path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True);raise
    return {'bytes':total,'scalarProducers':total_producers,'lengths':context,'logits':vocab,'finalParity':False}

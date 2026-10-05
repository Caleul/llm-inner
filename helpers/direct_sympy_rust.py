"""Emit closed mathematical strings as direct Rust next-token expressions.

No numerical algorithm or algebraic rewrite is introduced here. Remaining
conversion/activation/root/exponential functions and compiler placeholders are
rejected. Candidate emission is separate from native bit-parity admission.
"""
import ast
import os
from pathlib import Path
import re
import struct
import json
import subprocess
import tempfile

from direct_sympy_strings import syntax


class RustExpression:
    def __init__(self,input_count):
        self.input_count=input_count

    def emit(self,node,expected=None):
        if isinstance(node,ast.Constant):
            if type(node.value)is bool:kind,text='bool',str(node.value).lower()
            elif expected=='u64':
                if type(node.value)is not int:raise ValueError('Integer word operand required')
                kind,text='u64',str(node.value&((1<<64)-1))+'u64'
            else:
                bits=struct.unpack('>Q',struct.pack('>d',float(node.value)))[0]
                kind,text='f64',f'f64::from_bits(0x{bits:016x})'
        elif isinstance(node,ast.Name):
            match=re.fullmatch(r'X([1-9][0-9]*)',node.id)
            if match is None or not 1<=int(match[1])<=self.input_count:
                raise ValueError('Residual or out-of-domain input: '+node.id)
            kind,text='f64',f'input[{int(match[1])-1}]'
        elif isinstance(node,ast.UnaryOp):
            if isinstance(node.op,ast.Not):
                _,body=self.emit(node.operand,'bool');kind,text='bool',f'(!({body}))'
            elif expected=='u64' and isinstance(node.operand,ast.Constant) and type(node.operand.value)is int:
                value=-node.operand.value if isinstance(node.op,ast.USub) else node.operand.value
                return self.emit(ast.Constant(value),'u64')
            else:
                _,body=self.emit(node.operand,'f64');kind,text='f64',f'(-({body}))' if isinstance(node.op,ast.USub) else body
        elif isinstance(node,ast.BinOp):
            if isinstance(node.op,ast.Pow):
                # Literal scale constants need no runtime exponentiation.
                if isinstance(node.left,ast.Constant) and isinstance(node.right,(ast.Constant,ast.UnaryOp)):
                    exponent=ast.literal_eval(node.right)
                    if type(exponent)is not int or not -1074<=exponent<=1023:raise ValueError('Unsupported constant exponent')
                    return self.emit(ast.Constant(node.left.value**exponent),expected)
                # Original inputs are finite Half: their squares are exact
                # F64 products. General powers require a separate certificate.
                if isinstance(node.left,ast.Name) and isinstance(node.right,ast.Constant) and node.right.value==2:
                    _,body=self.emit(node.left,'f64');kind,text='f64',f'(({body}) * ({body}))'
                else:raise ValueError('Residual uncertified exponentiation')
            else:
                op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(node.op)]
                _,left=self.emit(node.left,'f64');_,right=self.emit(node.right,'f64')
                kind,text='f64',f'(({left}) {op} ({right}))'
        elif isinstance(node,ast.Compare):
            if isinstance(node.left,ast.Constant):
                right_kind,right=self.emit(node.comparators[0]);left_kind,left=self.emit(node.left,right_kind)
            else:
                left_kind,left=self.emit(node.left);right_kind,right=self.emit(node.comparators[0],left_kind)
            op={ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>=',ast.Eq:'==',ast.NotEq:'!='}[type(node.ops[0])]
            kind,text='bool',f'(({left}) {op} ({right}))'
        elif isinstance(node,ast.BoolOp):
            values=[self.emit(arg,'bool')[1] for arg in node.values]
            kind,text='bool','('+(' && ' if isinstance(node.op,ast.And) else ' || ').join(values)+')'
        elif isinstance(node,ast.Call):
            name=node.func.id
            if name=='Piecewise':
                if not node.args or any(not isinstance(arm,ast.Tuple) or len(arm.elts)!=2 for arm in node.args):
                    raise ValueError('Explicit Piecewise arms required')
                last=node.args[-1]
                if not isinstance(last.elts[1],ast.Constant) or last.elts[1].value is not True:
                    raise ValueError('Piecewise requires a complete final arm')
                kind,text=self.emit(last.elts[0],expected)
                for arm in reversed(node.args[:-1]):
                    _,condition=self.emit(arm.elts[1],'bool');_,value=self.emit(arm.elts[0],kind)
                    text=f'(if {condition} {{ {value} }} else {{ {text} }})'
            elif name in ('And','Or','Not'):
                if name=='Not':
                    if len(node.args)!=1:raise ValueError('Unary Not required')
                    kind,text='bool','(!('+self.emit(node.args[0],'bool')[1]+'))'
                else:
                    if not node.args:raise ValueError('Nonempty logical operation required')
                    kind,text='bool','('+(' && ' if name=='And' else ' || ').join(self.emit(arg,'bool')[1] for arg in node.args)+')'
            elif name in ('Bits64','Float64','F64FromU64','U64FromF64'):
                if len(node.args)!=1:raise ValueError('Unary word conversion required')
                source='f64' if name in ('Bits64','U64FromF64') else 'u64'
                _,body=self.emit(node.args[0],source)
                kind='u64' if name in ('Bits64','U64FromF64') else 'f64'
                text={'Bits64':f'({body}).to_bits()','Float64':f'f64::from_bits({body})',
                    'F64FromU64':f'(({body}) as f64)','U64FromF64':f'(({body}) as u64)'}[name]
            elif name in ('U64Add','U64Sub','U64Mul','U64Div','U64And','U64Or','U64Xor','U64Shr','U64Shl'):
                if len(node.args)!=2:raise ValueError('Binary word operation required')
                _,left=self.emit(node.args[0],'u64');_,right=self.emit(node.args[1],'u64');kind='u64'
                if name in ('U64Add','U64Sub','U64Mul'):
                    method={'U64Add':'wrapping_add','U64Sub':'wrapping_sub','U64Mul':'wrapping_mul'}[name]
                    text=f'({left}).{method}({right})'
                elif name in ('U64Shr','U64Shl'):
                    op='>>' if name=='U64Shr' else '<<'
                    text=f'(if ({right}) < 64u64 {{ ({left}) {op} ({right}) }} else {{ 0u64 }})'
                else:
                    op={'U64Div':'/','U64And':'&','U64Or':'|','U64Xor':'^'}[name]
                    text=f'(({left}) {op} ({right}))'
            else:raise ValueError('Residual numerical primitive or compiler placeholder: '+name)
        else:raise ValueError('Unsupported closed expression node')
        if expected is not None and kind!=expected:raise ValueError(f'Expected {expected}, got {kind}')
        return kind,text


def emit_scalar(expression,input_count):
    return RustExpression(input_count).emit(syntax(expression),'f64')[1]


def write_rust_candidate(path,by_length,*,width,vocab,context,max_bytes=2*1024**3):
    """One direct vector for every supported length; no runtime intermediates.

    Write atomically only when all lengths and all logits are closed. The
    caller must compile/read this candidate and prove native parity before
    publishing it as the final function.
    """
    if any(type(n)is not int or n<1 for n in (width,vocab,context,max_bytes)):
        raise ValueError('Positive discovered dimensions and budget required')
    if set(by_length)!=set(range(1,context+1)) or any(len(row)!=vocab for row in by_length.values()):
        raise ValueError('Incomplete next-token vector or supported input domain')
    if sum(len(value.encode('utf-8')) for row in by_length.values() for value in row)>max_bytes:
        raise ValueError('Accumulated expression and condition budget exceeded')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);total=0
    descriptor,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            def write(text):
                nonlocal total
                total+=len(text.encode('utf-8'))
                if total>max_bytes:raise ValueError('Rust candidate exceeds accumulated output budget')
                stream.write(text)
            write(f'// Input values are finite IEEE Half values widened exactly to f64.\npub fn predict_next_token(input: &[f64]) -> [f64; {vocab}] {{\n    match input.len() {{\n')
            for length in range(1,context+1):
                write(f'        {length*width} => [\n')
                for expression in by_length[length]:write('            '+emit_scalar(expression,length*width)+',\n')
                write('        ],\n')
            write('        _ => panic!("Unsupported input length"),\n    }\n}\n')
        os.replace(temp,path)
    except BaseException:
        Path(temp).unlink(missing_ok=True);raise
    return {'bytes':total,'logits':vocab,'lengths':context,'finalParity':False}


def validate_rust_candidate(path,reference,*,width,vocab,context,rustc='rustc',timeout=300):
    """Compile the actual candidate and compare last-position native bits.

    The validation executable owns the corpus; it is never appended to the
    standalone model function. Every supported length must be represented.
    """
    if (reference.get('width'),reference.get('vocab'),reference.get('context'))!=(width,vocab,context):
        raise ValueError('Reference shape differs from the discovered architecture')
    if reference.get('torch')!='2.12.1' or reference.get('backend')!='cpu-arm64-eager':
        raise ValueError('Original native numerical target required')
    lines=['include!('+json.dumps(str(Path(path).resolve()))+');','fn main() {'];lengths=set();comparisons=0
    for case in reference['cases']:
        rows=case['inputBits'];length=len(rows)
        if not 1<=length<=context or any(len(row)!=width for row in rows):raise ValueError('Invalid reference input shape')
        words=[]
        for row in rows:
            for bits in row:
                if type(bits)is not int or not 0<=bits<=65535 or bits&0x7c00==0x7c00:raise ValueError('Finite Half input required')
                value=struct.unpack('<e',struct.pack('<H',bits))[0]
                word=struct.unpack('>Q',struct.pack('>d',value))[0];words.append(f'f64::from_bits(0x{word:016x})')
        expected=case['logitF64Bits'][-1]
        if len(expected)!=vocab:raise ValueError('Incomplete last-position reference')
        wanted=','.join(f'0x{int(word,16):016x}u64' for word in expected)
        lines.append('for (actual, expected) in predict_next_token(&['+','.join(words)+']).iter().zip(['+wanted+']) { assert_eq!(actual.to_bits(), expected); }')
        lengths.add(length);comparisons+=vocab
    if lengths!=set(range(1,context+1)):raise ValueError('Every supported length needs native parity cases')
    lines.append('}')
    with tempfile.TemporaryDirectory(prefix='llm-inner-rust-parity-') as directory:
        source=Path(directory)/'parity.rs';binary=Path(directory)/'parity'
        source.write_text('\n'.join(lines)+'\n')
        try:
            built=subprocess.run([rustc,'--edition=2021','-O',str(source),'-o',str(binary)],capture_output=True,text=True,timeout=timeout)
        except (OSError,subprocess.TimeoutExpired) as error:
            raise ValueError('Rust candidate compilation unavailable or exceeded its deadline: '+str(error)) from error
        if built.returncode:raise ValueError('Rust candidate compilation failed: '+built.stderr[-4000:])
        try:
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=timeout)
        except (OSError,subprocess.TimeoutExpired) as error:
            raise ValueError('Rust candidate parity execution unavailable or exceeded its deadline: '+str(error)) from error
        if result.returncode:raise ValueError('Rust last-position bit parity failed: '+result.stderr[-4000:])
    return {'comparisons':comparisons,'lengths':sorted(lengths),'mismatches':0,'candidateParity':True}

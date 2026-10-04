"""Compiler-only IEEE sign projections. Magnitudes and numeric order stay intact."""
import ast
import copy
import math
import struct
from direct_sympy_strings import syntax

SIGN=1<<63


def call(name,*args):
    return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=list(args),keywords=[])


def project(node,session):
    """An unsigned sign field, including signed zeros, for a finite scalar.

    Finite multiplication/division inherit XOR of operand signs even on
    underflow. Unsigned addition of two isolated sign fields implements that
    XOR modulo 2**64. No sum, magnitude or rounding is reassociated.
    Unknown/nonfinite nodes remain an exact Bits64 mask of the whole scalar.
    """
    key=session.key(node)
    if key in session.sign_projections:return copy.deepcopy(session.sign_projections[key])
    def fallback():return call('U64And',call('Bits64',copy.deepcopy(node)),ast.Constant(SIGN))
    bounds=session.bounds(node)
    if bounds is None:return fallback()
    literal=session.constant(node)
    if literal is not None and math.isfinite(literal):
        return ast.Constant(struct.unpack('Q',struct.pack('d',literal))[0]&SIGN)
    if bounds.minimum>0 or (bounds.minimum>=0 and session.no_negative_zero(node)):
        return ast.Constant(0)
    if bounds.maximum<0:return ast.Constant(SIGN)
    if isinstance(node,ast.UnaryOp):
        if isinstance(node.op,ast.UAdd):return project(node.operand,session)
        if isinstance(node.op,ast.USub) and session.bounds(node.operand) is not None:
            return call('U64And',call('U64Add',project(node.operand,session),ast.Constant(SIGN)),ast.Constant(SIGN))
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Div)):
        left,right=session.bounds(node.left),session.bounds(node.right)
        if left is not None and right is not None:
            if isinstance(node.op,ast.Div) and right.minimum<=0<=right.maximum:return fallback()
            return call('U64And',call('U64Add',project(node.left,session),project(node.right,session)),ast.Constant(SIGN))
    if isinstance(node,ast.Call) and len(node.args)==1:
        if node.func.id in ('R16','R32') and session.bounds(node.args[0]) is not None:
            return project(node.args[0],session)
        if node.func.id=='Silu16':
            from direct_sympy_silu import supported
            if supported(ast.unparse(node.args[0]),session):return project(node.args[0],session)
    return fallback()


def is_signed_zero(node,zero_names=()):
    """Structural admission for persisted +/-zero projections only."""
    if isinstance(node,ast.Name):return node.id in zero_names
    if isinstance(node,ast.Constant):return type(node.value) is float and node.value==0.0
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        return is_signed_zero(node.operand,zero_names)
    if not isinstance(node,ast.Call) or node.func.id!='Float64' or len(node.args)!=1:return False
    def flag(word):
        if isinstance(word,ast.Constant):return type(word.value) is int and word.value in (0,SIGN)
        if not isinstance(word,ast.Call):return False
        if word.func.id=='Bits64' and len(word.args)==1:return is_signed_zero(word.args[0],zero_names)
        if len(word.args)!=2:return False
        a,b=word.args
        if word.func.id=='U64And' and isinstance(b,ast.Constant) and type(b.value) is int and b.value in (0,SIGN):return True
        if word.func.id in ('U64Or','U64Add'):return flag(a) and flag(b)
        return False
    return flag(node.args[0])

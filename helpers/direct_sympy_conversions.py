"""Finite IEEE conversions as mathematical strings of elementary operations.

Bits64/Float64 are bit reinterpretations; U64And/U64Or/U64Shr/U64Add are
unsigned integer operations, not numerical conversion helpers. Piecewise
retains every classification. Constructors require a finite-source certificate.
No R16/R32 or rounding-mode node is retained in their returned syntax.
"""
from dataclasses import dataclass
import ast
import math
import struct
from fractions import Fraction

from direct_sympy_strings import StringCompiler
from direct_sympy_strings import syntax
from direct_sympy_strings import quantum


@dataclass(frozen=True)
class FiniteSource:
    minimum: float
    maximum: float
    quantum: int|None=None

    def __post_init__(self):
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum) or self.minimum>self.maximum:
            raise ValueError("Finite source interval certificate required")
        if self.quantum is not None and not -1074<=self.quantum<=1023:
            raise ValueError("Invalid finite-source dyadic quantum")


def call(name,*args):
    return name+"("+", ".join(map(str,args))+")"


def binary_exponent(value):
    # log2 rounds to an integer for some values immediately below a power
    # of two. frexp exposes the actual IEEE binade without that ambiguity.
    return math.frexp(value)[1]-1


def lower_finite_conversion(expression,kind,certificate,compiler,domains,path=()):
    if not isinstance(certificate,FiniteSource):
        raise ValueError("Finite source interval certificate required")
    if kind not in ("R32","R16"):
        raise ValueError("Unsupported conversion boundary")
    # Fixed-point simplification must complete before this substitution too.
    source=compiler.stabilize("("+expression+")",domains,path)
    raw=call("Bits64","X999999997")
    magnitude=call("U64And",raw,0x7fffffffffffffff)
    sign=call("U64And",raw,0x8000000000000000)
    dropped=29 if kind=="R32" else 42
    bias=(1<<(dropped-1))-1
    mask=((1<<64)-1)^((1<<dropped)-1)
    odd=call("U64And",call("U64Shr",magnitude,dropped),1)
    normal=call("U64And",call("U64Add",magnitude,call("U64Add",bias,odd)),mask)
    # Fixed subnormal grids: the F64 addition genuinely rounds to the target
    # quantum (2^-149 / 2^-24). The subtraction is exact; never cancel it.
    offset="2**(-97)" if kind=="R32" else "2**28"
    subnormal=call("Bits64","(("+call("Float64",magnitude)+") + ("+offset+")) - ("+offset+")")
    small_threshold=0x3810000000000000 if kind=="R32" else 0x3f10000000000000
    overflow_threshold=0x47effffff0000000 if kind=="R32" else 0x40effe0000000000
    maximum=max(abs(certificate.minimum),abs(certificate.maximum))
    minimum=certificate.minimum if certificate.minimum>0 else -certificate.maximum if certificate.maximum<0 else 0
    smallest=2**(-126 if kind=="R32" else -14)
    overflow=2**128-2**103 if kind=="R32" else 65520
    zero_or_normal=certificate.quantum is not None and certificate.quantum>=(-126 if kind=="R32" else -14)
    if maximum<overflow and (minimum>=smallest or zero_or_normal):
        # Under this certificate the low-bit bias cannot carry into bit 63.
        # Round the signed word itself: the retained-bit parity is independent
        # of its sign. Avoid stripping/restoring a third copy of the producer
        # inside the same normal calculation; distinct paths are untouched.
        signed_odd=call("U64And",call("U64Shr",raw,dropped),1)
        signed=call("Float64",call("U64And",call("U64Add",raw,
            call("U64Add",bias,signed_odd)),mask))
        return compiler.substitute(signed,"X999999997",source,domains,path)
    above=normal if maximum<overflow else call("Piecewise",
        "("+normal+", "+magnitude+" < "+str(overflow_threshold)+")",
        "("+str(0x7ff0000000000000)+", True)")
    if maximum<smallest:positive=subnormal
    elif minimum>=smallest or zero_or_normal:positive=above
    else:positive=call("Piecewise","("+subnormal+", "+magnitude+" < "+str(small_threshold)+")","("+above+", True)")
    template=call("Float64",call("U64Or",positive,sign))
    # Substitute the producer only after the template is complete. Every arm
    # then runs factor/simplify under its own inherited condition context.
    return compiler.substitute(template,"X999999997",source,domains,path)


class ConversionSession:
    """Compiler-only intervals for already-substituted string producers.

    An interval is never an expression representation or runtime metadata.
    Unknown/overflowing intervals are barriers; no finite input is assumed
    merely because a kernel constructor exists.
    """
    def __init__(self,compiler,domains,*,input_dtype=None):
        self.compiler,self.domains=compiler,domains
        self.completed={}
        self.half_values={ast.dump(ast.Name(id=name,ctx=ast.Load())) for name in domains} if input_dtype=="f16" else set()
        self.f32_values=set(self.half_values)
        self.closed=0
        self.pending=0
        self.redundant=0

    @staticmethod
    def constant(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return float(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            x=ConversionSession.constant(node.operand)
            return None if x is None else x if isinstance(node.op,ast.UAdd) else -x
        if isinstance(node,ast.BinOp):
            a,b=ConversionSession.constant(node.left),ConversionSession.constant(node.right)
            if a is None or b is None:return None
            try:
                if isinstance(node.op,ast.Add):return a+b
                if isinstance(node.op,ast.Sub):return a-b
                if isinstance(node.op,ast.Mult):return a*b
                if isinstance(node.op,ast.Div):return a/b
            except (OverflowError,ZeroDivisionError):return None
        if isinstance(node,ast.Call) and len(node.args)==1 and node.func.id in ("R16","R32"):
            x=ConversionSession.constant(node.args[0])
            if x is None:return None
            fmt="e" if node.func.id=="R16" else "f"
            try:return struct.unpack(fmt,struct.pack(fmt,x))[0]
            except OverflowError:return None
        return None

    def value_kind(self,node):
        key=ast.dump(node)
        if key in self.half_values:return "half"
        if key in self.f32_values:return "f32"
        value=self.constant(node)
        if value is not None and math.isfinite(value):
            for fmt,kind in (("e","half"),("f","f32")):
                try:
                    if struct.unpack(fmt,struct.pack(fmt,value))[0]==value:return kind
                except OverflowError:pass
        if isinstance(node,ast.Call) and len(node.args)==1 and node.func.id in ("R16","R32") and self.bounds(node) is not None:
            return "half" if node.func.id=="R16" or self.value_kind(node.args[0])=="half" else "f32"
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return self.value_kind(node.operand)
        if not isinstance(node,ast.BinOp):return None
        a,b=self.value_kind(node.left),self.value_kind(node.right)
        left,right=self.constant(node.left),self.constant(node.right)
        if self.bounds(node.left) is None or self.bounds(node.right) is None:return None
        if isinstance(node.op,(ast.Add,ast.Sub)):
            if left==0 and b is not None:return b
            if right==0 and a is not None:return a
            if isinstance(node.op,ast.Sub) and ast.dump(node.left)==ast.dump(node.right):return "half"
        if isinstance(node.op,ast.Mult):
            if (left==0 and b is not None) or (right==0 and a is not None):return "half"
            if left is not None and abs(left)==1 and b is not None:return b
            if right is not None and abs(right)==1 and a is not None:return a
            # Products of finite Half values have at most 22 significant
            # bits, minimum nonzero 2^-48, and fit normal F32. No rounding.
            if a==b=="half":return "f32"
        if isinstance(node.op,ast.Div) and a is not None and right is not None and right:
            if abs(right)==1:return a
            d=self.bounds(node)
            if math.frexp(abs(right))[0]==0.5 and d is not None and d.quantum is not None and d.quantum>=-149 and max(abs(d.minimum),abs(d.maximum))<=3.4028234663852886e38:return "f32"
        return None

    def bounds(self,node):
        key=ast.dump(node)
        if key in self.completed:return self.completed[key]
        if isinstance(node,ast.Name):
            d=self.domains.get(node.id)
            return FiniteSource(float(d.minimum),float(d.maximum),d.quantum) if d else None
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            x=float(node.value)
            return FiniteSource(x,x,quantum(Fraction(x))) if math.isfinite(x) else None
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            d=self.bounds(node.operand)
            return None if d is None else d if isinstance(node.op,ast.UAdd) else FiniteSource(-d.maximum,-d.minimum,d.quantum)
        if isinstance(node,ast.BinOp):
            a,b=self.bounds(node.left),self.bounds(node.right)
            if a is None or b is None:return None
            try:
                if isinstance(node.op,ast.Add):values=[a.minimum+b.minimum,a.maximum+b.maximum]
                elif isinstance(node.op,ast.Sub):values=[a.minimum-b.maximum,a.maximum-b.minimum]
                elif isinstance(node.op,ast.Mult):values=[x*y for x in (a.minimum,a.maximum) for y in (b.minimum,b.maximum)]
                elif isinstance(node.op,ast.Div) and not b.minimum<=0<=b.maximum:values=[x/y for x in (a.minimum,a.maximum) for y in (b.minimum,b.maximum)]
                else:return None
                # Outward enclosure of actual F64 arithmetic. If a bound
                # reaches infinity it cannot certify a finite-source kernel.
                low=math.nextafter(min(values),-math.inf);high=math.nextafter(max(values),math.inf)
                if isinstance(node.op,ast.Mult) and ast.dump(node.left)==ast.dump(node.right):
                    low=0 if a.minimum<=0<=a.maximum else math.nextafter(min(a.minimum*a.minimum,a.maximum*a.maximum),-math.inf)
                q=None
                if a.quantum is not None and b.quantum is not None:
                    if isinstance(node.op,(ast.Add,ast.Sub)):q=min(a.quantum,b.quantum)
                    elif isinstance(node.op,ast.Mult):q=a.quantum+b.quantum
                    elif isinstance(node.op,ast.Div) and b.minimum==b.maximum and b.minimum:
                        d=Fraction(b.minimum)
                        if abs(d.numerator)&(abs(d.numerator)-1)==0:q=a.quantum-quantum(d)
                minimum=low if low>0 else -high if high<0 else 0
                inherent=max(-1074,binary_exponent(minimum)-52) if minimum else -1074
                q=max(inherent,q) if q is not None else inherent
                return FiniteSource(low,high,q)
            except (OverflowError,ValueError):return None
        if isinstance(node,ast.Call) and len(node.args)==1:
            d=self.bounds(node.args[0])
            if d is None:return None
            try:
                if node.func.id in ("R32","R16"):
                    fmt="f" if node.func.id=="R32" else "e"
                    cast=lambda x:struct.unpack(fmt,struct.pack(fmt,x))[0]
                    low,high=cast(d.minimum),cast(d.maximum)
                    minimum=low if low>0 else -high if high<0 else 0
                    q=-24 if fmt=="e" else max(-149,binary_exponent(minimum)-23) if minimum else -149
                    if fmt=="f" and d.quantum is not None:q=max(q,d.quantum)
                    return FiniteSource(low,high,q)
                if node.func.id=="sqrt" and d.minimum>=0:
                    low,high=math.sqrt(d.minimum),math.sqrt(d.maximum)
                    q=max(-1074,binary_exponent(low)-52) if low else -1074
                    return FiniteSource(low,high,q)
                if node.func.id=="Silu16":
                    # |silu(x)|<=|x|; a conservative finite-half enclosure,
                    # not an implementation or proof of the activation bits.
                    maximum=max(abs(d.minimum),abs(d.maximum))
                    if maximum<=65504:return FiniteSource(-maximum,maximum,-24)
            except (OverflowError,ValueError):return None
        return None

    def close(self,expression):
        session=self
        class Boundaries(ast.NodeTransformer):
            def visit_Call(self,node):
                before=session.bounds(node)
                literal=session.constant(node) if node.func.id in ("R16","R32") else None
                if literal is not None and math.isfinite(literal):
                    text=session.compiler.stabilize(repr(literal),session.domains)
                    rewritten=syntax(text);key=ast.dump(rewritten)
                    if before is not None:session.completed[key]=before
                    session.f32_values.add(key)
                    if node.func.id=="R16" or session.value_kind(rewritten)=="half":session.half_values.add(key)
                    session.redundant+=1
                    return rewritten
                # Half addition/subtraction followed immediately by Half
                # storage admits removal of its intermediate F32 boundary.
                # <=12 exponent gap fits F32; larger gaps cannot cross the
                # Half cell of the larger operand, even after F32 rounding.
                # This rule never extends to arbitrary sums of products.
                if node.func.id=="R16" and len(node.args)==1:
                    inner=node.args[0]
                    if isinstance(inner,ast.Call) and inner.func.id=="R32" and len(inner.args)==1:
                        op=inner.args[0]
                        if isinstance(op,ast.BinOp) and isinstance(op.op,(ast.Add,ast.Sub)) and session.value_kind(op.left)==session.value_kind(op.right)=="half":
                            node.args[0]=op;session.redundant+=1
                source=session.bounds(node.args[0]) if node.func.id in ("R32","R16") and len(node.args)==1 else None
                rewritten=self.generic_visit(node)
                if rewritten.func.id in ("R32","R16") and len(rewritten.args)==1:
                    target=rewritten.func.id
                    kind=session.value_kind(rewritten.args[0])
                    if (target=="R32" and kind is not None) or (target=="R16" and kind=="half"):
                        rewritten=syntax(session.compiler.stabilize(ast.unparse(rewritten.args[0]),session.domains));session.redundant+=1
                    elif source is not None:
                        text=lower_finite_conversion(ast.unparse(rewritten.args[0]),rewritten.func.id,
                            source,session.compiler,session.domains)
                        rewritten=syntax(text)
                        session.closed+=1
                    else:session.pending+=1
                    if before is not None:
                        key=ast.dump(rewritten)
                        session.f32_values.add(key)
                        if target=="R16" or kind=="half":session.half_values.add(key)
                if before is not None:session.completed[ast.dump(rewritten)]=before
                return rewritten
        tree=Boundaries().visit(syntax(expression))
        return self.compiler.stabilize(ast.unparse(tree),self.domains)

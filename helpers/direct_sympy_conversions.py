"""Finite IEEE conversions as mathematical strings of elementary operations.

Bits64/Float64 are bit reinterpretations; U64And/U64Or/U64Shr/U64Add are
unsigned integer operations, not numerical conversion helpers. Piecewise
retains every classification. Constructors require a finite-source certificate.
No R16/R32 or rounding-mode node is retained in their returned syntax.
"""
from dataclasses import dataclass
from collections import OrderedDict
from types import MappingProxyType
import ast
import math
import struct
import re
from fractions import Fraction

from direct_sympy_strings import StringCompiler
from direct_sympy_strings import syntax
from direct_sympy_strings import quantum
from direct_sympy_words import simplify_words
from direct_sympy_signatures import StructuralSignatures
from direct_sympy_tandem import supported as tandem_supported,lower_tandem
from direct_sympy_arithmetic import simplify_arithmetic


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


def lower_finite_conversion(expression,kind,certificate,compiler,domains,path=(),*,no_odd_f32_ties=False):
    if not isinstance(certificate,FiniteSource):
        raise ValueError("Finite source interval certificate required")
    if kind not in ("R32","R16"):
        raise ValueError("Unsupported conversion boundary")
    if no_odd_f32_ties and kind!="R32":raise ValueError("Odd-tie certificate applies only to F32")
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
    # A value on the target subnormal grid already has at most p-1
    # significant bits below the normal threshold. Normal word rounding
    # leaves it unchanged, including signed zero. A separate small kernel
    # is needed only when the source can lie between those grid points.
    zero_or_normal=certificate.quantum is not None and certificate.quantum>=(-149 if kind=="R32" else -24)
    if maximum<overflow and (minimum>=smallest or zero_or_normal):
        # Under this certificate the low-bit bias cannot carry into bit 63.
        # Round the signed word itself: the retained-bit parity is independent
        # of its sign. Avoid stripping/restoring a third copy of the producer
        # inside the same normal calculation; distinct paths are untouched.
        adjustment=bias if no_odd_f32_ties else call("U64Add",bias,call("U64And",call("U64Shr",raw,dropped),1))
        signed=call("Float64",call("U64And",call("U64Add",raw,adjustment),mask))
        return simplify_words(compiler.substitute(signed,"X999999997",source,domains,path),compiler,domains)
    above=normal if maximum<overflow else call("Piecewise",
        "("+normal+", "+magnitude+" < "+str(overflow_threshold)+")",
        "("+str(0x7ff0000000000000)+", True)")
    if maximum<smallest:positive=subnormal
    elif minimum>=smallest or zero_or_normal:positive=above
    else:positive=call("Piecewise","("+subnormal+", "+magnitude+" < "+str(small_threshold)+")","("+above+", True)")
    template=call("Float64",call("U64Or",positive,sign))
    # Substitute the producer only after the template is complete. Every arm
    # then runs factor/simplify under its own inherited condition context.
    return simplify_words(compiler.substitute(template,"X999999997",source,domains,path),compiler,domains)


class ConversionSession:
    """Compiler-only intervals for already-substituted string producers.

    An interval is never an expression representation or runtime metadata.
    Unknown/overflowing intervals are barriers; no finite input is assumed
    merely because a kernel constructor exists.
    """
    def __init__(self,compiler,domains,*,input_dtype=None):
        self.compiler,self.domains=compiler,MappingProxyType(dict(domains))
        self.signatures=StructuralSignatures()
        self.key=self.signatures.key
        self.completed={}
        self.half_values={self.key(ast.Name(id=name,ctx=ast.Load())) for name in domains} if input_dtype=="f16" else set()
        self.f32_values=set(self.half_values)
        self.closed=0
        self.pending=0
        self.redundant=0
        self.converted_regions=set()
        self.reused_regions=0
        self.visited_nodes=0
        self.no_negative_zero_values=set()
        self.arithmetic_eliminated=0
        self.closed_literals=OrderedDict()
        self.closed_literal_characters=0
        self.envelope_serial=0
        self.numeric_envelopes=[]

    def remember_closed_literal(self,expression,node=None):
        """Session-local proofs for immutable, completed literal strings."""
        if re.search(r"\bCASNumericRegion[0-9]+\b",expression):return
        node=syntax(expression) if node is None else node
        if self.key(node) not in self.converted_regions:return
        if expression in self.closed_literals:return
        while self.closed_literals and self.closed_literal_characters+len(expression)>4*self.compiler.max_characters:
            old,_=self.closed_literals.popitem(last=False);self.closed_literal_characters-=len(old)
        if len(expression)>4*self.compiler.max_characters:return
        self.closed_literals[expression]=(self.bounds(node),self.value_kind(node),self.no_negative_zero(node))
        self.closed_literal_characters+=len(expression)

    def propagate_closed_identity(self,original,replacement):
        """Transfer whole-root proofs after a proven pure control rewrite.

        Selected branch bodies do not inherit global dtype/range proofs:
        those may hold only under a branch's guard. No runtime aliases exist.
        """
        before=syntax(original);old=self.key(before)
        if old not in self.converted_regions:raise ValueError("Control rewrite requires a closed numeric frontier")
        after=syntax(replacement);new=self.key(after)
        bounds=self.bounds(before);kind=self.value_kind(before)
        if bounds is not None:self.completed[new]=bounds
        if kind=='half':self.half_values.add(new)
        if kind in ('half','f32'):self.f32_values.add(new)
        if self.no_negative_zero(before):self.no_negative_zero_values.add(new)
        self.converted_regions.add(new)
        self.remember_closed_literal(replacement,after)

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
                if isinstance(node.op,ast.Pow) and b==2:return a*a
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
        key=self.key(node)
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
        if isinstance(node,ast.Call) and len(node.args)==1 and node.func.id=="Silu16" and self.value_kind(node.args[0])=="half" and self.bounds(node) is not None:return "half"
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return self.value_kind(node.operand)
        if not isinstance(node,ast.BinOp):return None
        a,b=self.value_kind(node.left),self.value_kind(node.right)
        left,right=self.constant(node.left),self.constant(node.right)
        if self.bounds(node.left) is None or self.bounds(node.right) is None:return None
        if isinstance(node.op,ast.Pow) and right==2 and a=="half":return "f32"
        if isinstance(node.op,(ast.Add,ast.Sub)):
            if left==0 and b is not None:return b
            if right==0 and a is not None:return a
            if isinstance(node.op,ast.Sub) and self.key(node.left)==self.key(node.right):return "half"
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

    def no_negative_zero(self,node):
        key=self.key(node)
        if key in self.no_negative_zero_values:return True
        d=self.bounds(node)
        if d is None:return False
        if d.minimum>0 or d.maximum<0:return True
        literal=self.constant(node)
        if literal is not None:return literal!=0 or math.copysign(1,literal)>0
        if isinstance(node,ast.Name):
            return self.domains[node.id].excludes_negative_zero
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.UAdd):return self.no_negative_zero(node.operand)
        if isinstance(node,ast.BinOp):
            if isinstance(node.op,ast.Pow) and self.constant(node.right)==2 and self.value_kind(node.left)=="half":return True
            # Finite F64 addition cannot round a nonzero exact sum to zero:
            # both operands are integer multiples of the minimum subnormal.
            # The only negative-zero sum under RN-even is -0 + -0.
            if isinstance(node.op,ast.Add):return self.no_negative_zero(node.left) or self.no_negative_zero(node.right)
            if isinstance(node.op,ast.Sub) and self.key(node.left)==self.key(node.right):return True
        if isinstance(node,ast.Call) and len(node.args)==1 and node.func.id in ("R16","R32"):
            source=self.bounds(node.args[0]);minimum=-24 if node.func.id=="R16" else -149
            return source is not None and source.quantum is not None and source.quantum>=minimum and self.no_negative_zero(node.args[0])
        return False

    def bounds(self,node):
        key=self.key(node)
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
            # Only certified finite Half squares are admitted here. Their
            # 22-bit product is exact in F32 and F64, including positive zero.
            # Reuse the multiplication enclosure without duplicating text.
            if isinstance(node.op,ast.Pow) and self.constant(node.right)==2 and self.value_kind(node.left)=="half" and self.bounds(node.left) is not None:
                return self.bounds(ast.BinOp(left=node.left,op=ast.Mult(),right=node.left))
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
                if isinstance(node.op,ast.Mult) and self.key(node.left)==self.key(node.right):
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
                    if maximum<=65504 and self.value_kind(node.args[0])=="half":return FiniteSource(-maximum,maximum,-24)
            except (OverflowError,ValueError):return None
        return None

    def no_odd_f32_ties(self,node):
        """Certified sources cannot land on an odd normal F32 halfway cell.

        Write each nonzero Half as odd m*2^e, with <=11 bits in m.
        Equal square exponents give <=23 significant bits: already F32.
        Unequal square exponents differ by an even number; the aligned sum
        is 1 mod 4. An exact F64 sum can be an F32 tie only with one bit
        dropped, whose retained bit is then even. If F64 addition rounds,
        the smaller <=22-bit square is <1/128 of the larger square's F32
        ULP. That larger square is already exact F32, so no tie is possible.
        Zeros square to +0. This is not a certificate for general Half sums.

        For sqrt of finite nonnegative F32, a normal F32 midpoint has an
        odd 25-bit significand. Its square has 49 or 50 significant bits,
        so cannot be the 24-bit input. The minimum square-space separation
        is larger than the F64 rounding interval around that midpoint.
        For 1/F32, the midpoint's odd significand cannot divide a power of
        two; the <=24-bit denominator also bounds separation away from the
        F64 midpoint interval. Thus neither F64 operation can round onto
        a normal F32 midpoint. Generic subnormal/overflow kernels remain.
        """
        if isinstance(node,ast.Call) and node.func.id=="sqrt" and len(node.args)==1:
            source=self.bounds(node.args[0])
            return source is not None and source.minimum>=0 and self.value_kind(node.args[0]) in ("half","f32")
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and self.constant(node.left)==1:
            source=self.bounds(node.right)
            return source is not None and (source.minimum>0 or source.maximum<0) and self.value_kind(node.right) in ("half","f32")
        def square(value):
            if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Pow) and self.constant(value.right)==2:
                operand=value.left
            elif isinstance(value,ast.BinOp) and isinstance(value.op,ast.Mult) and self.key(value.left)==self.key(value.right):operand=value.left
            else:return False
            return self.value_kind(operand)=="half" and self.bounds(operand) is not None
        return isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add) and square(node.left) and square(node.right)

    def close(self,expression):
        if re.search(r"\bCASNumericRegion[0-9]+\b",expression):raise ValueError("Reserved compiler numeric placeholder")
        compact,regions=self.compiler.compact_regions(expression,self.compiler.context(self.domains))
        if regions and all(text in self.closed_literals for text in regions.values()):
            protected={};literal_keys={}
            for old,text in regions.items():
                self.envelope_serial+=1;token="CASNumericRegion"+str(self.envelope_serial)
                marker=syntax(token+"()");key=self.key(marker)
                bounds,kind,positive_zero=self.closed_literals[text]
                if bounds is not None:self.completed[key]=bounds
                if kind=="half":self.half_values.add(key)
                if kind in ("half","f32"):self.f32_values.add(key)
                if positive_zero:self.no_negative_zero_values.add(key)
                self.converted_regions.add(key);protected[token]=text
                literal_keys[key]=self.key(syntax(text))
                compact=re.sub(r"\b"+old+r"\b",token+"()",compact)
            # Preserve correlation certificates on enclosing operations.
            # Independent interval arithmetic cannot recover these from the
            # bounds of individual completed literals (e.g. RMS products).
            virtual_tree=syntax(compact)
            for node,original_key in self.signatures.translated_keys(virtual_tree,literal_keys).items():
                virtual_key=self.key(node)
                if original_key in self.completed:self.completed[virtual_key]=self.completed[original_key]
                if original_key in self.half_values:self.half_values.add(virtual_key)
                if original_key in self.f32_values:self.f32_values.add(virtual_key)
                if original_key in self.no_negative_zero_values:self.no_negative_zero_values.add(virtual_key)
                if original_key in self.converted_regions:self.converted_regions.add(virtual_key)
            result=self._close(compact)
            virtual=syntax(result);original=self.key(virtual)
            pattern=r"\bCASNumericRegion[0-9]+\(\)"
            expanded=len(result)
            for token,text in protected.items():expanded+=len(re.findall(r"\b"+token+r"\(\)",result))*(len(text)-len(token)-2)
            self.numeric_envelopes.append((len(expression),len(compact),len(result),expanded,len(protected)))
            if expanded>self.compiler.max_characters:
                raise ValueError(f"Closed numeric envelope exceeds string budget before allocation: expandedCharacters={expanded} limit={self.compiler.max_characters}")
            result=re.sub(pattern,lambda match:protected[match.group(0)[:-2]],result)
            if re.search(r"\bCASNumericRegion[0-9]+\b",result):raise ValueError("Numeric placeholder escaped restoration")
            # The virtual fixed point must retain its proof on the restored
            # whole root. Never infer a dtype from mere textual expansion.
            node=syntax(result);key=self.key(node)
            if original in self.completed:self.completed[key]=self.completed[original]
            if original in self.half_values:self.half_values.add(key)
            if original in self.f32_values:self.f32_values.add(key)
            if original in self.no_negative_zero_values:self.no_negative_zero_values.add(key)
            if original in self.converted_regions:self.converted_regions.add(key)
            self.remember_closed_literal(result,node)
            return result
        return self._close(expression)

    def _close(self,expression):
        expression=simplify_arithmetic(expression,self)
        session=self
        class Boundaries(ast.NodeTransformer):
            def visit(self,node):
                # This region has no conversion left, under this session's
                # unchanged input domain. Keep every literal copy, but avoid
                # walking its already-converted descendants again.
                if session.key(node) in session.converted_regions:
                    session.reused_regions+=1
                    return node
                session.visited_nodes+=1
                return super().visit(node)

            def generic_visit(self,node):
                # Bounds may have cached the complete pre-substitution tree.
                # Invalidate each mutable parent around child replacement.
                session.signatures.invalidate(node)
                result=super().generic_visit(node)
                session.signatures.invalidate(result)
                return result

            def visit_Call(self,node):
                before=session.bounds(node)
                positive_zero=session.no_negative_zero(node)
                literal=session.constant(node) if node.func.id in ("R16","R32") else None
                if literal is not None and math.isfinite(literal):
                    text=session.compiler.stabilize(repr(literal),session.domains)
                    rewritten=syntax(text);key=session.key(rewritten)
                    if before is not None:session.completed[key]=before
                    if positive_zero:session.no_negative_zero_values.add(key)
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
                if node.func.id=="R16" and len(node.args)==1:
                    inner=node.args[0]
                    if isinstance(inner,ast.Call) and inner.func.id=="R32" and len(inner.args)==1:
                        raw=inner.args[0];certificate=session.bounds(raw)
                        if tandem_supported(certificate) and session.value_kind(raw) is None:
                            raw=self.visit(raw)
                            text=lower_tandem(ast.unparse(raw),certificate,session.compiler,session.domains)
                            rewritten=syntax(text);session.closed+=2
                            if before is not None:
                                key=session.key(rewritten);session.completed[key]=before
                                if positive_zero:session.no_negative_zero_values.add(key)
                                session.half_values.add(key);session.f32_values.add(key)
                            return rewritten
                source=session.bounds(node.args[0]) if node.func.id in ("R32","R16") and len(node.args)==1 else None
                no_odd_ties=node.func.id=="R32" and len(node.args)==1 and session.no_odd_f32_ties(node.args[0])
                rewritten=self.generic_visit(node)
                if rewritten.func.id in ("R32","R16") and len(rewritten.args)==1:
                    target=rewritten.func.id
                    kind=session.value_kind(rewritten.args[0])
                    if (target=="R32" and kind is not None) or (target=="R16" and kind=="half"):
                        rewritten=syntax(session.compiler.stabilize(ast.unparse(rewritten.args[0]),session.domains));session.redundant+=1
                    elif source is not None:
                        text=lower_finite_conversion(ast.unparse(rewritten.args[0]),rewritten.func.id,
                            source,session.compiler,session.domains,no_odd_f32_ties=no_odd_ties)
                        rewritten=syntax(text)
                        session.closed+=1
                    else:session.pending+=1
                    if before is not None:
                        key=session.key(rewritten)
                        session.f32_values.add(key)
                        if target=="R16" or kind=="half":session.half_values.add(key)
                if before is not None:
                    key=session.key(rewritten);session.completed[key]=before
                    if positive_zero:session.no_negative_zero_values.add(key)
                return rewritten
        tree=Boundaries().visit(syntax(expression))
        result=self.compiler.stabilize(ast.unparse(tree),self.domains)
        if not re.search(r"\bR(?:16|32)\s*\(",result):
            self.converted_regions.add(self.key(syntax(result)))
            self.remember_closed_literal(result)
        return result

"""Finite IEEE conversions as mathematical strings of elementary operations.

Bits64/Float64 are bit reinterpretations; U64And/U64Or/U64Shr/U64Add are
unsigned integer operations, not numerical conversion helpers. Piecewise
retains every classification. Constructors require a finite-source certificate.
No R16/R32 or rounding-mode node is retained in their returned syntax.
"""
from dataclasses import dataclass
from collections import OrderedDict
from types import MappingProxyType
from contextlib import contextmanager
import ast
import math
import struct
import re
from fractions import Fraction

from direct_sympy_strings import StringCompiler
from direct_sympy_strings import syntax
from direct_sympy_strings import quantum,refine
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


def lower_finite_conversion(expression,kind,certificate,compiler,domains,path=(),*,no_odd_f32_ties=False,integer_word_exact=False,frontier=None,sign_word=None,facts=()):
    if not isinstance(certificate,FiniteSource):
        raise ValueError("Finite source interval certificate required")
    if kind not in ("R32","R16"):
        raise ValueError("Unsupported conversion boundary")
    if no_odd_f32_ties and kind!="R32":raise ValueError("Odd-tie certificate applies only to F32")
    # Fixed-point simplification must complete before this substitution too.
    source=compiler.stabilize("("+expression+")",domains,path,facts=facts)
    raw=call("Bits64","X999999997")
    magnitude=call("U64And",raw,0x7fffffffffffffff)
    sign=call("U64And",raw,0x8000000000000000) if sign_word is None else "X999999996"
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
    # A path can exclude the entire central magnitude interval while still
    # admitting both signs. A single signed interval cannot express that.
    # Reuse only the exact predicate already present in this branch's facts.
    classification=None
    if facts:
        guard=syntax(call("U64And",call("Bits64",source),0x7fffffffffffffff)+" < "+str(small_threshold))
        classification=compiler.branch_facts.truth(guard,facts)
        if classification is not None:compiler.condition_events.append(("numeric/"+kind,"small","path-proved-"+str(classification).lower()))
    normal_domain=minimum>=smallest or zero_or_normal or classification is False
    if maximum<overflow and normal_domain:
        # Under this certificate the low-bit bias cannot carry into bit 63.
        # Round the signed word itself: the retained-bit parity is independent
        # of its sign. Avoid stripping/restoring a third copy of the producer
        # inside the same normal calculation; distinct paths are untouched.
        adjustment=bias if no_odd_f32_ties else call("U64Add",bias,call("U64And",call("U64Shr",raw,dropped),1))
        signed=call("Float64",call("U64And",call("U64Add",raw,adjustment),mask))
        if integer_word_exact:
            exponent=81 if kind=="R32" else 94
            signed=call("Float64",call("U64FromF64","(("+call("F64FromU64",raw)+") + 2**"+str(exponent)+") - 2**"+str(exponent)))
        return simplify_words(compiler.substitute(signed,"X999999997",source,domains,path),compiler,domains)
    if integer_word_exact:
        exponent=81 if kind=="R32" else 94
        normal=call("U64And",call("U64FromF64","(("+call("F64FromU64",raw)+") + 2**"+str(exponent)+") - 2**"+str(exponent)),0x7fffffffffffffff)
    above=normal if maximum<overflow else call("Piecewise",
        "("+normal+", "+magnitude+" < "+str(overflow_threshold)+")",
        "("+str(0x7ff0000000000000)+", True)")
    mixed=False
    if classification is True or maximum<smallest:positive=subnormal
    elif normal_domain:positive=above
    else:
        mixed=True
        positive=call("Piecewise","("+subnormal+", "+magnitude+" < "+str(small_threshold)+")","("+above+", True)")
    template=call("Float64",call("U64Or",positive,sign))
    if sign_word is not None:template=compiler.substitute(template,"X999999996",sign_word,domains,path)
    # Substitute the producer only after the template is complete. Every arm
    # then runs factor/simplify under its own inherited condition context.
    result=simplify_words(compiler.substitute(template,"X999999997",source,domains,path),compiler,domains)
    if mixed and frontier is not None:
        guard=compiler.substitute(magnitude+' < '+str(small_threshold),'X999999997',source,domains,path)
        frontier.append((guard,-smallest,smallest,-149 if kind=='R32' else -24))
    return result


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
        self.activations_closed=0
        self.square_roots_closed=0
        self.pending=0
        self.redundant=0
        self.converted_regions=set()
        self.pure_numeric_regions=set()
        self.reused_regions=0
        self.visited_nodes=0
        self.no_negative_zero_values=set()
        self.arithmetic_eliminated=0
        self.closed_literals=OrderedDict()
        self.closed_literal_characters=0
        self.closed_literal_keys={}
        self.closed_literal_pure={}
        self.envelope_serial=0
        self.numeric_envelopes=[]
        self.selector_literals=OrderedDict()
        self.selector_view_characters=0
        self.branch_depth=0
        self.branch_facts=()
        self.frontier_bounds={}
        self.sign_projections={}
        self.closed_sign_literals={}
        self.frontier_events=[]
        self.rms_guards={}

    @contextmanager
    def branch_context(self,domains,proofs,facts):
        """Own every new numerical proof until this arm has been closed.

        Inherited proofs remain conservative on a subset of their domain.
        New proofs must not escape to siblings or enter global literal caches.
        Strings and structural signatures are shared; only proof tables copy.
        """
        fields=('completed','half_values','f32_values','converted_regions',
                'no_negative_zero_values','frontier_bounds','sign_projections','rms_guards','pure_numeric_regions')
        saved={name:getattr(self,name) for name in fields}
        old_domains,old_facts=self.domains,self.branch_facts
        for name,value in saved.items():setattr(self,name,value.copy())
        self.completed.update(proofs)
        self.domains=MappingProxyType(dict(domains));self.branch_facts=facts
        self.branch_depth+=1
        try:yield
        finally:
            for name,value in saved.items():setattr(self,name,value)
            self.domains,self.branch_facts=old_domains,old_facts
            self.branch_depth-=1

    def magnitude_guard_bounds(self,condition,truth,proofs):
        """Conservative finite-F64 preimage of an unsigned magnitude guard.

        A false guard spanning both signs has a disconnected preimage; leave
        it unrefined. Never infer a dtype from a word comparison.
        """
        if not isinstance(condition,ast.Compare) or not isinstance(condition.ops[0],(ast.Lt,ast.LtE)):return proofs
        left,right=condition.left,condition.comparators[0]
        if not isinstance(right,ast.Constant) or type(right.value) is not int or not 0<right.value<0x7ff0000000000000:return proofs
        if not isinstance(left,ast.Call) or left.func.id!='U64And' or len(left.args)!=2:return proofs
        raw,mask=left.args
        if not isinstance(mask,ast.Constant) or type(mask.value) is not int or mask.value!=0x7fffffffffffffff:return proofs
        if not isinstance(raw,ast.Call) or raw.func.id!='Bits64' or len(raw.args)!=1:return proofs
        source=raw.args[0];key=self.key(source);bound=proofs.get(key,self.bounds(source))
        if bound is None:return proofs
        threshold=struct.unpack('>d',struct.pack('>Q',right.value))[0]
        low,high=bound.minimum,bound.maximum
        if truth:
            endpoint=math.nextafter(threshold,0) if isinstance(condition.ops[0],ast.Lt) else threshold
            low,high=max(low,-endpoint),min(high,endpoint)
        elif low>=0:
            endpoint=threshold if isinstance(condition.ops[0],ast.Lt) else math.nextafter(threshold,math.inf)
            low=max(low,endpoint)
        elif high<=0:
            endpoint=threshold if isinstance(condition.ops[0],ast.Lt) else math.nextafter(threshold,math.inf)
            high=min(high,-endpoint)
        else:return proofs
        if low>high:return None
        result=dict(proofs);result[key]=FiniteSource(low,high,bound.quantum)
        return result

    def remember_frontier_bounds(self,expression,certificates):
        if not certificates:return
        from direct_sympy_synchronize import frontier
        functions={child.func.id for child in ast.walk(syntax(expression)) if isinstance(child,ast.Call) and self.key(child) in self.converted_regions}
        arms=frontier(expression,pure_functions=functions)
        if arms is None:return
        known={self.key(syntax(guard)):FiniteSource(low,high,q) for guard,low,high,q in certificates}
        bounds=tuple(known.get(self.key(condition)) for _,condition in arms)
        if any(value is not None for value in bounds):self.frontier_bounds[self.key(syntax(expression))]=bounds

    def remember_selector_literal(self,expression,view,leaves=None,arm_bounds=None):
        """Retain a bounded, compiler-only view of this rounding frontier.

        Leaf strings are immutable references, never copied or expanded here.
        They are not runtime intermediates and are restored before emission.
        """
        if expression in self.selector_literals or len(view)>1048576 or 'Piecewise(' not in view:return
        from direct_sympy_synchronize import GUARD_PURE
        leaves={} if leaves is None else leaves
        referenced=set(re.findall(r'\bCASNumericRegion[0-9]+\b',view))
        if not referenced.issubset(leaves):return
        leaves={token:leaves[token] for token in referenced}
        if any(isinstance(child,ast.Call) and child.func.id not in GUARD_PURE and child.func.id not in leaves for child in ast.walk(syntax(view))):return
        limit=16*1024*1024
        footprint=len(view)+128+sum(len(token)+128 for token in leaves)
        if footprint>limit:return
        while self.selector_literals and self.selector_view_characters+footprint>limit:
            _,old=self.selector_literals.popitem(last=False)
            self.selector_view_characters-=old[2]
        if arm_bounds is None:
            key=self.closed_literal_keys.get(expression)
            arm_bounds=self.frontier_bounds.get(key,())
        self.selector_literals[expression]=(view,leaves,footprint,arm_bounds)
        self.selector_view_characters+=footprint

    def selector_views(self,protected,pure_functions):
        """Expose only current producers; previous frontier leaves stay opaque."""
        views={};canonical={}
        for token,text in list(protected.items()):
            stored=self.selector_literals.get(text)
            if stored is None or token not in pure_functions:continue
            view,leaves,_,_=stored
            # A bounded proof cache may have evicted an earlier leaf. Keep
            # this producer opaque rather than restoring an unproven alias.
            if any(leaf not in self.closed_literal_keys for leaf in leaves.values()):continue
            aliases={}
            for old,leaf in leaves.items():
                # Separate leaf aliases prevent recursive expansion and preserve
                # equality of the same guard in two different producer views.
                if leaf not in canonical:
                    self.envelope_serial+=1
                    alias='CASNumericRegion'+str(self.envelope_serial)
                    canonical[leaf]=alias;protected[alias]=leaf
                    self.compiler.copy_completed_word_root(leaf,alias+'()',self.domains)
                    if self.closed_literal_pure.get(leaf,False):pure_functions.append(alias)
                aliases[old]=canonical[leaf]
            if aliases:view=re.sub(r'\bCASNumericRegion[0-9]+\b',lambda m:aliases.get(m.group(0),m.group(0)),view)
            views[token]=syntax(view)
        return views

    def remember_closed_literal(self,expression,node=None):
        """Session-local proofs for immutable, completed literal strings."""
        if self.branch_depth:return
        if re.search(r"\bCASNumericRegion[0-9]+\b",expression):return
        if expression in self.closed_literals:return
        node=syntax(expression) if node is None else node
        if self.key(node) not in self.converted_regions:return
        while self.closed_literals and self.closed_literal_characters+len(expression)>4*self.compiler.max_characters:
            old,_=self.closed_literals.popitem(last=False);self.closed_literal_characters-=len(old)
            self.closed_literal_keys.pop(old,None);self.closed_literal_pure.pop(old,None)
            self.closed_sign_literals.pop(old,None)
        if len(expression)>4*self.compiler.max_characters:return
        self.closed_literals[expression]=(self.bounds(node),self.value_kind(node),self.no_negative_zero(node))
        self.closed_literal_characters+=len(expression)
        self.compiler.register_completed_region(expression,self.domains,node,word_closed=True)
        self.closed_literal_keys[expression]=self.key(node)
        from direct_sympy_synchronize import GUARD_PURE
        self.closed_literal_pure[expression]=all(not isinstance(child,ast.Call) or child.func.id in GUARD_PURE for child in ast.walk(node))
        self.remember_selector_literal(expression,expression)
        projection=self.sign_projections.get(self.key(node))
        if projection is not None:
            text=simplify_words("Float64("+ast.unparse(projection)+")",self.compiler,self.domains)
            previous=self.closed_sign_literals.get(expression)
            if (previous is None or len(text)<len(previous)) and len(text)<len(expression) and not re.search(r'\bCAS(?:Numeric|Stable)Region[0-9]+\b',text):
                zero=syntax(text);key=self.key(zero)
                self.completed[key]=FiniteSource(0.0,0.0,-24)
                self.half_values.add(key);self.f32_values.add(key);self.converted_regions.add(key)
                self.remember_closed_literal(text,zero)
                self.closed_sign_literals[expression]=text

    def analyze_expression(self,expression):
        """Read-only type/range query over exact completed literal proofs.

        Return a small query tree and its original structural keys. Unique
        compiler-only atoms are never emitted or used as execution variables.
        Unknown contexts reopen the expression through the ordinary parser.
        """
        if expression in self.closed_literals:
            compact='CASStableRegion0';regions={compact:expression}
        else:
            compact,regions=self.compiler.compact_regions(expression,self.compiler.context(self.domains))
        if not regions or not all(text in self.closed_literals for text in regions.values()):
            return syntax(expression),{}
        replacements={}
        for name,text in regions.items():
            self.envelope_serial+=1;token='CASNumericRegion'+str(self.envelope_serial)
            marker=syntax(token+'()');key=self.key(marker)
            bounds,kind,positive_zero=self.closed_literals[text]
            if bounds is not None:self.completed[key]=bounds
            if kind=='half':self.half_values.add(key)
            if kind in ('half','f32'):self.f32_values.add(key)
            if positive_zero:self.no_negative_zero_values.add(key)
            self.converted_regions.add(key)
            replacements[key]=self.closed_literal_keys[text]
            compact=re.sub(r'\b'+name+r'\b',token+'()',compact)
        tree=syntax(compact);original=self.signatures.translated_keys(tree,replacements)
        for node,old in original.items():
            new=self.key(node)
            if old in self.completed:self.completed[new]=self.completed[old]
            if old in self.half_values:self.half_values.add(new)
            if old in self.f32_values:self.f32_values.add(new)
            if old in self.no_negative_zero_values:self.no_negative_zero_values.add(new)
            if old in self.converted_regions:self.converted_regions.add(new)
        return tree,original

    def restore_compact_literal(self,expression,node,regions,bounds,kind,positive_zero,closed,restored_keys,restored_purity,arm_bounds=()):
        """Recover exact structural identity from previously validated literals.

        Placeholders exist only during compilation. Interned structural keys,
        including every F64 token bit, compose the original expression key;
        immutable expression files remain fully substituted.
        """
        replacements={self.key(syntax(name)):restored_keys[text] for name,text in regions.items()}
        original=self.signatures.translated_keys(node,replacements)[node]
        if bounds is not None:self.completed[original]=bounds
        if kind=='half':self.half_values.add(original)
        if kind in ('half','f32'):self.f32_values.add(original)
        if positive_zero:self.no_negative_zero_values.add(original)
        from direct_sympy_synchronize import GUARD_PURE
        pure=all(not isinstance(child,ast.Call) or child.func.id in GUARD_PURE or self.key(child) in replacements for child in ast.walk(node)) and all(restored_purity[text] for text in regions.values())
        if not closed:return original,pure
        self.converted_regions.add(original)
        if expression in self.closed_literals:return original,pure
        while self.closed_literals and self.closed_literal_characters+len(expression)>4*self.compiler.max_characters:
            old,_=self.closed_literals.popitem(last=False);self.closed_literal_characters-=len(old)
            self.closed_literal_keys.pop(old,None);self.closed_literal_pure.pop(old,None)
            self.closed_sign_literals.pop(old,None)
        self.closed_literals[expression]=(bounds,kind,positive_zero)
        self.closed_literal_keys[expression]=original;self.closed_literal_pure[expression]=pure
        if arm_bounds:self.frontier_bounds[original]=arm_bounds
        self.closed_literal_characters+=len(expression)
        # Reconstruct the view from the validated, compact saved producer.
        # Persisted expressions contain no aliases; restore creates fresh
        # compiler-only leaf names instead of losing selector visibility.
        if pure:
            view=ast.unparse(node)
            leaves={}
            if len(view)<=1048576 and 'Piecewise(' in view:
                for old,text in regions.items():
                    self.envelope_serial+=1;token='CASNumericRegion'+str(self.envelope_serial)
                    leaves[token]=text
                    pattern=r'\b'+re.escape(old)+(r'\b' if not old.endswith(')') else '')
                    view=re.sub(pattern,lambda _:token+'()',view)
                self.remember_selector_literal(expression,view,leaves,arm_bounds)
        return original,pure

    def propagate_closed_identity(self,original,replacement):
        """Transfer whole-root proofs after a proven pure control rewrite.

        Selected branch bodies do not inherit global dtype/range proofs:
        those may hold only under a branch's guard. No runtime aliases exist.
        """
        before=syntax(original);old=self.key(before)
        if old not in self.converted_regions:raise ValueError("Control rewrite requires a closed numeric frontier")
        after=syntax(replacement);new=self.key(after)
        if old in self.sign_projections:self.sign_projections[new]=self.sign_projections[old]
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
            # A proved +/-zero can change the zero sign, but cannot move a
            # finite Half/F32 operand outside its own representable format.
            # Keep the sum/subtraction; remove only the redundant casts.
            ab,bb=self.bounds(node.left),self.bounds(node.right)
            if bb.minimum==bb.maximum==0 and a in ('half','f32'):return a
            if ab.minimum==ab.maximum==0 and b in ('half','f32'):return b
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
        # A proved dyadic grid with at most 24 significant bits fits F32
        # exactly. This preserves the F64 operation order; it only removes
        # a cast which cannot change either value or zero sign.
        d=self.bounds(node)
        if d is not None and d.quantum is not None and d.quantum>=-149:
            peak=max(abs(d.minimum),abs(d.maximum))
            if peak<=3.4028234663852886e38 and Fraction(peak)/(Fraction(2)**d.quantum)<=2**24:return 'f32'
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

    def same_sign_operand(self,node):
        """Smaller finite producer with the same sign bit, including -0.

        A strictly positive finite factor/divisor cannot change the IEEE
        sign. Unknown intervals, zero factors and nonfinite results are
        barriers. No magnitude or branch condition is substituted here.
        """
        if self.bounds(node) is None:return node
        if isinstance(node,ast.Call) and node.func.id in ("R16","R32") and len(node.args)==1:
            return self.same_sign_operand(node.args[0])
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Div)):
            right=self.bounds(node.right)
            if right is not None and right.minimum>0:
                return self.same_sign_operand(node.left)
            if isinstance(node.op,ast.Mult):
                left=self.bounds(node.left)
                if left is not None and left.minimum>0:
                    return self.same_sign_operand(node.right)
        return node

    def encoded_word_is_exact_integer(self,node):
        """A <=42-bit significand leaves >=11 trailing raw-word zeros.

        Even a signed raw word then fits the 53-bit F64 integer grid. This
        is a compile-time certificate, never a runtime approximation.
        """
        bounds=self.bounds(node)
        if bounds is None:return False
        kind=self.value_kind(node)
        if kind in ("half","f32"):return True
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):
            widths=[]
            for operand in (node.left,node.right):
                kind=self.value_kind(operand)
                if kind in ('half','f32'):widths.append(11 if kind=='half' else 24)
                else:
                    domain=self.bounds(operand)
                    maximum=None if domain is None else max(abs(domain.minimum),abs(domain.maximum))
                    if domain is None or domain.quantum is None:
                        widths=None;break
                    widths.append(0 if not maximum else min(53,binary_exponent(maximum)-domain.quantum+1))
            # Multiplying p- and q-bit finite significands uses at most p+q
            # bits; F64 rounding cannot increase that bound. The enclosure
            # and dyadic quantum also prove widths for untyped F64 operands.
            if widths is not None and sum(widths)<=42:return True
        maximum=max(abs(bounds.minimum),abs(bounds.maximum))
        return not maximum or (bounds.quantum is not None and binary_exponent(maximum)-bounds.quantum+1<=42)

    def remember_rms_guard(self,product,input_value,mean_value,inverse_value):
        """Adapter certificate for its actual ordered RMS inverse constructor.

        The adapter must supply s and the value it produced with
        R32(1/R32(sqrt(s))); a positive arbitrary factor is not sufficient.
        Only current-session immutable inputs and finite F32 means qualify.
        """
        from direct_sympy_rms_guard import bindings
        guard=bindings(input_value,mean_value,inverse_value,self)
        if guard is None:return False
        node,keys=self.analyze_expression(product)
        if not isinstance(node,ast.Call) or node.func.id!='R32' or len(node.args)!=1:return False
        raw=node.args[0]
        if not isinstance(raw,ast.BinOp) or not isinstance(raw.op,ast.Mult):return False
        self.rms_guards[keys.get(raw,self.key(raw))]=guard
        return True

    def constant_rounding_cell(self,node):
        """Prove a whole finite producer belongs to one IEEE rounding cell.

        Round-to-nearest-even is monotone on each signed finite interval.
        Identical endpoint *payloads*, not numeric equality, certify every
        enclosed result. An interval containing zero alone cannot establish
        its sign. Unknown/impure calls are never discarded by this proof.
        """
        if not isinstance(node,ast.Call) or node.func.id not in ('R16','R32') or len(node.args)!=1:return None
        source=node.args[0];bound=self.bounds(source)
        if bound is None:return None
        from direct_sympy_synchronize import GUARD_PURE
        for child in ast.walk(source):
            if isinstance(child,ast.Call) and child.func.id not in GUARD_PURE and self.key(child) not in self.pure_numeric_regions:return None
        fmt='e' if node.func.id=='R16' else 'f'
        try:
            low=struct.pack(fmt,bound.minimum);high=struct.pack(fmt,bound.maximum)
        except (OverflowError,ValueError):return None
        if low!=high:return None
        result=struct.unpack(fmt,low)[0]
        if not math.isfinite(result):return None
        if result==0 and bound.minimum<=0<=bound.maximum and not self.no_negative_zero(source):return None
        return result

    def half_update_is_invisible(self,value,update):
        """A strict Half-cell proof, including the smaller binade neighbor.

        Half +/- Half followed by F32 then Half equals direct Half storage.
        Stay strictly inside both neighbors; unknown types and zero-crossing
        enclosures cannot prove that the center is a nonzero Half value.
        """
        if self.value_kind(value)!='half' or self.value_kind(update)!='half':return False
        radius=self.half_cell_radius(value);delta=self.bounds(update)
        return radius is not None and delta is not None and max(abs(delta.minimum),abs(delta.maximum))<radius

    def half_cell_radius(self,value):
        if self.value_kind(value)!='half':return None
        source=self.bounds(value)
        if source is None:return None
        minimum=source.minimum if source.minimum>0 else -source.maximum if source.maximum<0 else 0
        if not minimum:return None
        # Subnormal Half spacing stops shrinking at 2**-24.
        return 2**max(-25,binary_exponent(minimum)-12)

    def f32_update_is_invisible(self,value,update):
        """Strict F32 cell containment for finite F32 operands.

        Use the smaller neighbor at binade boundaries and the subnormal
        spacing. F32 operands also prevent a prior F64 sum from crossing
        an odd midpoint through an unmodelled fine-grid update.
        """
        if self.value_kind(value) not in ('half','f32') or self.value_kind(update) not in ('half','f32'):
            return False
        source=self.bounds(value);delta=self.bounds(update)
        if source is None or delta is None:return False
        minimum=source.minimum if source.minimum>0 else -source.maximum if source.maximum<0 else 0
        if not minimum:return False
        radius=2**max(-150,binary_exponent(minimum)-25)
        return max(abs(delta.minimum),abs(delta.maximum))<radius

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
                    # Use the smallest binade reached by this interval. A
                    # zero crossing requires the format's subnormal grid.
                    # Rounding also preserves a coarser source dyadic grid.
                    precision,minimum_quantum=(11,-24) if fmt=='e' else (24,-149)
                    q=max(minimum_quantum,binary_exponent(minimum)-(precision-1)) if minimum else minimum_quantum
                    if d.quantum is not None:q=max(q,d.quantum)
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
            return self._close_completed(compact,regions,len(expression))
        return self._close(expression)

    def compose_closed(self,template,bindings):
        """Close a new operation before allocating its completed operands.

        References exist only during compilation. Each substitution runs CAS
        to a fixed point; every literal is restored before returning a string.
        Only proofs owned by this session are admissible.
        """
        if re.search(r"\bCAS(?:Numeric|Stable)Region[0-9]+\b",template):
            raise ValueError("Reserved compiler numeric placeholder")
        tree=syntax(template)
        if any(isinstance(node,ast.Call) and node.func.id=="Piecewise" for node in ast.walk(tree)):
            raise ValueError("Compose branch arms separately in their own context")
        estimated=len(template)
        for name,literal in bindings.items():
            if not re.fullmatch(r"X[1-9][0-9]*",name):
                raise ValueError("Substitution variables must be Xn")
            if name in self.domains or not re.search(r"\b"+name+r"\b",template):
                raise ValueError("Composition bindings must be distinct from original inputs and used")
            if literal not in self.closed_literals or literal not in self.closed_literal_keys:
                raise ValueError("Composition requires completed literals from this session")
            estimated+=len(re.findall(r"\b"+name+r"\b",template))*(len(literal)+2-len(name))
        return self._close_completed(template,bindings,estimated,substitute=True)

    def close_frontier_candidates(self,compact,baseline,protected,pure_functions,views,measure):
        """Try one certified producer frontier after the substitution fixed
        point. Close each selected arm with its own output interval/type,
        then admit only an improvement in the fully restored expression.
        """
        from direct_sympy_synchronize import frontier
        best=baseline;best_size=measure(baseline)
        for token,view in views.items():
            literal=protected[token];stored=self.selector_literals.get(literal)
            if stored is None or not any(value is not None for value in stored[3]):continue
            whole,kind,_=self.closed_literals[literal]
            if whole is None or kind is None:continue
            arms=frontier(ast.unparse(view),pure_functions=pure_functions)
            if arms is None or len(arms)!=len(stored[3]) or len(arms)>16:continue
            remaining=dict(self.domains);facts=self.branch_facts;result=[];result_bounds=[]
            marker_key=self.key(syntax(token+'()'))
            for (selected,condition),local in zip(arms,stored[3]):
                own_facts=self.compiler.branch_facts.assume(condition,True,facts)
                own=refine(remaining,condition,True) if own_facts is not None else None
                low=whole.minimum if local is None else max(whole.minimum,local.minimum)
                high=whole.maximum if local is None else min(whole.maximum,local.maximum)
                if low>high:own=None
                bounds=whole if local is None or own is None else FiniteSource(low,high,max(whole.quantum,local.quantum) if whole.quantum is not None and local.quantum is not None else whole.quantum if whole.quantum is not None else local.quantum)
                if own is not None:
                    selected_key=self.key(selected)
                    with self.branch_context(own,{selected_key:bounds},own_facts):
                        self.converted_regions.add(selected_key)
                        if kind=='half':self.half_values.add(selected_key)
                        self.f32_values.add(selected_key)
                        body=re.sub(r'\b'+token+r'\(\)',lambda _: '('+ast.unparse(selected)+')',compact)
                        # Keep original correlation certificates on enclosing
                        # operations, intersected with tighter arm bounds.
                        tree=syntax(body)
                        for node,old in self.signatures.translated_keys(tree,{selected_key:marker_key}).items():
                            key=self.key(node)
                            if key==selected_key:continue
                            previous=self.completed.get(old)
                            if previous is not None:
                                inferred=self.bounds(node)
                                if inferred is not None:
                                    previous=FiniteSource(max(previous.minimum,inferred.minimum),min(previous.maximum,inferred.maximum),previous.quantum)
                                self.completed[key]=previous
                            if old in self.half_values:self.half_values.add(key)
                            if old in self.f32_values:self.f32_values.add(key)
                        body=self._close(self.compiler.stabilize(body,self.domains,facts=own_facts))
                        result_bounds.append(self.bounds(syntax(body)))
                    result.append('('+body+', '+ast.unparse(condition)+')')
                facts=self.compiler.branch_facts.assume(condition,False,facts)
                remaining=refine(remaining,condition,False)
                if remaining is None or facts is None:break
            if not result:continue
            candidate=self.compiler.stabilize('Piecewise('+', '.join(result)+')',self.domains)
            candidate_size=measure(candidate)
            admitted=candidate_size<best_size
            self.frontier_events.append((token,best_size,candidate_size,len(result),admitted))
            if admitted:
                self.propagate_closed_identity(baseline,candidate)
                extracted=frontier(candidate,pure_functions=pure_functions)
                if extracted is not None and len(extracted)==len(result_bounds):self.frontier_bounds[self.key(syntax(candidate))]=tuple(result_bounds)
                best,best_size=candidate,candidate_size
        return best

    def _close_completed(self,compact,regions,input_characters,*,substitute=False):
        from direct_sympy_synchronize import GUARD_PURE
        protected={};literal_keys={};pure_functions=[]
        for old,text in regions.items():
            self.envelope_serial+=1;token="CASNumericRegion"+str(self.envelope_serial)
            marker=syntax(token+"()");key=self.key(marker)
            bounds,kind,positive_zero=self.closed_literals[text]
            if bounds is not None:self.completed[key]=bounds
            if kind=="half":self.half_values.add(key)
            if kind in ("half","f32"):self.f32_values.add(key)
            if positive_zero:self.no_negative_zero_values.add(key)
            self.converted_regions.add(key);protected[token]=text
            self.compiler.copy_completed_word_root(text,token+'()',self.domains)
            literal_keys[key]=self.closed_literal_keys[text]
            if self.closed_literal_pure[text]:
                self.pure_numeric_regions.add(key)
                pure_functions.append(token)
            if substitute:
                compact=self.compiler.substitute(compact,old,token+"()",self.domains)
            else:
                compact=re.sub(r"\b"+old+r"\b",token+"()",compact)
        # A previously compiled sign-only scalar can stand in for the sign
        # query on this producer. It never stands in for its magnitude.
        for token,text in list(protected.items()):
            projection=self.closed_sign_literals.get(text)
            if projection is None or projection not in self.closed_literals:continue
            self.envelope_serial+=1;alias="CASNumericRegion"+str(self.envelope_serial)
            protected[alias]=projection;pure_functions.append(alias)
            self.compiler.copy_completed_word_root(projection,alias+'()',self.domains)
            marker=syntax(alias+"()");key=self.key(marker)
            self.pure_numeric_regions.add(key)
            self.completed[key]=FiniteSource(0.0,0.0,-24)
            self.half_values.add(key);self.f32_values.add(key);self.converted_regions.add(key)
            literal_keys[key]=self.closed_literal_keys[projection]
            self.sign_projections[self.key(syntax(token+"()"))]=syntax("Bits64("+alias+"())")
        # Preserve correlation certificates on enclosing operations.
        # Independent interval arithmetic cannot recover these from the
        # bounds of individual completed literals (e.g. RMS products).
        virtual_tree=syntax(compact)
        def bind_rms_guard(guard):
            template,bindings=guard
            by_text={text:token for token,text in protected.items()}
            for name,text in bindings.items():
                if text in self.domains:
                    replacement=text
                else:
                    token=by_text.get(text)
                    if token is None:
                        self.envelope_serial+=1;token='CASNumericRegion'+str(self.envelope_serial)
                        protected[token]=text;by_text[text]=token
                        self.compiler.copy_completed_word_root(text,token+'()',self.domains)
                        bounds,kind,positive_zero=self.closed_literals[text]
                        marker=syntax(token+'()');key=self.key(marker)
                        if bounds is not None:self.completed[key]=bounds
                        if kind=='half':self.half_values.add(key)
                        if kind in ('half','f32'):self.f32_values.add(key)
                        if positive_zero:self.no_negative_zero_values.add(key)
                        self.converted_regions.add(key)
                        literal_keys[key]=self.closed_literal_keys[text]
                        if self.closed_literal_pure[text]:
                            self.pure_numeric_regions.add(key)
                            pure_functions.append(token)
                    replacement=token+'()'
                template=self.compiler.substitute(template,name,replacement,self.domains)
            return template,{}
        for node,original_key in self.signatures.translated_keys(virtual_tree,literal_keys).items():
            virtual_key=self.key(node)
            if original_key in self.completed:self.completed[virtual_key]=self.completed[original_key]
            if original_key in self.half_values:self.half_values.add(virtual_key)
            if original_key in self.f32_values:self.f32_values.add(virtual_key)
            if original_key in self.no_negative_zero_values:self.no_negative_zero_values.add(virtual_key)
            if original_key in self.converted_regions:self.converted_regions.add(virtual_key)
            if original_key in self.rms_guards:
                self.rms_guards[virtual_key]=bind_rms_guard(self.rms_guards[original_key])
        result=self._close(compact)
        # Search the newly closed envelope before restoring large literals.
        # Cost is the real restored string, not the short placeholder text.
        # Equal selectors synchronize only after CAS has stabilized, and
        # selected arms never inherit whole-root numeric proofs.
        def expanded_size(candidate):
            size=self.compiler.expression_size(candidate)
            for token,text in protected.items():
                size+=len(re.findall(r"\b"+token+r"\(\)",candidate))*(self.compiler.expression_size(text)-len(token)-2)
            return size
        views=self.selector_views(protected,pure_functions)
        # New leaf aliases in the views also own the original whole-producer
        # proofs. Selected arm proofs are introduced only inside their guards.
        for token,text in protected.items():
            marker=syntax(token+'()');key=self.key(marker)
            self.compiler.copy_completed_word_root(text,token+'()',self.domains)
            bounds,kind,positive_zero=self.closed_literals[text]
            if bounds is not None:self.completed[key]=bounds
            if kind=='half':self.half_values.add(key)
            if kind in ('half','f32'):self.f32_values.add(key)
            if positive_zero:self.no_negative_zero_values.add(key)
            self.converted_regions.add(key)
        synchronized=self.compiler.synchronize(result,self.domains,pure_functions=pure_functions,measure=expanded_size,completed_views=views)
        if synchronized!=result and self.key(syntax(result)) in self.converted_regions:
            self.propagate_closed_identity(result,synchronized)
        result=synchronized
        baseline=result
        counters=('closed','activations_closed','square_roots_closed','pending','redundant','arithmetic_eliminated','reused_regions','visited_nodes')
        before={name:getattr(self,name) for name in counters}
        try:result=self.close_frontier_candidates(compact,result,protected,pure_functions,views,expanded_size)
        except ValueError as error:
            if 'budget' not in str(error).lower():raise
            self.frontier_events.append(('',expanded_size(result),None,0,False))
        finally:
            for name,value in before.items():setattr(self,name,value)
        synchronized=self.compiler.synchronize(result,self.domains,pure_functions=pure_functions,measure=expanded_size,completed_views=views) if result!=baseline else result
        if synchronized!=result and self.key(syntax(result)) in self.converted_regions:
            self.propagate_closed_identity(result,synchronized)
        result=synchronized
        virtual=syntax(result);original=self.key(virtual)
        pattern=r"\bCASNumericRegion[0-9]+\(\)"
        # The logical measure chooses the smallest fully substituted form.
        # Admission limits the string we actually restore in memory; using
        # the logical cost here would defeat compiler-only sharing entirely.
        expanded=len(result)+sum(len(re.findall(r"\b"+token+r"\(\)",result))*(len(text)-len(token)-2) for token,text in protected.items())
        self.numeric_envelopes.append((input_characters,len(compact),len(result),expanded,len(protected)))
        if expanded>self.compiler.max_characters:
            raise ValueError(f"Closed numeric envelope exceeds string budget before allocation: expandedCharacters={expanded} limit={self.compiler.max_characters}")
        selector_view=result
        sign_literal=None
        if not self.branch_depth and original in self.converted_regions and self.bounds(virtual) is not None:
            from direct_sympy_signs import project
            sign_view=simplify_words("Float64("+ast.unparse(project(virtual,self))+")",self.compiler,self.domains)
            sign_expanded=len(sign_view)+sum(len(re.findall(r"\b"+token+r"\(\)",sign_view))*(len(text)-len(token)-2) for token,text in protected.items())
            # Compare fully substituted costs with each other, then enforce
            # the separate resident allocation limit before restoring bytes.
            if sign_expanded<=self.compiler.max_characters and expanded_size(sign_view)<expanded_size(result):
                sign_literal=re.sub(pattern,lambda match:protected[match.group(0)[:-2]],sign_view)
        result=re.sub(pattern,lambda match:protected[match.group(0)[:-2]],result)
        if re.search(r"\bCASNumericRegion[0-9]+\b",result):raise ValueError("Numeric placeholder escaped restoration")
        # The virtual fixed point must retain its proof on the restored
        # whole root. Never infer a dtype from mere textual expansion.
        if original in self.converted_regions:
            arm_bounds=self.frontier_bounds.get(original,())
            self.remember_selector_literal(result,selector_view,protected,arm_bounds)
            literals={token+'()':text for token,text in protected.items()}
            self.restore_compact_literal(result,virtual,literals,self.bounds(virtual),self.value_kind(virtual),self.no_negative_zero(virtual),True,self.closed_literal_keys.copy(),self.closed_literal_pure.copy(),arm_bounds)
            # Root shape is unchanged by restoration. Whole-placeholder
            # roots already denote a previously registered literal.
            if isinstance(virtual,ast.Call) and virtual.func.id not in protected:
                self.compiler.register_completed_region(result,self.domains,virtual,word_closed=True)
        else:
            node=syntax(result);key=self.key(node)
            if original in self.completed:self.completed[key]=self.completed[original]
            if original in self.half_values:self.half_values.add(key)
            if original in self.f32_values:self.f32_values.add(key)
            if original in self.no_negative_zero_values:self.no_negative_zero_values.add(key)
            self.remember_closed_literal(result,node)
        if sign_literal is not None and sign_literal!=result:
            # The complete sign projection has exactly +/-zero as values.
            # Reuse original leaf proofs to intern its structural identity.
            sign_node=syntax(sign_view)
            self.restore_compact_literal(sign_literal,sign_node,literals,FiniteSource(0.0,0.0,-24),"half",False,True,self.closed_literal_keys.copy(),self.closed_literal_pure.copy())
            previous=self.closed_sign_literals.get(result)
            if previous is None or len(sign_literal)<len(previous):self.closed_sign_literals[result]=sign_literal
            bounds=self.bounds(virtual)
            if bounds.minimum==bounds.maximum==0 and self.closed_literal_pure.get(result,False):
                # The proved magnitude is zero. Its complete sign expression
                # is therefore the entire value, preserving both IEEE zeros.
                self.arithmetic_eliminated+=1
                return sign_literal
        return result

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
                projection=None
                if isinstance(node,ast.Call) and node.func.id in ('R16','R32','Silu16') and len(node.args)==1 and session.bounds(node) is not None:
                    from direct_sympy_signs import project
                    candidate=project(node,session)
                    text=ast.unparse(candidate)
                    if len(text)<=1048576 and not re.search(r'\b(?:R16|R32|Silu16|sqrt)\s*\(',text):projection=candidate
                result=super().visit(node)
                if projection is not None and not re.search(r'\b(?:R16|R32|Silu16|sqrt)\s*\(',ast.unparse(result)):session.sign_projections[session.key(result)]=projection
                return result

            def generic_visit(self,node):
                # Bounds may have cached the complete pre-substitution tree.
                # Invalidate each mutable parent around child replacement.
                session.signatures.invalidate(node)
                result=super().generic_visit(node)
                session.signatures.invalidate(result)
                return result

            def visit_Call(self,node):
                if node.func.id=='Piecewise':
                    # Completed elementary selectors have no boundary left
                    # to close. Avoid repeating CAS and copying proof tables
                    # for every already-lowered numerical classification.
                    if not any(isinstance(child,ast.Call) and child.func.id in ('R16','R32','Silu16') for child in ast.walk(node)):
                        return self.generic_visit(node)
                    # Classification conditions are ordered: each next arm
                    # inherits the negation of every previous condition.
                    remaining=dict(session.domains);proofs={};facts=session.branch_facts
                    arms=[]
                    for pair in node.args:
                        if not isinstance(pair,ast.Tuple) or len(pair.elts)!=2:raise ValueError('Piecewise requires (expression, condition) pairs')
                        body,condition=pair.elts
                        with session.branch_context(remaining,proofs,facts):
                            condition=self.visit(condition)
                        own_facts=session.compiler.branch_facts.assume(condition,True,facts)
                        own=refine(remaining,condition,True) if own_facts is not None else None
                        own_proofs=session.magnitude_guard_bounds(condition,True,proofs) if own is not None else None
                        if own is not None and own_proofs is not None:
                            with session.branch_context(own,own_proofs,own_facts):
                                # CAS before numerical closure, then CAS again
                                # before returning this arm to the parent.
                                text=simplify_arithmetic(ast.unparse(body),session)
                                rewritten=self.visit(syntax(text))
                                text=session.compiler.stabilize(ast.unparse(rewritten),session.domains,facts=own_facts)
                            arms.append(ast.Tuple(elts=[syntax(text),condition],ctx=ast.Load()))
                        facts=session.compiler.branch_facts.assume(condition,False,facts)
                        remaining=refine(remaining,condition,False)
                        proofs=session.magnitude_guard_bounds(condition,False,proofs)
                        if remaining is None or proofs is None or facts is None:break
                    if not arms:raise ValueError('No reachable conversion branch')
                    return ast.Call(func=ast.Name(id='Piecewise',ctx=ast.Load()),args=arms,keywords=[])
                constant=session.constant_rounding_cell(node)
                if constant is not None:
                    # Conditions have already narrowed this arm's domains.
                    # Eliminate the producer before visiting its descendants.
                    rewritten=syntax(session.compiler.stabilize(repr(constant),session.domains))
                    key=session.key(rewritten)
                    session.completed[key]=FiniteSource(constant,constant,quantum(Fraction(constant)))
                    session.f32_values.add(key)
                    if node.func.id=='R16' or session.value_kind(rewritten)=='half':session.half_values.add(key)
                    if constant!=0 or math.copysign(1,constant)>0:session.no_negative_zero_values.add(key)
                    session.arithmetic_eliminated+=1
                    return rewritten
                before=session.bounds(node)
                positive_zero=session.no_negative_zero(node)
                if node.func.id=='R32' and len(node.args)==1:
                    inner=node.args[0]
                    if isinstance(inner,ast.Call) and inner.func.id=='sqrt' and len(inner.args)==1:
                        from direct_sympy_sqrt import supported,expand
                        source=ast.unparse(inner.args[0])
                        if supported(source,session):
                            source=ast.unparse(self.visit(inner.args[0]))
                            rewritten=syntax(expand(source,session));key=session.key(rewritten)
                            if before is not None:session.completed[key]=before
                            session.f32_values.add(key)
                            if positive_zero:session.no_negative_zero_values.add(key)
                            session.closed+=1;session.square_roots_closed+=1
                            return rewritten
                if node.func.id=='Silu16' and len(node.args)==1:
                    from direct_sympy_silu import supported,expand
                    source=ast.unparse(node.args[0])
                    if supported(source,session):
                        source=ast.unparse(self.visit(node.args[0]))
                        # The outer close owns and restores any certified
                        # numeric literals. Stay in that same proof context;
                        # public close correctly rejects those reserved names.
                        text=session._close(expand(source,session))
                        rewritten=syntax(text);key=session.key(rewritten)
                        if before is not None:session.completed[key]=before
                        session.half_values.add(key);session.f32_values.add(key)
                        if positive_zero:session.no_negative_zero_values.add(key)
                        session.activations_closed+=1
                        return rewritten
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
                            sign=session.same_sign_operand(raw)
                            sign_expression=None if sign is raw else ast.unparse(sign)
                            from direct_sympy_silu import quadratic_subnormal_guard,quadratic_tandem_source
                            condition=quadratic_subnormal_guard(raw,session)
                            rms=session.rms_guards.get(session.key(raw))
                            if condition is None and rms is not None:
                                condition,bindings=rms
                                for name,literal in bindings.items():condition=session.compiler.substitute(condition,name,literal,session.domains)
                            polynomial=quadratic_tandem_source(raw,session)
                            magnitude=ast.unparse(raw)
                            exact_word=session.encoded_word_is_exact_integer(raw)
                            frontier=[]
                            text=lower_tandem(magnitude,certificate,session.compiler,session.domains,integer_word_exact=exact_word,sign_expression=sign_expression,small_condition=condition,frontier=frontier,facts=session.branch_facts)
                            if polynomial is not None:
                                candidate_frontier=[]
                                candidate=lower_tandem(polynomial[0],certificate,session.compiler,session.domains,integer_word_exact=True,sign_expression=polynomial[1],small_condition=condition,frontier=candidate_frontier,facts=session.branch_facts)
                                # Compare complete literal costs. A longer
                                # compiler envelope can still contain fewer
                                # copies of an expensive completed operand.
                                if session.compiler.expression_size(candidate)<session.compiler.expression_size(text):
                                    text,frontier=candidate,candidate_frontier
                            session.remember_frontier_bounds(text,frontier)
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
                        frontier=[]
                        from direct_sympy_signs import project
                        projection=project(rewritten.args[0],session)
                        sign_word=simplify_words(ast.unparse(projection),session.compiler,session.domains)
                        text=lower_finite_conversion(ast.unparse(rewritten.args[0]),rewritten.func.id,
                            source,session.compiler,session.domains,no_odd_f32_ties=no_odd_ties,integer_word_exact=session.encoded_word_is_exact_integer(node.args[0]),frontier=frontier,sign_word=sign_word,facts=session.branch_facts)
                        session.remember_frontier_bounds(text,frontier)
                        rewritten=syntax(text)
                        session.sign_projections[session.key(rewritten)]=syntax(sign_word)
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
        prior_key=self.key(tree)
        result=self.compiler.stabilize(ast.unparse(tree),self.domains)
        if prior_key in self.sign_projections:self.sign_projections[self.key(syntax(result))]=self.sign_projections[prior_key]
        if prior_key in self.frontier_bounds:self.frontier_bounds[self.key(syntax(result))]=self.frontier_bounds[prior_key]
        if not re.search(r"\bR(?:16|32)\s*\(",result):
            self.converted_regions.add(self.key(syntax(result)))
            self.remember_closed_literal(result)
        return result

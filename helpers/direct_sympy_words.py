"""Exact bit-word identities on mathematical strings, before the next producer.

Integer operations are unsigned 64-bit; reinterpretations preserve all bits.
No floating arithmetic is reassociated. Each accepted replacement returns
through mandatory factor/simplify under the caller's numeric context.
"""
import ast
import re
import struct
import sympy as sp
from direct_sympy_strings import syntax
from direct_sympy_signatures import StructuralSignatures

MASK=(1<<64)-1


def factor_word_polynomial(node,widths=None,*,signatures=None):
    """Factor an integer polynomial in Z/(2**64), with SymPy.

    Only U64Add/U64Mul are opened. Other typed words are independent
    polynomial leaves: floating arithmetic, masks, shifts and decisions
    are never reassociated or distributed. Integer polynomial identities
    remain true modulo 2**64 even when intermediate products overflow.
    """
    leaves={};originals={};counts={};operations=0
    signatures=StructuralSignatures() if signatures is None else signatures
    def encode(child):
        nonlocal operations
        value=integer(child)
        if value is not None:return sp.Integer(value),1
        if isinstance(child,ast.Call) and child.func.id in ('U64Add','U64Mul') and len(child.args)==2:
            operations+=1
            if operations>128:raise ValueError('Word polynomial search budget')
            (a,ta),(b,tb)=map(encode,child.args)
            terms=ta+tb if child.func.id=='U64Add' else ta*tb
            if terms>1024:raise ValueError('Word polynomial expansion search budget')
            return (sp.Add(a,b,evaluate=False) if child.func.id=='U64Add' else sp.Mul(a,b,evaluate=False)),terms
        if word_width(child,widths) is None:raise ValueError('Unsigned word proof required')
        key=signatures.key(child);symbol=leaves.get(key)
        if symbol is None:
            symbol=sp.Symbol('WordFactor'+str(len(leaves)),integer=True)
            leaves[key]=symbol;originals[symbol]=child
        counts[symbol]=counts.get(symbol,0)+1
        return symbol,1
    if not isinstance(node,ast.Call) or node.func.id not in ('U64Add','U64Mul'):return node
    try:value,_=encode(node)
    except ValueError:return node
    # Distinct opaque words cannot yield a common symbolic factor. Avoid
    # asking CAS to search their interiors, especially floating boundaries.
    if not any(count>1 for count in counts.values()):return node
    factored=sp.factor(value)
    simplified=sp.simplify(factored)
    candidate=sp.factor(simplified)
    def call(name,a,b):return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=[a,b],keywords=[])
    def decode(value):
        if value in originals:return originals[value]
        if value.is_Integer:return ast.Constant(value=int(value)&MASK)
        if value.func in (sp.Add,sp.Mul):
            args=list(map(decode,value.args));result=args.pop(0)
            for operand in args:result=call('U64Add' if value.func==sp.Add else 'U64Mul',result,operand)
            return result
        if value.func==sp.Pow and value.exp.is_Integer and 0<=int(value.exp)<=128:
            base=decode(value.base);exponent=int(value.exp);result=ast.Constant(value=1)
            while exponent:
                if exponent&1:result=base if integer(result)==1 else call('U64Mul',result,base)
                exponent>>=1
                if exponent:base=call('U64Mul',base,base)
            return result
        raise ValueError('Non-polynomial CAS result')
    try:replacement=decode(candidate)
    except ValueError:return node
    return replacement if len(ast.unparse(replacement))<len(ast.unparse(node)) else node


def factor_word_polynomials(node):
    """Apply the modular proof at every reachable integer subtree."""
    changes=0;signatures=StructuralSignatures()
    class Polynomials(ast.NodeTransformer):
        def generic_visit(self,child):
            # A replaced call can also change an enclosing arithmetic node,
            # tuple or argument list. Invalidate every edited ancestor.
            signatures.invalidate(child)
            result=super().generic_visit(child)
            signatures.invalidate(result)
            return result

        def visit_Call(self,child):
            nonlocal changes
            # Bottom-up edits change parent fields. Reuse signatures only
            # for untouched descendants; a rewritten parent is re-keyed.
            child=self.generic_visit(child)
            replacement=factor_word_polynomial(child,signatures=signatures)
            if replacement is not child:changes+=1
            return replacement
    result=Polynomials().visit(node)
    return result,changes


def integer(node):
    return node.value if isinstance(node,ast.Constant) and type(node.value) is int and 0<=node.value<=MASK else None


def word_width(node,widths=None):
    if isinstance(node,ast.Name):return (widths or {}).get(node.id)
    value=integer(node)
    if value is not None:return value.bit_length()
    if not isinstance(node,ast.Call):return None
    name=node.func.id
    if name in ("Bits64","U64FromF64") and len(node.args)==1:return 64
    if len(node.args)!=2:return None
    a,b=node.args
    if name=="U64Shr":
        width,shift=word_width(a,widths),integer(b)
        return max(0,width-shift) if width is not None and shift is not None and shift<64 else None
    wa,wb=word_width(a,widths),word_width(b,widths)
    if wa is None or wb is None:return None
    if name=="U64And":return min(wa,wb)
    if name=="U64Or":return max(wa,wb)
    if name in ("U64Add","U64Mul"):return 64
    return None


def float_source(node,domains,floats=()):
    if isinstance(node,ast.Name):return node.id in domains or node.id in floats
    if isinstance(node,ast.Constant):return type(node.value) is float
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return float_source(node.operand,domains,floats)
    if isinstance(node,ast.Call):
        return node.func.id in ("Float64","F64FromU64","R16","R32","sqrt","Silu16") and len(node.args)==1
    return False


def known_bits(node,known=None):
    """Conservative zero/one masks for explicit unsigned operations.

    Never infer bits of floating arithmetic. Addition and multiplication
    only preserve proved trailing zero bits, including modular overflow.
    """
    if isinstance(node,ast.Name):return (known or {}).get(node.id,(0,0))
    value=integer(node)
    if value is not None:return MASK^value,value
    if not isinstance(node,ast.Call) or len(node.args)!=2:return 0,0
    name=node.func.id
    if name not in ('U64And','U64Or','U64Shr','U64Add','U64Mul'):return 0,0
    if word_width(node,{name:64 for name in (known or {})}) is None:return 0,0
    a,b=node.args;za,oa=known_bits(a,known)
    if name=='U64Shr':
        shift=integer(b)
        if shift is None or shift>=64:return 0,0
        return (za>>shift)|((MASK<<(64-shift))&MASK),oa>>shift
    zb,ob=known_bits(b,known)
    if name=='U64And':return za|zb,oa&ob
    if name=='U64Or':return za&zb,oa|ob
    def trailing(mask):
        unknown=MASK^mask
        return 64 if not unknown else (unknown&-unknown).bit_length()-1
    count=min(trailing(za),trailing(zb)) if name=='U64Add' else min(64,trailing(za)+trailing(zb))
    return (1<<count)-1,0


def reduce_call(node,domains,widths=None,floats=(),known=None):
    name=node.func.id
    if len(node.args)==1:
        value=node.args[0]
        if name=='Bits64':
            sign=1
            if isinstance(value,ast.UnaryOp) and isinstance(value.op,(ast.UAdd,ast.USub)):
                sign=-1 if isinstance(value.op,ast.USub) else 1;value=value.operand
            if isinstance(value,ast.Constant) and type(value.value) is float:
                literal=value.value if sign==1 else -value.value
                return ast.Constant(value=struct.unpack('>Q',struct.pack('>d',literal))[0])
        if name=='Float64' and integer(value) is not None:
            literal=struct.unpack('>d',struct.pack('>Q',value.value))[0]
            # Keep nonfinite payloads encoded; textual nan/inf would lose
            # their exact bits and are outside the literal grammar.
            import math
            if math.isfinite(literal):return ast.Constant(value=literal)
    if len(node.args)==1 and isinstance(node.args[0],ast.Call):
        inner=node.args[0]
        if len(inner.args)==1:
            if name=="Bits64" and inner.func.id=="Float64" and word_width(inner.args[0],widths) is not None:return inner.args[0]
            if name=="Float64" and inner.func.id=="Bits64" and float_source(inner.args[0],domains,floats):return inner.args[0]
    if len(node.args)!=2 or name not in ("U64And","U64Or","U64Add","U64Mul","U64Shr"):return node
    a,b=node.args;wa,wb=word_width(a,widths),word_width(b,widths)
    if wa is None or wb is None:return node
    av,bv=integer(a),integer(b)
    if name=="U64Shr" and (bv is None or bv>=64):return node
    if av is not None and bv is not None:
        value={"U64And":lambda:av&bv,"U64Or":lambda:av|bv,"U64Add":lambda:(av+bv)&MASK,"U64Mul":lambda:(av*bv)&MASK,"U64Shr":lambda:av>>bv}[name]()
        return ast.Constant(value=value)
    if bv==0:
        return ast.Constant(value=0) if name in ("U64And","U64Mul") else a
    if av==0 and name in ("U64And","U64Or","U64Add","U64Mul"):
        return ast.Constant(value=0) if name in ("U64And","U64Mul") else b
    if name in ("U64And","U64Or") and type(a) is type(b):
        possible=not isinstance(a,ast.Call) or a.func.id==b.func.id
        if possible and (a is b or ast.dump(a)==ast.dump(b)):return a
        # Factor the same unsigned word under two constant masks. This is
        # exact bit algebra, including NaN payloads and signed-zero fields.
        if isinstance(a,ast.Call) and isinstance(b,ast.Call) and a.func.id==b.func.id=="U64And" and len(a.args)==len(b.args)==2:
            ma,mb=integer(a.args[1]),integer(b.args[1])
            if ma is not None and mb is not None and ast.dump(a.args[0])==ast.dump(b.args[0]):
                mask=ma|mb if name=="U64Or" else ma&mb
                return ast.Call(func=ast.Name(id="U64And",ctx=ast.Load()),args=[a.args[0],ast.Constant(value=mask)],keywords=[])
    if name=="U64And" and bv is not None:
        zero,one=known_bits(a,known)
        if ((MASK^bv)&(MASK^zero))==0:return a
        if (bv&(MASK^(zero|one)))==0:return ast.Constant(value=bv&one)
        if bv&((1<<wa)-1)==(1<<wa)-1:return a
        if isinstance(a,ast.Call) and a.func.id=="U64And" and len(a.args)==2:
            previous=integer(a.args[1])
            if previous is not None:return ast.Call(func=ast.Name(id="U64And",ctx=ast.Load()),args=[a.args[0],ast.Constant(value=previous&bv)],keywords=[])
    if name=='U64Or' and bv is not None:
        zero,one=known_bits(a,known)
        if bv&(MASK^one)==0:return a
        if (MASK^bv)&(MASK^(zero|one))==0:return ast.Constant(value=bv|one)
    if name=="U64Shr" and bv is not None and isinstance(a,ast.Call) and a.func.id=="U64And" and len(a.args)==2:
        mask=integer(a.args[1])
        if mask is not None:
            shifted=ast.Call(func=node.func,args=[a.args[0],b],keywords=[])
            return ast.Call(func=ast.Name(id="U64And",ctx=ast.Load()),args=[shifted,ast.Constant(value=mask>>bv)],keywords=[])
    return node


def simplify_words(expression,compiler,domains):
    # Completed producers already have their numeric word identities closed.
    # Keep their literal strings in the output, but expose a Float64 root so
    # a newly adjacent Bits64 can still cancel it. The hidden argument owns
    # an exact unsigned-word width proof; it is not a runtime variable.
    if len(expression)>compiler.max_characters:raise ValueError("Word string exceeds input budget")
    if re.search(r"\bCASWordPayload[0-9]+\b",expression):raise ValueError("Reserved compiler word placeholder")
    compact,regions=compiler.compact_regions(expression,compiler.context(domains))
    if any((compiler.context(domains),text) not in compiler._region_roots for text in regions.values()):
        compact,regions=expression,{}
    payloads={};payload_roots={};widths={};floats=set();known={}
    for name,text in regions.items():
        root=compiler._region_roots.get((compiler.context(domains),text))
        if root is None:continue
        function,width=root
        if function in ("Float64","F64FromU64","R16","R32","sqrt","Silu16"):floats.add(name)
        if function=="Float64" and width is not None:
            payload="CASWordPayload"+str(len(payloads))
            stored=compiler._region_word_payloads.get((compiler.context(domains),text))
            payloads[payload]=text[text.index('(')+1:-1] if stored is None else stored; widths[payload]=width
            known[payload]=known_bits(syntax(payloads[payload]))
            payload_roots[payload]=text
            compact=re.sub(r"\b"+name+r"\b","Float64("+payload+")",compact)
    def restore(text):
        # Preserve an unchanged completed value as its original literal or
        # compile-only marker. Only a cancelled outer reinterpretation needs
        # its proven word payload substituted into the new expression.
        for payload,original in payload_roots.items():
            text=re.sub(r'\bFloat64\(\s*'+payload+r'\s*\)',lambda _:original,text)
        text=re.sub(r"\bCASWordPayload[0-9]+\b",lambda match:payloads[match.group(0)],text) if payloads else text
        return re.sub(r"\bCASStableRegion[0-9]+\b",lambda match:regions[match.group(0)],text) if regions else text
    current=compact
    for _ in range(32):
        changes=0
        class Words(ast.NodeTransformer):
            def visit_Call(self,node):
                nonlocal changes
                original=self.generic_visit(node)
                replacement=reduce_call(original,domains,widths,floats,known)
                if replacement is original:return original
                changes+=1
                return syntax(compiler.stabilize("("+ast.unparse(replacement)+")",domains))
        candidate=ast.unparse(Words().visit(syntax(current)))
        if not changes:
            result=restore(compiler.stabilize(candidate,domains))
            if len(result)>compiler.max_characters:raise ValueError("Word string exceeds output budget")
            return result
        current=candidate
    raise ValueError("Word identities did not stabilize; next producer forbidden")

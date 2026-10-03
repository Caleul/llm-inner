"""Exact bit-word identities on mathematical strings, before the next producer.

Integer operations are unsigned 64-bit; reinterpretations preserve all bits.
No floating arithmetic is reassociated. Each accepted replacement returns
through mandatory factor/simplify under the caller's numeric context.
"""
import ast
import re
from direct_sympy_strings import syntax

MASK=(1<<64)-1


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


def reduce_call(node,domains,widths=None,floats=()):
    name=node.func.id
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
        if bv&((1<<wa)-1)==(1<<wa)-1:return a
        if isinstance(a,ast.Call) and a.func.id=="U64And" and len(a.args)==2:
            previous=integer(a.args[1])
            if previous is not None:return ast.Call(func=ast.Name(id="U64And",ctx=ast.Load()),args=[a.args[0],ast.Constant(value=previous&bv)],keywords=[])
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
    payloads={};widths={};floats=set()
    for name,text in regions.items():
        root=compiler._region_roots.get((compiler.context(domains),text))
        if root is None:continue
        function,width=root
        if function in ("Float64","F64FromU64","R16","R32","sqrt","Silu16"):floats.add(name)
        if function=="Float64" and width is not None:
            payload="CASWordPayload"+str(len(payloads))
            payloads[payload]=text[text.index('(')+1:-1];widths[payload]=width
            compact=re.sub(r"\b"+name+r"\b","Float64("+payload+")",compact)
    def restore(text):
        text=re.sub(r"\bCASWordPayload[0-9]+\b",lambda match:payloads[match.group(0)],text) if payloads else text
        return re.sub(r"\bCASStableRegion[0-9]+\b",lambda match:regions[match.group(0)],text) if regions else text
    current=compact
    for _ in range(32):
        changes=0
        class Words(ast.NodeTransformer):
            def visit_Call(self,node):
                nonlocal changes
                original=self.generic_visit(node)
                replacement=reduce_call(original,domains,widths,floats)
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

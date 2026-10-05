"""Exact arithmetic identities before conversion expansion, preserving zero signs."""
import ast
import copy
import math
import re
from fractions import Fraction

import sympy
from direct_sympy_strings import Domain,certificate,symbolic,syntax


def linear_zero_signs_agree(original,candidate,leaves):
    """Exact homogeneous one-variable islands can only vanish at input zero.

    Caller proves all operations exact independently. A nonzero linear
    coefficient excludes cancellation to zero at every nonzero input. The
    two remaining IEEE cases are checked in original operation order, without
    evaluating model inputs or assuming that real equality preserves -0.
    """
    if len(leaves)!=1:return False
    name=next(iter(leaves));variable=sympy.Symbol(name,real=True)
    try:
        expressions=[symbolic(node) for node in (original,candidate)]
        if any(expression.free_symbols!={variable} for expression in expressions):return False
        polynomials=[sympy.Poly(expression,variable) for expression in expressions]
        if polynomials[0]!=polynomials[1] or polynomials[0].degree()!=1 or polynomials[0].nth(0)!=0:return False
    except (ValueError,sympy.PolynomialError):return False
    def evaluate(node,zero):
        if isinstance(node,ast.Name) and node.id==name:return zero
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return float(node.value)
        if isinstance(node,ast.UnaryOp):
            value=evaluate(node.operand,zero)
            if isinstance(node.op,ast.UAdd):return value
            if isinstance(node.op,ast.USub):return -value
        if isinstance(node,ast.BinOp):
            a,b=evaluate(node.left,zero),evaluate(node.right,zero)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow) and b.is_integer() and 0<=b<=8:
                result=1.0
                for _ in range(int(b)):result*=a
                return result
        raise ValueError('Unsupported zero-sign proof operation')
    try:
        for zero in (0.0,-0.0):
            a,b=evaluate(original,zero),evaluate(candidate,zero)
            if a!=0 or b!=0 or math.copysign(1,a)!=math.copysign(1,b):return False
        return True
    except (ValueError,ZeroDivisionError,OverflowError):return False


def factor_certified(expression,session):
    """Factor exact F64 islands with scoped, compiler-only numerical atoms.

    Closed producers keep their own operation order. Only the surrounding
    arithmetic is reassociated, after proving every original and proposed
    operation exact on the current dyadic domains. No atoms survive emission.
    """
    tree=syntax(expression)
    if not isinstance(tree,(ast.BinOp,ast.UnaryOp)):return expression
    # Large closed payloads must not be parsed by CAS as polynomials. This
    # pass is optional; normal mandatory stabilization still handles them.
    if sum(1 for _ in ast.walk(tree))>4096:return expression
    domains=dict(session.domains);leaves={};names={};failed=False

    class Atoms(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal failed
            name=node.func.id
            pure=name in {'R16','R32','Silu16','sqrt','Float64'} or re.fullmatch(r'(?:CompileValue|CASNumericRegion)[0-9]+',name)
            bound=session.bounds(node) if pure else None
            if bound is None or bound.quantum is None:
                failed=True;return node
            key=session.key(node)
            if key not in names:
                index=len(names)
                atom='CASExactLeaf'+str(index)
                while atom in domains or any(isinstance(child,ast.Name) and child.id==atom for child in ast.walk(tree)):
                    index+=1;atom='CASExactLeaf'+str(index)
                names[key]=atom;leaves[atom]=node
                domains[atom]=Domain(Fraction(bound.minimum),Fraction(bound.maximum),bound.quantum,session.no_negative_zero(node))
            return ast.Name(id=names[key],ctx=ast.Load())

    compact=Atoms().visit(copy.deepcopy(tree))
    if failed or not leaves:return expression
    def occurrences(node):
        return sum(isinstance(child,ast.Name) and child.id in leaves for child in ast.walk(node))
    copies=occurrences(compact)
    if copies<=len(leaves):return expression
    original=certificate(compact,domains,'f64')
    if original is None:return expression
    factored=sympy.factor(symbolic(compact))
    candidate=syntax(sympy.sstr(sympy.factor(sympy.simplify(factored))))
    replacement=certificate(candidate,domains,'f64')
    if replacement is None:return expression
    # A shorter spelling of a coefficient can obscure later numerical
    # kernels and grow their closed output. This pass addresses duplication:
    # require fewer producer occurrences, not merely shorter resident text.
    if occurrences(candidate)>=copies:return expression
    if original.excludes_negative_zero and not replacement.excludes_negative_zero:
        # Both arithmetic islands are already proved exact and finite.
        # The original can only produce +0 at a real zero. Restore that
        # sign explicitly when CAS drops the reduction's +0 seed; every
        # nonzero value stays unchanged. No extra branch or leaf copy.
        candidate=ast.BinOp(left=candidate,op=ast.Add(),right=ast.Constant(value=0.0))
        replacement=certificate(candidate,domains,'f64')
        if replacement is None or not replacement.excludes_negative_zero:return expression
    if not (original.excludes_negative_zero and replacement.excludes_negative_zero):
        if not linear_zero_signs_agree(compact,candidate,leaves):return expression

    class Restore(ast.NodeTransformer):
        def visit_Name(self,node):return copy.deepcopy(leaves[node.id]) if node.id in leaves else node

    result=ast.unparse(Restore().visit(candidate))
    return result if session.compiler.expression_size(result)<session.compiler.expression_size(expression) else expression


def simplify_arithmetic(expression,session):
    class Arithmetic(ast.NodeTransformer):
        def visit(self,node):
            if session.key(node) in session.converted_regions:return node
            return super().visit(node)

        def generic_visit(self,node):
            session.signatures.invalidate(node)
            result=super().generic_visit(node)
            session.signatures.invalidate(result)
            return result

        def accept(self,original,replacement):
            bounds=session.bounds(original)
            kind=session.value_kind(original)
            positive_zero=session.no_negative_zero(original)
            result=syntax(session.compiler.stabilize("("+ast.unparse(replacement)+")",session.domains))
            # The admitted replacement may expose a stronger exact grid
            # than the original cast's format-wide enclosure. Preserve
            # both proofs of the same value before the next dependency.
            actual=session.bounds(result)
            if bounds is not None and actual is not None:
                from direct_sympy_conversions import FiniteSource
                low,high=max(bounds.minimum,actual.minimum),min(bounds.maximum,actual.maximum)
                magnitude=max(bounds.minimum_magnitude,actual.minimum_magnitude)
                if low>high or magnitude>max(abs(low),abs(high)):raise ValueError('Equivalent arithmetic enclosures disagree')
                known=[q for q in (bounds.quantum,actual.quantum) if q is not None]
                bounds=FiniteSource(low,high,max(known) if known else None,magnitude)
            key=session.key(result)
            if bounds is not None:session.completed[key]=bounds
            if kind=="half":session.half_values.add(key)
            if kind in ("half","f32"):session.f32_values.add(key)
            if positive_zero:session.no_negative_zero_values.add(key)
            session.arithmetic_eliminated+=1
            return result

        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            a,b=session.constant(node.left),session.constant(node.right)
            if isinstance(node.op,ast.Add):
                if b==0 and math.copysign(1,b)<0 and session.bounds(node.left) is not None:return self.accept(node,node.left)
                if a==0 and math.copysign(1,a)<0 and session.bounds(node.right) is not None:return self.accept(node,node.right)
                if b==0 and math.copysign(1,b)>0 and session.no_negative_zero(node.left):return self.accept(node,node.left)
                if a==0 and math.copysign(1,a)>0 and session.no_negative_zero(node.right):return self.accept(node,node.right)
            if isinstance(node.op,ast.Mult):
                if b==1 and session.bounds(node.left) is not None:return self.accept(node,node.left)
                if a==1 and session.bounds(node.right) is not None:return self.accept(node,node.right)
            if isinstance(node.op,ast.Div) and b==1 and session.bounds(node.left) is not None:return self.accept(node,node.left)
            source=ast.unparse(node)
            factored=factor_certified(source,session)
            if factored!=source:return self.accept(node,syntax(factored))
            return node

        def visit_Call(self,node):
            constant=session.constant_rounding_cell(node)
            if constant is not None:return self.accept(node,ast.Constant(value=constant))
            scale=session.unit_grid_rounding_scale(node)
            if scale is not None:return self.accept(node,scale)
            node=self.generic_visit(node)
            if node.func.id=='R32' and len(node.args)==1:
                inner=node.args[0]
                if isinstance(inner,ast.BinOp) and isinstance(inner.op,(ast.Add,ast.Sub)):
                    if session.f32_update_is_invisible(inner.left,inner.right):return self.accept(node,inner.left)
                    if isinstance(inner.op,ast.Add) and session.f32_update_is_invisible(inner.right,inner.left):return self.accept(node,inner.right)
            if node.func.id=='R16' and len(node.args)==1:
                inner=node.args[0]
                if isinstance(inner,ast.Call) and inner.func.id=='R32' and len(inner.args)==1:
                    inner=inner.args[0]
                if isinstance(inner,ast.BinOp) and isinstance(inner.op,(ast.Add,ast.Sub)):
                    a,b=inner.left,inner.right
                    if session.half_update_is_invisible(a,b):return self.accept(node,a)
            if node.func.id in ("R16","R32") and len(node.args)==1:
                # Positive F32 sqrt is lowered as one fused boundary. An
                # exact-format proof alone must not expose a standalone,
                # still-unexpanded sqrt (including either signed zero).
                if node.func.id=='R32' and isinstance(node.args[0],ast.Call) and node.args[0].func.id=='sqrt':return node
                kind=session.value_kind(node.args[0])
                if session.bounds(node.args[0]) is not None and ((node.func.id=="R32" and kind in ("half","f32")) or (node.func.id=="R16" and kind=="half")):
                    return self.accept(node,node.args[0])
            return node

    current=expression
    for _ in range(32):
        before=session.arithmetic_eliminated
        result=ast.unparse(Arithmetic().visit(syntax(current)))
        if before==session.arithmetic_eliminated:
            factored=factor_certified(result,session)
            if factored!=result:
                session.arithmetic_eliminated+=1
                current=factored
                continue
            return session.compiler.stabilize(result,session.domains)
        current=result
    raise ValueError("Arithmetic identities did not stabilize; next dependency forbidden")

"""Exact arithmetic identities before conversion expansion, preserving zero signs."""
import ast
import math
from direct_sympy_strings import syntax


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
            return node

        def visit_Call(self,node):
            constant=session.constant_rounding_cell(node)
            if constant is not None:return self.accept(node,ast.Constant(value=constant))
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
                kind=session.value_kind(node.args[0])
                if session.bounds(node.args[0]) is not None and ((node.func.id=="R32" and kind in ("half","f32")) or (node.func.id=="R16" and kind=="half")):
                    return self.accept(node,node.args[0])
            return node

    current=expression
    for _ in range(32):
        before=session.arithmetic_eliminated
        result=ast.unparse(Arithmetic().visit(syntax(current)))
        if before==session.arithmetic_eliminated:return session.compiler.stabilize(result,session.domains)
        current=result
    raise ValueError("Arithmetic identities did not stabilize; next dependency forbidden")

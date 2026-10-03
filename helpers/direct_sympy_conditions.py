"""Exact truth facts for pure mathematical branch guards, compiler-only."""
import ast
from direct_sympy_signatures import StructuralSignatures


class BranchFacts:
    def __init__(self):self.signatures=StructuralSignatures()

    @staticmethod
    def negate(node):
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):return node.operand
        if isinstance(node,ast.Call) and node.func.id=="Not" and len(node.args)==1:return node.args[0]
        return None

    @staticmethod
    def logical(node):
        if isinstance(node,ast.BoolOp):return ("And" if isinstance(node.op,ast.And) else "Or",node.values)
        if isinstance(node,ast.Call) and node.func.id in ("And","Or"):return node.func.id,node.args
        return None

    def truth(self,node,facts):
        if isinstance(node,ast.Constant) and type(node.value) is bool:return node.value
        inner=self.negate(node)
        if inner is not None:
            value=self.truth(inner,facts);return None if value is None else not value
        known=dict(facts).get(self.signatures.key(node))
        if known is not None:return known
        logical=self.logical(node)
        if logical:
            kind,children=logical;values=[self.truth(child,facts) for child in children]
            if kind=="And":return False if False in values else True if all(x is True for x in values) else None
            return True if True in values else False if all(x is False for x in values) else None
        return None

    def assume(self,node,value,facts):
        known=self.truth(node,facts)
        if known is not None:return facts if known==value else None
        inner=self.negate(node)
        if inner is not None:return self.assume(inner,not value,facts)
        own=dict(facts);own[self.signatures.key(node)]=value
        result=tuple(sorted(own.items()))
        logical=self.logical(node)
        if logical and ((logical[0]=="And" and value) or (logical[0]=="Or" and not value)):
            for child in logical[1]:
                result=self.assume(child,value,result)
                if result is None:return None
        return result

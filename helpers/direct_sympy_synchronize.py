"""Synchronize identical ordered selectors in pure scalar operations.

The input must already be a CAS fixed point. Temporary unary lifting exposes
matching siblings; independent selectors are never multiplied by this pass.
All expressions are restored as literal strings, with no runtime sharing.
"""
import ast
import copy
from direct_sympy_strings import syntax
from direct_sympy_signatures import StructuralSignatures

PURE={'R16','R32','sqrt','Silu16','Bits64','Float64','U64And','U64Or','U64Shr','U64Add','U64Mul'}
GUARD_PURE=PURE|{'Piecewise','And','Or','Not','Eq','Ne','Lt','Le','Gt','Ge'}


def propose(expression,*,max_lifts=256,max_nodes=262144):
    if expression.count('Piecewise(')<2:return expression,0,0,'no-matching-siblings'
    signatures=StructuralSignatures();lifts=0;zips=0;costs={}
    def cost(node):
        known=costs.get(node)
        if known is None:
            known=1+sum(cost(child) for child in ast.iter_child_nodes(node));costs[node]=known
        return known
    def selector(node):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Piecewise'):return None
        if not node.args or any(not isinstance(pair,ast.Tuple) or len(pair.elts)!=2 for pair in node.args):return None
        if any(isinstance(child,ast.Call) and child.func.id not in GUARD_PURE for pair in node.args for child in ast.walk(pair.elts[1])):return None
        last=node.args[-1].elts[1]
        if not isinstance(last,ast.Constant) or last.value is not True:return None
        return tuple(signatures.key(pair.elts[1]) for pair in node.args)
    class Sync(ast.NodeTransformer):
        def visit_Call(self,node):
            if node.func.id=='Piecewise':
                for pair in node.args:pair.elts[0]=self.visit(pair.elts[0])
                costs.pop(node,None);return node
            node=self.generic_visit(node)
            return self.combine(node,node.args) if node.func.id in PURE else node
        def visit_BinOp(self,node):
            node=self.generic_visit(node);return self.combine(node,[node.left,node.right])
        def visit_UnaryOp(self,node):
            node=self.generic_visit(node)
            return self.combine(node,[node.operand]) if isinstance(node.op,(ast.UAdd,ast.USub)) else node
        def combine(self,node,children):
            nonlocal lifts,zips
            indexes=[i for i,child in enumerate(children) if selector(child) is not None]
            if not indexes:return node
            if any(isinstance(child,ast.Call) and child.func.id not in GUARD_PURE for value in children for child in ast.walk(value)):return node
            shape=selector(children[indexes[0]])
            if any(selector(children[i])!=shape for i in indexes):return node
            lifts+=1;zips+=len(indexes)>1
            if lifts>max_lifts:raise ValueError('synchronization-lift-budget')
            first=children[indexes[0]];arms=[]
            for arm,pair in enumerate(first.args):
                own=[child.args[arm].elts[0] if i in indexes else child for i,child in enumerate(children)]
                body=copy.copy(node)
                if isinstance(body,ast.Call):body.args=own
                elif isinstance(body,ast.BinOp):body.left,body.right=own
                else:body.operand=own[0]
                arms.append(ast.Tuple(elts=[body,pair.elts[1]],ctx=ast.Load()))
            result=ast.Call(func=ast.Name(id='Piecewise',ctx=ast.Load()),args=arms,keywords=[])
            if cost(result)>max_nodes:raise ValueError('synchronization-node-budget')
            return result
    try:
        result=Sync().visit(syntax(expression))
        if cost(result)>max_nodes:raise ValueError('synchronization-node-budget')
        return (ast.unparse(result),lifts,zips,'proposed') if zips else (expression,lifts,zips,'no-matching-siblings')
    except ValueError as error:
        if str(error) not in ('synchronization-lift-budget','synchronization-node-budget'):raise
        return expression,lifts,zips,str(error)

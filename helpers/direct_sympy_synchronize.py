"""Synchronize identical ordered selectors in pure scalar operations.

The input must already be a CAS fixed point. Temporary unary lifting exposes
matching siblings; independent selectors are never multiplied by this pass.
All expressions are restored as literal strings, with no runtime sharing.
"""
import ast
import copy
from direct_sympy_strings import syntax
from direct_sympy_signatures import StructuralSignatures

PURE={'R16','R32','sqrt','Silu16','Bits64','Float64','U64And','U64Or','U64Shr','U64Add','U64Mul','U64FromF64','F64FromU64'}
GUARD_PURE=PURE|{'Piecewise','And','Or','Not','Eq','Ne','Lt','Le','Gt','Ge'}


def propose(expression,*,max_lifts=256,max_nodes=262144,pure_functions=(),completed_views=None):
    # Only compiler-admitted, same-context views may expose an opaque producer.
    # Their leaves remain opaque: reopening every prior producer would recreate
    # the complete graph before finding a single matching decision.
    completed_views=completed_views or {}
    if expression.count('Piecewise(')<2 and not completed_views:return expression,0,0,'no-matching-siblings'
    signatures=StructuralSignatures();lifts=0;zips=0;costs={}
    guard_pure=GUARD_PURE|set(pure_functions)
    def cost(node):
        known=costs.get(node)
        if known is None:
            known=1+sum(cost(child) for child in ast.iter_child_nodes(node));costs[node]=known
        return known
    def selector(node):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Piecewise'):return None
        if not node.args or any(not isinstance(pair,ast.Tuple) or len(pair.elts)!=2 for pair in node.args):return None
        if any(not pure(pair.elts[1]) for pair in node.args):return None
        last=node.args[-1].elts[1]
        if not isinstance(last,ast.Constant) or last.value is not True:return None
        return tuple(signatures.key(pair.elts[1]) for pair in node.args)
    views={};purity={}
    def pure(node):
        # Synchronization introduces only whitelisted operations. A node's
        # purity is therefore invariant under its bottom-up rewrites.
        if node in purity:return purity[node]
        result=not (isinstance(node,ast.Call) and node.func.id not in guard_pure) and all(pure(child) for child in ast.iter_child_nodes(node))
        purity[node]=result
        return result
    def rebuild(node,children):
        body=copy.copy(node)
        if isinstance(body,ast.Call):body.args=children
        elif isinstance(body,ast.BinOp):body.left,body.right=children
        else:body.operand=children[0]
        return body
    def operands(node):
        if isinstance(node,ast.Call) and node.func.id in PURE:return node.args
        if isinstance(node,ast.BinOp):return [node.left,node.right]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return [node.operand]
        return None
    def view(node):
        # Expose a selector through pure operations without creating a
        # Piecewise at each unary ancestor. Arms are materialized only when
        # two siblings actually share the same ordered decision.
        if node in views:return views[node]
        if isinstance(node,ast.Call) and not node.args and node.func.id in completed_views:
            result=view(completed_views[node.func.id]) if pure(node) else None
            views[node]=result
            return result
        shape=selector(node)
        if shape is not None:
            result=(shape,[pair.elts for pair in node.args],0) if pure(node) else None
        else:
            children=operands(node);result=None
            if children is not None and pure(node):
                own=[view(child) for child in children]
                indexes=[i for i,value in enumerate(own) if value is not None]
                if indexes and all(own[i][0]==own[indexes[0]][0] for i in indexes):
                    first=own[indexes[0]]
                    arms=[]
                    for arm,pair in enumerate(first[1]):
                        selected=[own[i][1][arm][0] if i in indexes else child for i,child in enumerate(children)]
                        arms.append([rebuild(node,selected),pair[1]])
                    result=(first[0],arms,1+sum(own[i][2] for i in indexes))
        views[node]=result
        return result
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
            # Retain the original unary expression. Its selector can be
            # inspected by a parent without distributing it speculatively.
            return self.generic_visit(node)
        def combine(self,node,children):
            nonlocal lifts,zips
            own=[view(child) for child in children]
            indexes=[i for i,value in enumerate(own) if value is not None]
            if len(indexes)<2 or not pure(node):return node
            first=own[indexes[0]]
            if any(own[i][0]!=first[0] for i in indexes):return node
            lifts+=1+sum(own[i][2] for i in indexes);zips+=1
            if lifts>max_lifts:raise ValueError('synchronization-lift-budget')
            arms=[]
            for arm,pair in enumerate(first[1]):
                selected=[own[i][1][arm][0] if i in indexes else child for i,child in enumerate(children)]
                arms.append(ast.Tuple(elts=[rebuild(node,selected),pair[1]],ctx=ast.Load()))
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

"""Bounded diagnostic evaluation of fully substituted mathematical strings.

Only the verifier uses these compiler-certified literals. Files and final Rust
never contain this evaluator, its cache or its temporary calls. Lazy calls
preserve unreachable branch behavior; caches are owned by one input evaluation.
"""
import ast
from collections import OrderedDict
from copy import copy

from direct_sympy_strings import syntax
from direct_sympy_architecture_test import LazyBranches


class WorkingProgram:
    def __init__(self,text,compiler,functions):
        self.functions=functions
        # Registration keys already carry the complete domain. Keep only the
        # actual session's regions; callers must not mix compiler sessions.
        self.compiler=compiler
        self.programs={}
        self.parsed_characters=0
        self.root=self._compile(text)

    def _compile(self,text):
        previous=self.programs.get(text)
        if previous is not None:return previous
        view=copy(self.compiler)
        # Strictly smaller literals give an acyclic diagnostic program. The
        # whole result itself may have just been registered by composition.
        view._regions=OrderedDict((key,value) for key,value in self.compiler._regions.items() if len(key[1])<len(text))
        contexts={key[0] for key in view._regions}
        if len(contexts)>1:raise ValueError('Working verifier cannot mix producer contexts')
        context=next(iter(contexts),())
        compact,regions=view.compact_regions(text,context,validate_context=False)
        # No algebra is applied here. An exact, already admitted literal is
        # evaluated under the very same input on first reachable use.
        aliases={name:self._compile(literal) for name,literal in regions.items()}
        class Calls(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id in aliases:
                    return ast.copy_location(ast.Call(func=ast.Name(id=node.id,ctx=ast.Load()),args=[],keywords=[]),node)
                return node
        tree=LazyBranches().visit(Calls().visit(ast.Expression(syntax(compact))))
        code=compile(ast.fix_missing_locations(tree),'<working-expression-verifier>','eval')
        self.parsed_characters+=len(compact)
        result=(code,aliases,len(self.programs));self.programs[text]=result
        return result

    def __call__(self,values):
        cached={}
        def evaluate(program):
            code,aliases,key=program
            if key not in cached:
                thunks={name:(lambda producer=producer:evaluate(producer)) for name,producer in aliases.items()}
                cached[key]=eval(code,{'__builtins__':{},**self.functions,**thunks},values)
            return cached[key]
        return evaluate(self.root)

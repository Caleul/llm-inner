"""Exact compiler-only AST signatures without repeatedly printing subtrees.

Interned tuples compare structurally, not by a lossy digest. Float tokens use
all IEEE bits, including signed zero. Mutation clients invalidate every node
whose fields change; the emitted representation remains a mathematical string.
"""
import ast
import struct
from weakref import WeakKeyDictionary


class StructuralSignatures:
    def __init__(self):
        self.nodes=WeakKeyDictionary()
        self.interned={}

    def invalidate(self,node):
        self.nodes.pop(node,None)

    def key(self,node):
        cached=self.nodes.get(node)
        if cached is not None:return cached
        def token(value):
            if isinstance(value,ast.AST):return ("node",self.key(value))
            if isinstance(value,list):return ("list",tuple(token(x) for x in value))
            if type(value) is float:return ("float64",struct.pack(">d",value))
            return (type(value).__name__,value)
        structure=(type(node).__name__,tuple((name,token(value)) for name,value in ast.iter_fields(node)))
        key=self.interned.get(structure)
        if key is None:
            key=len(self.interned)+1
            self.interned[structure]=key
        self.nodes[node]=key
        return key

    def translated_keys(self,root,replacements):
        """Return original structural keys for exact compiler placeholders.

        Never overwrite ordinary AST keys: virtual and literal syntax remain
        distinct. Substitution compares interned structure, not hashes.
        """
        translated={}
        def visit(node):
            actual=self.key(node)
            if node in translated:return translated[node]
            if actual in replacements:
                translated[node]=replacements[actual];return translated[node]
            def token(value):
                if isinstance(value,ast.AST):return ('node',visit(value))
                if isinstance(value,list):return ('list',tuple(token(x) for x in value))
                if type(value) is float:return ('float64',struct.pack('>d',value))
                return (type(value).__name__,value)
            structure=(type(node).__name__,tuple((name,token(value)) for name,value in ast.iter_fields(node)))
            key=self.interned.get(structure)
            if key is None:
                key=len(self.interned)+1;self.interned[structure]=key
            translated[node]=key;return key
        visit(root)
        return translated

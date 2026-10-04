"""Optional equivalent stdlib-regex scans; numerical compiler sources stay unchanged.

Necessary-literal negative checks, group-free identifier findall and large
region searches are accelerated. Region candidates undergo full literal
verification; regex matches, Unicode boundaries and flags use stdlib semantics. This changes neither expressions nor numeric proofs.
"""
from contextlib import contextmanager
import importlib
import ast
import inspect
import textwrap
import re
from pathlib import Path


BACKEND_VERSION='equivalent-stdlib-v2'


NECESSARY={
    r'\bCASStableRegion[0-9]+\b':('CASStableRegion',),
    r'\bCASNumericRegion[0-9]+\b':('CASNumericRegion',),
    r'\bPiecewise\s*\(':('Piecewise',),
    r'\bR(?:16|32)\s*\(':('R16','R32'),
    r'\b(?:R16|R32|sqrt|Silu16)\s*\(':('R16','R32','sqrt','Silu16'),
    r'\bCAS(?:Boundary|StableRegion|NumericRegion)[0-9]+\b':('CASBoundary','CASStableRegion','CASNumericRegion'),
}
IDENTIFIER=re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')


class StreamingTextPath(type(Path())):
    """Exact ASCII diagnostic writes without a full second encoded copy."""
    def write_text(self,data,encoding=None,errors=None,newline=None):
        if type(data) is not str or not data.isascii() or len(data)<1024*1024:
            return super().write_text(data,encoding=encoding,errors=errors,newline=newline)
        with self.open(mode='w',encoding=encoding,errors=errors,newline=newline) as stream:
            for start in range(0,len(data),1024*1024):stream.write(data[start:start+1024*1024])
        return len(data)


class EquivalentScans:
    def __init__(self):
        self.negative_searches=0;self.identifier_scans=0;self.characters=0
        self.literal_region_searches=0;self.literal_region_candidates=0

    def __getattr__(self,name):return getattr(re,name)

    def search(self,pattern,text,flags=0):
        if type(flags) in (int,bool,re.RegexFlag) and flags==0 and type(pattern) is str and type(text) is str:
            needles=NECESSARY.get(pattern)
            if needles is not None and not any(word in text for word in needles):
                self.negative_searches+=1;self.characters+=len(text)
                return None
        return re.search(pattern,text,flags)

    def findall(self,pattern,text,flags=0):
        literal=None
        if type(flags) in (int,bool,re.RegexFlag) and flags==0 and type(pattern) is str and type(text) is str and pattern.startswith(r'\b'):
            if pattern.endswith(r'\b'):
                name=pattern[2:-2]
                if IDENTIFIER.fullmatch(name):literal=name
            elif pattern.endswith(r'\(\)'):
                name=pattern[2:-4]
                if IDENTIFIER.fullmatch(name):literal=name+'()'
        if literal is None:return re.findall(pattern,text,flags)
        self.identifier_scans+=1;self.characters+=len(text)
        compiled=re.compile(pattern);result=[];position=text.find(literal)
        while position>=0:
            # The real matcher decides boundaries, including Unicode. Only
            # group zero is observable through these group-free patterns.
            match=compiled.match(text,position)
            if match is not None:result.append(match.group(0))
            position=text.find(literal,position+len(literal))
        return result

    def literal_find(self,text,region,start=0):
        """Exact first occurrence, without preprocessing a giant needle.

        A full occurrence must contain the same suffix at its fixed offset.
        Search that bounded anchor first, then verify the entire literal.
        Candidate offsets increase in the same order as str.find.
        """
        if type(text) is not str or type(region) is not str or type(start) is not int or start<0 or start>len(text) or len(region)<1048576:
            return text.find(region,start)
        self.literal_region_searches+=1
        size=len(region);width=4096
        if size>len(text)-start:return -1
        suffix=region[-width:];prefix=region[:width];middle_offset=size//2;middle=region[middle_offset:middle_offset+width]
        offset=size-width;position=text.find(suffix,start+offset)
        while position>=0:
            candidate=position-offset;self.literal_region_candidates+=1
            if text.startswith(prefix,candidate) and text.startswith(middle,candidate+middle_offset) and text.startswith(region,candidate):return candidate
            position=text.find(suffix,position+1)
        return -1

    def summary(self):return {'negativeSearches':self.negative_searches,'identifierScans':self.identifier_scans,'characters':self.characters,'literalRegionSearches':self.literal_region_searches,'literalRegionCandidates':self.literal_region_candidates}


def region_search_method(original,backend):
    """Replace only the two literal searches; preserve every grammar,
    domain/context, boundary and restoration check in the original method.
    Numerical compiler files and savepoint identity remain unchanged.
    """
    source=getattr(original,'_literal_search_original',original)
    tree=ast.parse(textwrap.dedent(inspect.getsource(source)));count=0
    class Searches(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and node.func.attr=='find'
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='current'
                    and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='text'):
                count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_equivalent_region_find',ctx=ast.Load()),args=[node.func.value,*node.args],keywords=node.keywords),node)
            return node
    tree=Searches().visit(tree)
    if count!=2:raise ValueError('Unsupported compact_regions implementation; expected two literal searches')
    namespace=dict(source.__globals__);namespace['_equivalent_region_find']=backend.literal_find
    exec(compile(ast.fix_missing_locations(tree),'<equivalent compact_regions>','exec'),namespace)
    result=namespace[source.__name__];result._literal_search_original=source
    return result


@contextmanager
def install():
    backend=EquivalentScans();saved=[]
    try:
        for name in ('direct_sympy_strings','direct_sympy_conversions','direct_sympy_checkpoint','direct_sympy_savepoints'):
            module=importlib.import_module(name);saved.append((module,'re',module.re));module.re=backend
            if name=='direct_sympy_strings':
                cls=module.StringCompiler;saved.append((cls,'compact_regions',cls.compact_regions));cls.compact_regions=region_search_method(cls.compact_regions,backend)
            if name=='direct_sympy_checkpoint':
                saved.append((module,'Path',module.Path));module.Path=StreamingTextPath
        yield backend
    finally:
        for module,attribute,original in reversed(saved):setattr(module,attribute,original)

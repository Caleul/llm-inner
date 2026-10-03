"""Optional equivalent stdlib-regex scans; numerical compiler sources stay unchanged.

Only necessary-literal negative checks and group-free identifier findall are
accelerated. Positive matches, Unicode boundaries and flags still use the
stdlib engine. This changes neither expressions nor numeric proofs.
"""
from contextlib import contextmanager
import importlib
import re
from pathlib import Path


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

    def summary(self):return {'negativeSearches':self.negative_searches,'identifierScans':self.identifier_scans,'characters':self.characters}


@contextmanager
def install():
    backend=EquivalentScans();saved=[]
    try:
        for name in ('direct_sympy_strings','direct_sympy_conversions','direct_sympy_checkpoint','direct_sympy_savepoints'):
            module=importlib.import_module(name);saved.append((module,'re',module.re));module.re=backend
            if name=='direct_sympy_checkpoint':
                saved.append((module,'Path',module.Path));module.Path=StreamingTextPath
        yield backend
    finally:
        for module,attribute,original in reversed(saved):setattr(module,attribute,original)

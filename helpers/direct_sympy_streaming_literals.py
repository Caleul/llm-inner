"""Compiler-only literal sharing; final emission restores every dependency.

The registry contains mathematical strings, not a runtime representation.
Its temporary calls may be used only by the compiler and verifier. The
atomic final writer substitutes all of them, with parentheses, in a stream.
"""
import ast
from contextlib import contextmanager
import gzip
import hashlib
import os
from pathlib import Path
import re
import tempfile

from direct_sympy_strings import syntax
from direct_sympy_synchronize import GUARD_PURE


ALIASES=re.compile(r'\bCompileValue([0-9]+)\s*\(\s*\)')
ELEMENTARY=GUARD_PURE-{'R16','R32','sqrt','Silu16'}


class StreamingLiterals:
    def __init__(self,model):
        if model.memo or model.on_completed is not None or model.conversions is None:
            raise ValueError('Streaming literals require a fresh numerical compiler without a savepoint callback')
        self.model=model;self.definitions=[];self.by_text={};self.names={};self.events=[]
        self.definition_proofs=[];self.definition_recipes={}
        self.context=model.compiler.context(model.domains)

    def audit(self,text):
        if self.model.compiler.context(self.model.domains)!=self.context:
            raise ValueError('Literal input context changed')
        node=syntax(text);functions={child.func for child in ast.walk(node) if isinstance(child,ast.Call)}
        for child in ast.walk(node):
            if isinstance(child,ast.Name) and child not in functions and child.id not in self.model.domains:
                raise ValueError('Unsubstituted runtime variable: '+child.id)
            if not isinstance(child,ast.Call):continue
            if child.func.id in ELEMENTARY:continue
            match=re.fullmatch('CompileValue([0-9]+)',child.func.id)
            if match is None or child.args or int(match[1])>=len(self.definitions) or child.func.id!='CompileValue'+str(int(match[1])):
                raise ValueError('Unknown or unfinished literal call: '+child.func.id)
        return node

    def intern(self,text):
        node=self.audit(text)
        if ALIASES.fullmatch(text):return text
        previous=self.by_text.get(text)
        if previous is not None:return previous
        if not isinstance(node,ast.Call):
            # Constants and fundamental exact expressions need no alias.
            return text
        session=self.model.conversions
        query,_=session.analyze_expression(text)
        if session.key(query) not in session.converted_regions:
            raise ValueError('Unclosed numerical literal cannot be shared')
        bounds,kind,positive_zero=session.bounds(query),session.value_kind(query),session.no_negative_zero(query)
        name='CompileValue'+str(len(self.definitions));alias=name+'()'
        self.definitions.append(text);self.by_text[text]=alias
        self.definition_proofs.append((bounds,kind,positive_zero,session.frontier_bounds.get(session.closed_literal_keys.get(text,session.key(node)))))
        marker=syntax(alias);key=session.key(marker)
        if bounds is not None:session.completed[key]=bounds
        if kind=='half':session.half_values.add(key)
        if kind in ('half','f32'):session.f32_values.add(key)
        if positive_zero:session.no_negative_zero_values.add(key)
        session.converted_regions.add(key)
        session.closed_literals[alias]=(bounds,kind,positive_zero)
        session.closed_literal_characters+=len(alias)
        session.closed_literal_keys[alias]=key;session.closed_literal_pure[alias]=True
        old=session.closed_literal_keys.get(text,session.key(node))
        if old in session.sign_projections:session.sign_projections[key]=session.sign_projections[old]
        if old in session.frontier_bounds:session.frontier_bounds[key]=session.frontier_bounds[old]
        selector=session.selector_literals.get(text)
        if selector is not None:
            session.selector_literals[alias]=selector;session.selector_view_characters+=selector[2]
        self.model.compiler.register_completed_region(alias,self.model.domains,marker,word_closed=True)
        if node.func.id=='Float64' and len(node.args)==1:
            from direct_sympy_words import word_width
            width=word_width(node.args[0])
            if width is not None:
                region=(self.context,alias)
                self.model.compiler._region_roots[region]=('Float64',width)
                self.model.compiler._region_word_payloads[region]=ast.unparse(node.args[0])
        sign=session.closed_sign_literals.get(text)
        if sign is not None and sign!=text:session.closed_sign_literals[alias]=self.intern(sign)
        return alias

    def size(self,text):
        sizes=[]
        def own(value):
            return len(value)+sum(sizes[int(m[1])]+2-len(m[0]) for m in ALIASES.finditer(value))
        for definition in self.definitions:sizes.append(own(definition))
        return own(text)

    def chunks(self,text):
        """Iterative substitution; no Python recursion or expanded string."""
        self.audit(text)
        stack=[iter((text,))]
        while stack:
            try:part=next(stack[-1])
            except StopIteration:
                stack.pop();continue
            if type(part) is int:
                stack.append(iter(('(',self.definitions[part],')')));continue
            matches=list(ALIASES.finditer(part))
            if not matches:
                for start in range(0,len(part),1024*1024):yield part[start:start+1024*1024]
                continue
            pieces=[];start=0
            for match in matches:
                pieces.extend((part[start:match.start()],int(match[1])));start=match.end()
            pieces.append(part[start:]);stack.append(iter(pieces))

    def write(self,path,text,*,max_characters,compressed=True):
        expected=self.size(text)
        if expected>max_characters:raise ValueError(f'Expanded artifact exceeds character budget: {expected} > {max_characters}')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256();written=0
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            try:
                encoded=gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=1,mtime=0,filename='') if compressed else stream
                try:
                    for chunk in self.chunks(text):
                        if 'CompileValue' in chunk:raise ValueError('Compiler literal escaped expansion')
                        block=chunk.encode();written+=len(chunk);digest.update(block);encoded.write(block)
                finally:
                    if compressed:encoded.close()
                if written!=expected:raise ValueError('Incomplete final literal expansion')
                stream.flush();os.fsync(stream.fileno());os.replace(temporary,path)
            finally:temporary.unlink(missing_ok=True)
        return {'path':str(path),'characters':written,'sha256':digest.hexdigest(),'compressedBytes':path.stat().st_size,'compilerAliases':0}



@contextmanager
def streaming_literals(model):
    registry=StreamingLiterals(model);original=model.producer
    original_measure=model.compiler.expression_size
    model.compiler.expression_size=registry.size
    def producer(key,build):
        if key in model.memo:return model.memo[key]
        recipe=[]
        def capture():
            expression=build();recipe.append(expression);return expression
        text=original(key,capture);alias=registry.intern(text);model.memo[key]=alias
        match=ALIASES.fullmatch(alias)
        if match is not None and recipe:
            index=int(match[1])
            dependencies=[int(m[1]) for m in ALIASES.finditer(recipe[0])]
            if all(dependency<index for dependency in dependencies):
                registry.definition_recipes.setdefault(index,recipe[0])
        registry.names[key]=alias
        registry.events.append((key,len(text),registry.size(alias),len(registry.definitions)))
        return alias
    model.producer=producer
    try:yield registry
    finally:
        model.producer=original
        model.compiler.expression_size=original_measure

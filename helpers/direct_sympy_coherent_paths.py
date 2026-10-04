"""Distribute stabilized decisions with one shared fact per pure predicate.

Definitions exist only in the compiler. Every emitted arm and guard expands
to the original inputs. A guard owns its prefix context, never the narrower
context of its eventual leaf; guards are evaluated in their original order.
"""
import ast
import copy
from dataclasses import dataclass
import gzip
import hashlib
import os
from pathlib import Path
import tempfile

from direct_sympy_streaming_literals import ALIASES,StreamingLiterals
from direct_sympy_strings import syntax,refine


class LiteralView:
    audit=StreamingLiterals.audit
    size=StreamingLiterals.size
    chunks=StreamingLiterals.chunks
    write=StreamingLiterals.write

    def __init__(self,registry,definitions):
        self.model=registry.model;self.context=registry.context
        self.definitions=tuple(definitions)
        for index,text in enumerate(self.definitions):
            if any(int(m[1])>=index for m in ALIASES.finditer(text)):
                raise ValueError('Forward or cyclic literal dependency')
            if index>=len(registry.definitions) or text!=registry.definitions[index]:self.audit(text)


@dataclass(frozen=True)
class Guard:
    view:LiteralView
    expression:str
    truth:bool


@dataclass(frozen=True)
class PathArm:
    view:LiteralView
    expression:str
    guards:tuple[Guard,...]

    def size(self):
        return self.view.size(self.expression)+sum(g.view.size(g.expression) for g in self.guards)


class CoherentPaths:
    def __init__(self,registry,*,max_paths=64,max_search_nodes=2000000):
        if max_paths<1 or max_search_nodes<1:raise ValueError('Positive path/search budgets required')
        self.registry=registry;self.compiler=registry.model.compiler
        self.facts=self.compiler.branch_facts
        self.originals=tuple(registry.definitions)
        self.trees=tuple(registry.audit(text) for text in self.originals)
        LiteralView(registry,self.originals)
        self.max_paths=max_paths;self.max_search_nodes=max_search_nodes
        self.predicates={}
        self.stats={'splitContexts':0,'completedPaths':0,'contradictions':0,'selectorSelections':0,'CASPasses':0}

    @staticmethod
    def alias(node):
        if isinstance(node,ast.Call) and not node.args:
            match=ALIASES.fullmatch(ast.unparse(node))
            if match:return int(match[1])
        return None

    def remember(self,node):
        self.predicates[self.facts.signatures.key(node)]=node
        negated=self.facts.negate(node)
        if negated is not None:self.remember(negated)
        logical=self.facts.logical(node)
        if logical:
            for child in logical[1]:self.remember(child)

    def assume(self,node,truth,facts):
        self.remember(node)
        result=self.facts.assume(node,truth,facts)
        while result is not None:
            before=result
            for key,predicate in tuple(self.predicates.items()):
                expected=dict(result).get(key);logical=self.facts.logical(predicate)
                if expected is None or logical is None:continue
                kind,children=logical
                values=[self.facts.truth(child,result) for child in children]
                actual=False if kind=='And' and False in values else True if kind=='Or' and True in values else (all(values) if kind=='And' else any(values)) if None not in values else None
                if actual is not None and actual!=expected:return None
                unknown=[child for child,value in zip(children,values) if value is None]
                if len(unknown)==1 and ((kind=='And' and not expected and all(value is True for value in values if value is not None)) or (kind=='Or' and expected and all(value is False for value in values if value is not None))):
                    result=self.facts.assume(unknown[0],expected,result)
                    if result is None:return None
            if result==before:return result
        return None

    def domains(self,facts):
        result=dict(self.registry.model.domains)
        for node in self.predicates.values():
            truth=self.facts.truth(node,facts)
            if truth is not None:
                result=refine(result,node,truth)
                if result is None:return None
        return result

    def next_decision(self,root,facts):
        memo={};visited=0
        def find(node):
            nonlocal visited
            visited+=1
            if visited>self.max_search_nodes:raise ValueError('Decision search budget exceeded; no complete result')
            alias=self.alias(node)
            if alias is not None:
                if alias not in memo:memo[alias]=find(self.trees[alias])
                return memo[alias]
            logical=self.facts.logical(node)
            if logical:
                kind,children=logical
                for index,child in enumerate(children):
                    nested=find(child)
                    if nested is not None:return nested
                    truth=self.facts.truth(child,facts)
                    if truth is (False if kind=='And' else True):return None
                    if truth is None and index+1<len(children):return child
                return None
            if isinstance(node,ast.Call) and node.func.id=='Piecewise':
                for pair in node.args:
                    body,condition=pair.elts
                    truth=self.facts.truth(condition,facts)
                    if truth is False:continue
                    if truth is None:
                        nested=find(condition)
                        return condition if nested is None else nested
                    return find(body)
                raise ValueError('No reachable selector arm')
            for child in ast.iter_child_nodes(node):
                nested=find(child)
                if nested is not None:return nested
            return None
        return find(root)

    def literal(self,root,facts,domains):
        texts=list(self.originals);memo=set();before=len(self.compiler.events)
        # Candidate costs must expand the selected definitions of this
        # path, rather than the original definitions of other branches.
        cost_view=object.__new__(LiteralView)
        cost_view.model=self.registry.model;cost_view.context=self.registry.context
        cost_view.definitions=texts
        def select(node):
            alias=self.alias(node)
            if alias is not None:
                if alias not in memo:
                    selected=select(self.trees[alias])
                    texts[alias]=self.compiler.stabilize('('+ast.unparse(selected)+')',domains,facts=facts)
                    memo.add(alias)
                return node
            if isinstance(node,ast.Call) and node.func.id=='Piecewise':
                for pair in node.args:
                    body,condition=pair.elts
                    truth=self.facts.truth(condition,facts)
                    if truth is None:raise ValueError('Undecided selector in a supposedly complete path')
                    if truth:
                        self.stats['selectorSelections']+=1
                        return select(body)
                raise ValueError('No reachable selector arm')
            logical=self.facts.logical(node)
            if logical:
                kind,children=logical;selected=[]
                for child in children:
                    truth=self.facts.truth(child,facts)
                    if truth is None:selected.append(select(child))
                    elif truth is (False if kind=='And' else True):
                        selected.append(ast.Constant(value=truth));break
                    # Known neutral operands already hold in this prefix.
                if not selected:return ast.Constant(value=kind=='And')
                return ast.Call(func=ast.Name(id=kind,ctx=ast.Load()),args=selected,keywords=[])
            result=copy.copy(node)
            for name,value in ast.iter_fields(node):
                if isinstance(value,ast.AST):setattr(result,name,select(value))
                elif isinstance(value,list):setattr(result,name,[select(child) if isinstance(child,ast.AST) else child for child in value])
            return result
        original_cost=self.compiler.expression_size
        self.compiler.expression_size=cost_view.size
        try:
            expression=self.compiler.stabilize('('+ast.unparse(select(root))+')',domains,facts=facts)
        finally:self.compiler.expression_size=original_cost
        self.stats['CASPasses']+=len(self.compiler.events)-before
        return LiteralView(self.registry,texts),expression

    def arms(self,expression):
        root=self.registry.audit(expression);pending=[((),())]
        while pending:
            facts,guards=pending.pop();domains=self.domains(facts)
            if domains is None:self.stats['contradictions']+=1;continue
            decision=self.next_decision(root,facts)
            if decision is None:
                if self.stats['completedPaths']>=self.max_paths:raise ValueError('Path budget exceeded; no complete result')
                view,body=self.literal(root,facts,domains)
                self.stats['completedPaths']+=1
                yield PathArm(view,body,guards);continue
            # Freeze this guard BEFORE assuming its own truth or a later
            # predicate. Later arm-local identities must not alter dispatch.
            view,condition=self.literal(decision,facts,domains)
            self.stats['splitContexts']+=1
            for truth in (False,True):
                own=self.assume(decision,truth,facts)
                if own is None:self.stats['contradictions']+=1;continue
                pending.append((own,guards+(Guard(view,condition,truth),)))

    def write(self,path,expression,*,max_characters,compressed=False):
        if max_characters<1:raise ValueError('Positive artifact budget required')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        written=0;digest=hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            try:
                encoded=gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=1,mtime=0,filename='') if compressed else stream
                def emit(text):
                    nonlocal written
                    if 'CompileValue' in text:raise ValueError('Compiler alias in final path')
                    written+=len(text)
                    if written>max_characters:raise ValueError('Flat artifact budget exceeded; no complete result')
                    block=text.encode();digest.update(block);encoded.write(block)
                try:
                    emit('Piecewise(');count=0
                    for arm in self.arms(expression):
                        body_size=arm.view.size(arm.expression)
                        guard_sizes=[guard.view.size(guard.expression) for guard in arm.guards]
                        self.stats['lastArm']={'bodyCharacters':body_size,
                            'guardCharacters':sum(guard_sizes),'decisions':len(arm.guards)}
                        reachable=set();pending=[arm.expression]
                        while pending:
                            for match in ALIASES.finditer(pending.pop()):
                                index=int(match[1])
                                if index not in reachable:
                                    reachable.add(index);pending.append(arm.view.definitions[index])
                        self.stats['lastArm']['dependencyGrowth']=[{
                            'definition':index,'storedCharacters':len(arm.view.definitions[index]),
                            'expandedCharacters':arm.view.size(f'CompileValue{index}()')}
                            for index in sorted(reachable)]
                        # Refuse before expanding a large arm into bytes. No
                        # bounded prefix may replace an existing artifact.
                        if written+body_size+sum(guard_sizes)>max_characters:
                            raise ValueError('Flat artifact budget exceeded; no complete result')
                        if count:emit(', ')
                        emit('((')
                        for text in arm.view.chunks(arm.expression):emit(text)
                        emit('), ');emit('And(' if arm.guards else 'True')
                        for index,guard in enumerate(arm.guards):
                            if index:emit(', ')
                            if not guard.truth:emit('Not(')
                            for text in guard.view.chunks(guard.expression):emit(text)
                            if not guard.truth:emit(')')
                        if arm.guards:emit(')')
                        emit(')');count+=1
                    emit(')')
                finally:
                    if compressed:encoded.close()
                # Every arm and prefix was already stabilized. Stabilize
                # the final combination too, within the compiler's explicit
                # resident-string budget; an oversized result is not admitted.
                if written>self.compiler.max_characters:raise ValueError('Final combination exceeds CAS budget; no complete result')
                stream.flush()
                if compressed:
                    with gzip.open(temporary,'rt',encoding='utf-8') as reader:combined=reader.read()
                else:combined=temporary.read_text()
                before=len(self.compiler.events)
                final=self.compiler.stabilize(combined,self.registry.model.domains)
                self.stats['CASPasses']+=len(self.compiler.events)-before
                if 'CompileValue' in final:raise ValueError('Compiler alias in combined artifact')
                written=len(final)
                if written>max_characters:raise ValueError('Stabilized artifact exceeds budget; no complete result')
                digest=hashlib.sha256(final.encode())
                stream.seek(0);stream.truncate()
                if compressed:
                    with gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=1,mtime=0,filename='') as writer:
                        for offset in range(0,len(final),1024*1024):writer.write(final[offset:offset+1024*1024].encode())
                else:
                    for offset in range(0,len(final),1024*1024):stream.write(final[offset:offset+1024*1024].encode())
                stream.flush();os.fsync(stream.fileno());os.replace(temporary,path)
            finally:temporary.unlink(missing_ok=True)
        return {'path':str(path),'characters':written,'sha256':digest.hexdigest(),'paths':count,'compilerAliases':0,'complete':True}

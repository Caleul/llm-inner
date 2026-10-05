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
import re
import tempfile

from direct_sympy_streaming_literals import ALIASES,StreamingLiterals
from direct_sympy_strings import Domain,StringCompiler,syntax,refine


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


class UnreachableNumericPath(ValueError):
    """Certified numerical bounds exclude this branch context."""


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


def same_literal(left,right):
    """Compare actual expanded bytes without allocating a complete string."""
    a=iter(left.view.chunks(left.expression));b=iter(right.view.chunks(right.expression))
    x=y='';i=j=0
    while True:
        if i==len(x):x=next(a,'');i=0
        if j==len(y):y=next(b,'');j=0
        if not x or not y:return not x and not y
        size=min(len(x)-i,len(y)-j)
        if x[i:i+size]!=y[j:j+size]:return False
        i+=size;j+=size


class OrderedGuardPruner:
    """Drop only predicates implied by the original ordered dispatch prefix.

    SymPy propositions represent byte-identical closed predicates, not their
    numerical internals. SAT proves implications for every possible truth
    assignment. Runtime strings retain their original lazy order. Hashes only
    select candidates; actual expanded bytes establish predicate identity.
    """
    def __init__(self,stats,*,max_atoms=128,max_nodes=16384):
        import sympy as sp
        self.sp=sp;self.remaining=sp.true;self.predicates={};self.atoms=0
        self.max_atoms=max_atoms;self.max_nodes=max_nodes;self.stats=stats

    def atom(self,guard):
        digest=hashlib.sha256()
        for chunk in guard.view.chunks(guard.expression):digest.update(chunk.encode())
        bucket=self.predicates.setdefault(digest.digest(),[])
        for prior,atom in bucket:
            if same_literal(prior,guard):return atom
        if self.atoms>=self.max_atoms:return None
        atom=self.sp.Symbol('GuardProof'+str(self.atoms));self.atoms+=1
        bucket.append((guard,atom));return atom

    def prune(self,arm):
        sp=self.sp;original=[];retained=[];prefix=self.remaining;eliminated=0
        if prefix is None:return arm
        for guard in arm.guards:
            atom=self.atom(guard)
            if atom is None:
                self.remaining=None;self.stats['guardProofBudgetStops']+=1
                return arm
            value=atom if guard.truth else sp.Not(atom);original.append(value)
            query=sp.And(prefix,sp.Not(value))
            if sp.count_ops(query)>self.max_nodes:
                self.remaining=None;self.stats['guardProofBudgetStops']+=1
                return arm
            self.stats['guardProofPasses']+=1
            if sp.satisfiable(query) is False:
                # This is an auxiliary entailment proof. Numerical strings
                # already reached their factor/simplify fixed points in
                # literal(); the final emitted combination passes them again.
                eliminated+=1
            else:retained.append(guard)
            prefix=sp.And(prefix,value)
        self.stats['orderedGuardEliminations']+=eliminated
        # Conditions of previous ORIGINAL paths are false when this arm is
        # reached. Removing implied guards preserves that fact inductively.
        self.remaining=sp.And(self.remaining,sp.Not(sp.And(*original)))
        return PathArm(arm.view,arm.expression,tuple(retained))


class CoherentPaths:
    def __init__(self,registry,*,max_paths=64,max_search_nodes=2000000):
        if max_paths<1 or max_search_nodes<1:raise ValueError('Positive path/search budgets required')
        self.registry=registry;self.compiler=registry.model.compiler
        self.facts=self.compiler.branch_facts
        self.originals=tuple(registry.definitions)
        self.trees=tuple(registry.audit(text) for text in self.originals)
        LiteralView(registry,self.originals)
        self.max_paths=max_paths;self.max_search_nodes=max_search_nodes
        self.predicates={};self.rms_bounds_cache={};self.guard_bounds_cache={};self.guard_implications=set()
        self.stats={'splitContexts':0,'completedPaths':0,'contradictions':0,'selectorSelections':0,'CASPasses':0,'numericArithmeticEliminated':0,'numericRecipeAttempts':0,'numericRecipeAdmissions':0,'numericRecipeBudgetStops':0,'coupledProjectionContradictions':0,'rmsBranchRefinements':0,'rmsConstantCells':0,'orderedGuardEliminations':0,'guardProofPasses':0,'guardProofBudgetStops':0}
        self.stats['numericGuardImplications']=0

    def condition_truth(self,node,facts):
        """Prove a comparison from prior path bounds before distributing it.

        No predicate assumes its own truth. Numerical enclosures are scoped
        to the frozen prefix, and unsupported/overlapping comparisons stay
        undecided. Logical operands keep their existing lazy traversal.
        """
        known=self.facts.truth(node,facts)
        if known is not None:return known
        if not isinstance(node,ast.Compare) or len(node.ops)!=1 or len(node.comparators)!=1:return None
        from direct_sympy_conversions import ConversionSession,FiniteSource
        key=(facts,len(self.predicates))
        if key not in self.guard_bounds_cache:
            domains=self.domains(facts)
            if domains is None:return None
            refined=self.rms_bounds(facts)
            for name,domain in list(domains.items()):
                local=refined.get(name)
                if local is not None:
                    low,high=max(domain.minimum,local.minimum),min(domain.maximum,local.maximum)
                    if low>high:return None
                    domains[name]=Domain(low,high,domain.quantum,domain.excludes_negative_zero)
            numeric=ConversionSession(self.compiler,domains)
            source=getattr(self.registry.model,'conversions',None)
            if source is not None:
                for name in domains:
                    own=numeric.key(syntax(name));old=source.key(syntax(name))
                    if old in source.half_values:numeric.half_values.add(own)
                    if old in source.f32_values:numeric.f32_values.add(own)
            for alias,proof in enumerate(getattr(self.registry,'definition_proofs',())):
                bound,kind,_,_=proof
                if bound is None:continue
                local=refined.get(alias)
                if local is not None:
                    low,high=max(bound.minimum,local.minimum),min(bound.maximum,local.maximum)
                    if low>high:continue
                    bound=FiniteSource(low,high,min(bound.quantum,local.quantum) if bound.quantum is not None and local.quantum is not None else None,max(bound.minimum_magnitude,local.minimum_magnitude))
                own=numeric.key(syntax(f'CompileValue{alias}()'));numeric.completed[own]=bound
                if kind=='half':numeric.half_values.add(own)
                if kind in ('half','f32'):numeric.f32_values.add(own)
            self.guard_bounds_cache[key]=numeric
        numeric=self.guard_bounds_cache[key];left=node.left;right=node.comparators[0];op=node.ops[0]
        if (isinstance(left,ast.Call) and left.func.id=='U64And' and len(left.args)==2
            and isinstance(left.args[1],ast.Constant) and left.args[1].value==0x7fffffffffffffff
            and isinstance(left.args[0],ast.Call) and left.args[0].func.id=='Bits64' and len(left.args[0].args)==1
            and isinstance(right,ast.Constant) and type(right.value)is int and 0<=right.value<0x7ff0000000000000):
            import struct
            a=numeric.bounds(left.args[0].args[0])
            if a is None:return None
            low=max(a.minimum_magnitude,a.minimum if a.minimum>0 else -a.maximum if a.maximum<0 else 0)
            high=max(abs(a.minimum),abs(a.maximum))
            threshold=struct.unpack('d',struct.pack('Q',right.value))[0]
            a=FiniteSource(low,high);b=FiniteSource(threshold,threshold)
        else:a,b=numeric.bounds(left),numeric.bounds(right)
        if a is None or b is None:return None
        truth=None
        if isinstance(op,ast.Lt):truth=True if a.maximum<b.minimum else False if a.minimum>=b.maximum else None
        elif isinstance(op,ast.LtE):truth=True if a.maximum<=b.minimum else False if a.minimum>b.maximum else None
        elif isinstance(op,ast.Gt):truth=True if a.minimum>b.maximum else False if a.maximum<=b.minimum else None
        elif isinstance(op,ast.GtE):truth=True if a.minimum>=b.maximum else False if a.maximum<b.minimum else None
        # Numeric equality cannot establish payload equality (+0/-0), and
        # overlapping intervals cannot establish a new dispatch decision.
        if truth is not None:
            proof=(facts,self.facts.signatures.key(node))
            if proof not in self.guard_implications:
                self.guard_implications.add(proof);self.stats['numericGuardImplications']+=1
        return truth

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
        memo={};visited=0;bounds=self.rms_bounds(facts)
        def find(node):
            nonlocal visited
            visited+=1
            if visited>self.max_search_nodes:raise ValueError('Decision search budget exceeded; no complete result')
            alias=self.alias(node)
            if alias is not None:
                own=bounds.get(alias)
                if own is not None and own.minimum==own.maximum and own.minimum!=0:return None
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
                    truth=self.condition_truth(condition,facts)
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

    def rms_bounds(self,facts):
        from direct_sympy_projection_constraints import rms_branch_bounds,rms_source_branch_bounds,projection_branch_bounds
        key=(facts,len(self.predicates))
        if key not in self.rms_bounds_cache:
            assumptions=[(p,truth) for p in self.predicates.values() if (truth:=self.facts.truth(p,facts)) is not None]
            refined=rms_branch_bounds(self.registry,assumptions)
            for name,bound in rms_source_branch_bounds(self.registry,assumptions).items():
                previous=refined.get(name)
                if previous is None:refined[name]=bound
                else:
                    from direct_sympy_conversions import FiniteSource
                    low,high=max(previous.minimum,bound.minimum),min(previous.maximum,bound.maximum)
                    if low<=high:refined[name]=FiniteSource(low,high,min(previous.quantum,bound.quantum),max(previous.minimum_magnitude,bound.minimum_magnitude))
            self.rms_bounds_cache[key]=projection_branch_bounds(self.registry,refined)
        return self.rms_bounds_cache[key]

    def literal(self,root,facts,domains):
        texts=list(self.originals);memo=set();recipe_bounds={};before=len(self.compiler.events);rms_bounds=self.rms_bounds(facts)
        # Tighten only this selected leaf. Prefix guards retain the context in
        # which they were frozen, and sibling/global input domains stay intact.
        domains=dict(domains)
        for name in domains:
            own=rms_bounds.get(name)
            if own is None:continue
            previous=domains[name];low=max(previous.minimum,own.minimum);high=min(previous.maximum,own.maximum)
            if low>high:raise UnreachableNumericPath('RMS source guard contradicts its input domain')
            domains[name]=Domain(low,high,previous.quantum,previous.excludes_negative_zero)
        # Candidate costs must expand the selected definitions of this
        # path, rather than the original definitions of other branches.
        cost_view=object.__new__(LiteralView)
        cost_view.model=self.registry.model;cost_view.context=self.registry.context
        cost_view.definitions=texts
        # Proofs of original roots must not be reused after choosing an
        # arm: their payload can still contain branches discarded here.
        # This isolated compiler registers only selected, completed values.
        words=StringCompiler(dtype=self.compiler.dtype,max_characters=self.compiler.max_characters,
            max_passes=self.compiler.max_passes)
        words.branch_facts=self.facts;words.expression_size=cost_view.size
        from direct_sympy_words import simplify_words
        from direct_sympy_conversions import ConversionSession,FiniteSource
        from direct_sympy_arithmetic import simplify_arithmetic
        from direct_sympy_synchronize import frontier
        # Type/range certificates remain true on a reachable subset; word
        # payload identities do not. This session contains only current-path
        # aliases and never mutates the original producer proof tables.
        numeric=ConversionSession(words,domains)
        numeric.branch_facts=facts
        if getattr(self.registry.model,'conversions',None) is not None:
            source=self.registry.model.conversions
            for name in domains:
                original=source.key(syntax(name));own=numeric.key(syntax(name))
                if original in source.half_values:numeric.half_values.add(own)
                if original in source.f32_values:numeric.f32_values.add(own)
        def certify_selected(alias):
            proofs=getattr(self.registry,'definition_proofs',())
            if alias>=len(proofs):return
            bound,kind,positive_zero,arm_bounds=proofs[alias]
            if bound is None:return
            local=recipe_bounds.get(alias)
            if local is not None:
                low,high=max(bound.minimum,local.minimum),min(bound.maximum,local.maximum)
                if low>high:raise UnreachableNumericPath('Selected recipe contradicts its stored enclosure')
                bound=FiniteSource(low,high,min(bound.quantum,local.quantum),max(bound.minimum_magnitude,local.minimum_magnitude))
            selected=syntax(texts[alias])
            if isinstance(selected,ast.Constant) or isinstance(selected,ast.UnaryOp) and isinstance(selected.op,ast.USub) and isinstance(selected.operand,ast.Constant):
                actual=numeric.bounds(selected)
                if actual is not None:
                    low,high=max(bound.minimum,actual.minimum),min(bound.maximum,actual.maximum)
                    if low>high:raise UnreachableNumericPath('Selected constant contradicts its stored enclosure')
                    bound=FiniteSource(low,high,bound.quantum,max(bound.minimum_magnitude,actual.minimum_magnitude))
            if alias in rms_bounds:
                local=rms_bounds[alias];low,high=max(bound.minimum,local.minimum),min(bound.maximum,local.maximum)
                magnitude=max(bound.minimum_magnitude,local.minimum_magnitude)
                if low>high or magnitude>max(abs(low),abs(high)):
                    raise UnreachableNumericPath('RMS guard contradicts the selected component enclosure')
                bound=FiniteSource(low,high,bound.quantum,magnitude)
                self.stats['rmsBranchRefinements']+=1
            if arm_bounds:
                arms=frontier(self.originals[alias],pure_functions=['CompileValue'+str(i) for i in range(alias)])
                if arms is not None and len(arms)==len(arm_bounds):
                    for (_,condition),local in zip(arms,arm_bounds):
                        truth=self.facts.truth(condition,facts)
                        if truth is None:break
                        if not truth:continue
                        if local is not None:
                            low,high=max(bound.minimum,local.minimum),min(bound.maximum,local.maximum)
                            magnitude=max(bound.minimum_magnitude,local.minimum_magnitude)
                            if low>high or magnitude>max(abs(low),abs(high)):
                                raise ValueError('Selected producer contradicts its numerical certificate')
                            bound=FiniteSource(low,high,bound.quantum,magnitude)
                        break
            key=numeric.key(syntax('CompileValue'+str(alias)+'()'))
            numeric.completed[key]=bound;numeric.converted_regions.add(key)
            # This marker names the audited, selected numerical producer.
            # Its immutable scalar is pure in this prefix. Without this
            # certificate a following constant-cell proof rejects the marker
            # as an unknown call, even when its value is already a singleton.
            numeric.pure_numeric_regions.add(key)
            if kind=='half':numeric.half_values.add(key)
            if kind in ('half','f32'):numeric.f32_values.add(key)
            if positive_zero:numeric.no_negative_zero_values.add(key)
            # Conditions may exclude the central magnitude gap of an alias
            # without selecting one sign. Refine only predicates proved here.
            for predicate in self.predicates.values():
                truth=self.facts.truth(predicate,facts)
                if truth is None:continue
                own=numeric.magnitude_guard_bounds(predicate,truth,numeric.completed)
                if own is None:raise UnreachableNumericPath('Selected producer contradicts its path conditions')
                numeric.completed=own
        def stabilize_selected(expression):
            stable=self.compiler.stabilize(expression,domains,facts=facts)
            arithmetic=simplify_arithmetic(stable,numeric)
            if cost_view.size(arithmetic)<cost_view.size(stable):stable=arithmetic
            if ALIASES.fullmatch(stable) or not re.search(r'\b(?:Bits64|Float64|U64[A-Za-z0-9]*)\s*\(',stable):
                return stable
            candidate=simplify_words(stable,words,domains)
            candidate=self.compiler.stabilize('('+candidate+')',domains,facts=facts)
            return candidate if cost_view.size(candidate)<cost_view.size(stable) else stable
        def reclose_recipe(alias,baseline):
            recipe=getattr(self.registry,'definition_recipes',{}).get(alias)
            if recipe is None:return baseline
            dependencies=list(dict.fromkeys(int(m[1]) for m in ALIASES.finditer(recipe)))
            if any(dependency>=alias for dependency in dependencies):return baseline
            # Word closure can inline a previous producer's payload, hiding
            # its marker from the selected tree. Its numerical recipe still
            # needs that scalar's current-context certificate. Select and
            # stabilize those dependencies first instead of silently falling
            # back to the old, broadly expanded payload. Undecided original
            # dependencies remain a barrier; no new path is assumed here.
            for dependency in dependencies:
                if dependency in memo:continue
                if self.next_decision(self.trees[dependency],facts) is not None:return baseline
                select(syntax(f'CompileValue{dependency}()'))
            self.stats['numericRecipeAttempts']+=1
            source=getattr(self.registry.model,'conversions',None)
            with numeric.branch_context(domains,{},facts):
                # These enclosures describe the exact pre-expansion operation
                # on the same operand values. Transfer only numerical proofs,
                # never an original closed word payload or conversion identity.
                if source is not None:
                    for child in ast.walk(syntax(recipe)):
                        old=source.key(child);own=numeric.key(child)
                        # A globally valid enclosure may be much wider than
                        # this branch. Recompute from selected operands first;
                        # copy the old bound only when no local proof exists.
                        if old in source.completed and numeric.bounds(child) is None:numeric.completed.setdefault(own,source.completed[old])
                        if old in source.half_values:numeric.half_values.add(own)
                        if old in source.f32_values:numeric.f32_values.add(own)
                        if old in source.no_negative_zero_values:numeric.no_negative_zero_values.add(own)
                        if old in source.rms_guards:numeric.rms_guards[own]=source.rms_guards[old]
                local_bound=numeric.bounds(syntax(recipe))
                if local_bound is not None:recipe_bounds[alias]=local_bound
                try:candidate=numeric.close('('+recipe+')')
                except ValueError as error:
                    if 'budget' not in str(error).lower():raise
                    self.stats['numericRecipeBudgetStops']+=1;return baseline
                if re.search(r'\b(?:R16|R32|sqrt|Silu16)\s*\(',candidate):return baseline
                tree=syntax(candidate)
                if self.next_decision(tree,facts) is not None:return baseline
                candidate=stabilize_selected('('+ast.unparse(select(tree))+')')
                if cost_view.size(candidate)<cost_view.size(baseline):
                    self.stats['numericRecipeAdmissions']+=1;return candidate
            return baseline
        def select(node):
            alias=self.alias(node)
            if alias is not None:
                if alias not in memo:
                    own=rms_bounds.get(alias)
                    if own is not None and own.minimum==own.maximum and own.minimum!=0:self.stats['rmsConstantCells']+=1
                    selected=ast.Constant(value=own.minimum) if own is not None and own.minimum==own.maximum and own.minimum!=0 else select(self.trees[alias])
                    texts[alias]=stabilize_selected('('+ast.unparse(selected)+')')
                    texts[alias]=reclose_recipe(alias,texts[alias])
                    words.register_completed_region(texts[alias],domains,word_closed=True)
                    words.copy_completed_word_root(texts[alias],f'CompileValue{alias}()',domains)
                    certify_selected(alias)
                    memo.add(alias)
                return node
            if isinstance(node,ast.Call) and node.func.id=='Piecewise':
                for pair in node.args:
                    body,condition=pair.elts
                    truth=self.condition_truth(condition,facts)
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
            expression=stabilize_selected('('+ast.unparse(select(root))+')')
        finally:
            self.compiler.expression_size=original_cost
            self.compiler.events.extend(words.events)
        self.stats['numericArithmeticEliminated']+=numeric.arithmetic_eliminated
        self.stats['CASPasses']+=len(self.compiler.events)-before
        return LiteralView(self.registry,texts),expression

    def arms(self,expression):
        root=self.registry.audit(expression);pending=[((),())]
        while pending:
            facts,guards=pending.pop();domains=self.domains(facts)
            if domains is None:self.stats['contradictions']+=1;continue
            from direct_sympy_projection_constraints import impossible_projections
            if impossible_projections(self.registry,guards):
                self.stats['contradictions']+=1;self.stats['coupledProjectionContradictions']+=1;continue
            decision=self.next_decision(root,facts)
            if decision is None:
                if self.stats['completedPaths']>=self.max_paths:raise ValueError('Path budget exceeded; no complete result')
                try:view,body=self.literal(root,facts,domains)
                except UnreachableNumericPath:
                    self.stats['contradictions']+=1;continue
                self.stats['completedPaths']+=1
                yield PathArm(view,body,guards);continue
            # Freeze this guard BEFORE assuming its own truth or a later
            # predicate. Later arm-local identities must not alter dispatch.
            try:view,condition=self.literal(decision,facts,domains)
            except UnreachableNumericPath:
                self.stats['contradictions']+=1;continue
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
                    emit('Piecewise(');count=0;pruner=OrderedGuardPruner(self.stats)
                    for arm in self.arms(expression):
                        arm=pruner.prune(arm)
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
                    if not count:raise ValueError('No reachable numerical path; no complete result')
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

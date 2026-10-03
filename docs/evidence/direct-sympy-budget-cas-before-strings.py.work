"""String-only substitution/factor/simplify engine for the direct compiler.

JSON is not an expression format here. SymPy supplies algebra; an independent
dyadic certificate gates IEEE rewrites. Legacy JSON compilers remain reference
tools until the checkpoint adapter has migrated and its final artifact passes.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass, replace
from fractions import Fraction
import re

import sympy as sp


# Only grammar admission is reusable. Always parse a fresh tree: conversion
# and branch transformers mutate their ASTs. Numeric/context proofs are not
# part of this cache and remain owned by the compiler session.
_validated_syntax=OrderedDict()
_validated_characters=0
_syntax_cache_limit=8*1024*1024


@dataclass(frozen=True)
class Domain:
    minimum: Fraction
    maximum: Fraction
    quantum: int
    excludes_negative_zero: bool = False

    def __post_init__(self):
        if self.minimum > self.maximum or not -1074 <= self.quantum <= 1023:
            raise ValueError("Invalid finite dyadic domain")


def power_two(exponent):
    return Fraction(2) ** exponent


def quantum(value):
    if not value:
        return 0
    denominator = value.denominator
    if denominator & (denominator - 1):
        raise ValueError("Non-dyadic value")
    numerator = abs(value.numerator)
    return (numerator & -numerator).bit_length() - denominator.bit_length()


def syntax(text):
    """Parse a deliberately small SymPy-compatible grammar, without eval."""
    global _validated_characters
    tree = ast.parse(text, mode="eval").body
    if text in _validated_syntax:
        _validated_syntax.move_to_end(text)
        return tree
    allowed = (ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp, ast.Name,
               ast.Constant, ast.Call, ast.Tuple, ast.Compare, ast.BoolOp,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub,
               ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.And,
               ast.Or, ast.Not)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("Unsupported mathematical string syntax")
        if isinstance(node, ast.Name) and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.id):
            raise ValueError("Invalid mathematical identifier")
        if isinstance(node, ast.Constant) and (not isinstance(node.value, (int, float, bool))):
            raise ValueError("Only numeric/boolean constants are admitted")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords):
            raise ValueError("Only named mathematical functions are admitted")
        if isinstance(node, ast.Compare) and len(node.ops) != 1:
            raise ValueError("Use explicit And for multiple comparisons")
    if len(text)<=_syntax_cache_limit:
        while _validated_syntax and _validated_characters+len(text)>_syntax_cache_limit:
            oldest,_=_validated_syntax.popitem(last=False)
            _validated_characters-=len(oldest)
        _validated_syntax[text]=True
        _validated_characters+=len(text)
    return tree


def symbolic(node):
    """No sympify/parse_expr eval, and no early arithmetic reassociation."""
    if isinstance(node, ast.Name):
        return sp.Symbol(node.id, real=True)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return sp.true if node.value else sp.false
        return sp.Rational(str(node.value))
    if isinstance(node, ast.Tuple):
        return sp.Tuple(*(symbolic(x) for x in node.elts))
    if isinstance(node, ast.UnaryOp):
        value = symbolic(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.Not):
            return sp.Not(value, evaluate=False)
        return sp.Mul(-1, value, evaluate=False)
    if isinstance(node, ast.BinOp):
        a, b = symbolic(node.left), symbolic(node.right)
        if isinstance(node.op, ast.Add):
            return sp.Add(a, b, evaluate=False)
        if isinstance(node.op, ast.Sub):
            return sp.Add(a, sp.Mul(-1, b, evaluate=False), evaluate=False)
        if isinstance(node.op, ast.Mult):
            return sp.Mul(a, b, evaluate=False)
        if isinstance(node.op, ast.Div):
            return sp.Mul(a, sp.Pow(b, -1, evaluate=False), evaluate=False)
        return sp.Pow(a, b, evaluate=False)
    if isinstance(node, ast.Compare):
        kind = {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge,
                ast.Eq: sp.Eq, ast.NotEq: sp.Ne}[type(node.ops[0])]
        return kind(symbolic(node.left), symbolic(node.comparators[0]), evaluate=False)
    if isinstance(node, ast.BoolOp):
        return (sp.And if isinstance(node.op, ast.And) else sp.Or)(
            *(symbolic(x) for x in node.values), evaluate=False)
    if isinstance(node, ast.Call):
        args = [symbolic(x) for x in node.args]
        if node.func.id == "Piecewise":
            return sp.Piecewise(*args, evaluate=False)
        if node.func.id in ("And", "Or", "Not", "Eq", "Ne", "Lt", "Le", "Gt", "Ge"):
            return getattr(sp, node.func.id)(*args, evaluate=False)
        # IEEE/bitwise boundaries are named symbolic functions; SymPy may
        # inspect their arguments but cannot silently turn them into real ops.
        return sp.Function(node.func.id)(*args)
    raise ValueError("Unsupported mathematical syntax")


def certificate(node, domains, dtype):
    """Prove all original/replacement operations exact using rational bounds.

    Bounds never use floating endpoint arithmetic. Unknown functions, general
    division and signed-zero-sensitive islands are barriers, not assumptions.
    """
    precision, minimum_quantum, max_exp = (24, -149, 127) if dtype == "f32" else (53, -1074, 1023)
    maximum_value = (2 - power_two(1 - precision)) * power_two(max_exp)

    def bounded(low, high, q, no_negative_zero):
        peak = max(abs(low), abs(high))
        if q < minimum_quantum or peak > maximum_value or peak / power_two(q) > 2 ** precision:
            return None
        return Domain(low, high, q, no_negative_zero or low > 0 or high < 0)

    def visit(n):
        if isinstance(n, ast.Name):
            return domains.get(n.id)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            # Python's decimal-to-binary conversion is the runtime literal
            # meaning. Reject a decimal not exactly equal to its rational.
            value = Fraction(n.value)
            if value != Fraction(str(n.value)):
                return None
            return bounded(value, value, quantum(value), True)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            d = visit(n.operand)
            if not d:
                return None
            if isinstance(n.op, ast.UAdd):
                return d
            return bounded(-d.maximum, -d.minimum, d.quantum, d.minimum > 0 or d.maximum < 0)
        if not isinstance(n, ast.BinOp):
            return None
        a, b = visit(n.left), visit(n.right)
        if not a or not b:
            return None
        if isinstance(n.op, ast.Add):
            return bounded(a.minimum + b.minimum, a.maximum + b.maximum,
                           min(a.quantum, b.quantum), a.excludes_negative_zero or b.excludes_negative_zero)
        if isinstance(n.op, ast.Sub):
            return bounded(a.minimum - b.maximum, a.maximum - b.minimum,
                           min(a.quantum, b.quantum), a.excludes_negative_zero)
        if isinstance(n.op, ast.Mult):
            endpoints = [x * y for x in (a.minimum, a.maximum) for y in (b.minimum, b.maximum)]
            return bounded(min(endpoints), max(endpoints), a.quantum + b.quantum,
                           (a.minimum > 0 or a.maximum < 0) and (b.minimum > 0 or b.maximum < 0))
        if isinstance(n.op, ast.Div) and b.minimum == b.maximum and b.minimum:
            value = b.minimum
            if abs(value.numerator) & (abs(value.numerator) - 1):
                return None
            endpoints = [a.minimum / value, a.maximum / value]
            return bounded(min(endpoints), max(endpoints), a.quantum - quantum(value),
                           a.minimum > 0 or a.maximum < 0)
        if isinstance(n.op, ast.Pow) and isinstance(n.right, ast.Constant) and type(n.right.value) is int and 0 <= n.right.value <= 8:
            count = n.right.value
            result = Domain(Fraction(1), Fraction(1), 0, True)
            for _ in range(count):
                endpoints = [x*y for x in (result.minimum,result.maximum) for y in (a.minimum,a.maximum)]
                result = bounded(min(endpoints), max(endpoints), result.quantum+a.quantum,
                                 result.minimum>0 and (a.minimum>0 or a.maximum<0))
                if result is None:
                    return None
            return result
        return None

    return visit(node)


def refine(domains, condition, truth):
    """Refine one branch, never mutate the sibling context."""
    if isinstance(condition, ast.Constant) and type(condition.value) is bool:
        return dict(domains) if condition.value == truth else None
    if not isinstance(condition, ast.Compare):
        return dict(domains)
    if not isinstance(condition.left, ast.Name) or not isinstance(condition.comparators[0], ast.Constant):
        return dict(domains)
    name = condition.left.id
    d = domains.get(name)
    if d is None:
        return dict(domains)
    target = Fraction(condition.comparators[0].value)
    op = type(condition.ops[0])
    if not truth:
        op = {ast.Lt:ast.GtE,ast.LtE:ast.Gt,ast.Gt:ast.LtE,ast.GtE:ast.Lt,ast.Eq:ast.NotEq,ast.NotEq:ast.Eq}[op]
    step = power_two(d.quantum)
    floor = target // step * step
    ceiling = -((-target) // step) * step
    low, high = d.minimum, d.maximum
    if op == ast.Lt:
        high = min(high, ceiling-step)
    elif op == ast.LtE:
        high = min(high, floor)
    elif op == ast.Gt:
        low = max(low, floor+step)
    elif op == ast.GtE:
        low = max(low, ceiling)
    elif op == ast.Eq:
        if floor != target:
            return None
        low, high = max(low,target), min(high,target)
    if low > high:
        return None
    result = dict(domains)
    result[name] = replace(d,minimum=low,maximum=high,
                          excludes_negative_zero=d.excludes_negative_zero or low>0 or high<0)
    return result


def cas_view(expression,*,keep_piecewise=False):
    """Protect function boundaries before asking CAS to inspect an envelope.

    Branch bodies are already individually stabilized. Their outer Piecewise
    envelope may be inspected without recursively repeating that same work.
    """
    protected={}
    atoms={}
    tree=syntax(expression)
    class Protect(ast.NodeTransformer):
        def visit_Call(self,child):
            # Descendant branches have already reached their own fixed point.
            # The envelope must not reopen all copies of those branch trees.
            if keep_piecewise and child is tree and child.func.id=="Piecewise":return self.generic_visit(child)
            body=ast.unparse(child)
            # Equal deterministic calls represent the same quantity, not
            # independent CAS variables. Exact strings include signed zeros.
            key=atoms.get(body)
            if key is None:
                key="CASBoundary"+str(len(protected))
                atoms[body]=key
                protected[key]=body
            return ast.copy_location(ast.Name(id=key,ctx=ast.Load()),child)
    return Protect().visit(tree),protected


class StringCompiler:
    def __init__(self, *, dtype="f32", max_characters=1048576, max_passes=32):
        if dtype not in ("f32","f64"):
            raise ValueError("Expected IEEE f32/f64 dtype")
        self.dtype, self.max_characters, self.max_passes = dtype,max_characters,max_passes
        self.events = []
        self.substitution_events = []
        self.failed_substitution = None
        self._stable={}
        self._cache_characters=0
        self._regions=OrderedDict()
        self._region_characters=0
        self._region_roots={}

    def context(self,domains):
        return tuple(sorted((name,d.minimum,d.maximum,d.quantum,d.excludes_negative_zero) for name,d in domains.items()))

    def register_completed_region(self,expression,domains,node=None,*,word_closed=False):
        """Admit a completed producer, never an interrupted intermediate.

        Callers must already have certified its fixed point in this domain
        (or restored an identity-checked producer). Only a whole call may be
        hidden: it was already opaque to algebra in cas_view. Runtime strings
        remain fully substituted; compiler placeholders are always restored.
        """
        if len(expression)>self.max_characters or re.search(r"\bCASStableRegion[0-9]+\b",expression):return
        node=syntax(expression) if node is None else node
        if not isinstance(node,ast.Call):return
        key=(self.context(domains),expression)
        if key in self._regions:return
        while self._regions and self._region_characters+len(expression)>4*self.max_characters:
            old,_=self._regions.popitem(last=False);self._region_characters-=len(old[1]);self._region_roots.pop(old,None)
        self._regions[key]=True;self._region_characters+=len(expression)
        from direct_sympy_words import word_width
        if word_closed:self._region_roots[key]=(node.func.id,word_width(node.args[0]) if node.func.id=="Float64" and len(node.args)==1 else None)

    def compact_regions(self,expression,context):
        if re.search(r"\bCASStableRegion[0-9]+\b",expression):return expression,{}
        protected={};current=expression
        regions=sorted((text for own,text in self._regions if own==context),key=len,reverse=True)
        for text in regions:
            # A registered region starts with an identifier and ends with a
            # balanced call. Never match a suffix of another function name.
            position=current.find(text)
            if position<0:continue
            name="CASStableRegion"+str(len(protected))
            chunks=[];start=0;replaced=False
            while position>=0:
                if position==0 or not (current[position-1].isalnum() or current[position-1]=='_'):
                    chunks.extend((current[start:position],name));start=position+len(text);replaced=True
                position=current.find(text,position+len(text))
            if replaced:
                chunks.append(current[start:]);current=''.join(chunks);protected[name]=text
        if not protected:return expression,{}
        # Only retain the envelope when every reachable arm has this exact
        # input domain. A new guard on a fundamental Xn can narrow it and
        # requires reopening the original producer in that arm. Word guards
        # which refine no input domain still run their own factor/simplify.
        if re.search(r"\bPiecewise\s*\(",current):
            domains={name:Domain(low,high,q,zero) for name,low,high,q,zero in context}
            for node in ast.walk(syntax(current)):
                if not isinstance(node,ast.Call) or node.func.id!="Piecewise":continue
                remaining=dict(domains)
                for pair in node.args:
                    if not isinstance(pair,ast.Tuple) or len(pair.elts)!=2:return expression,{}
                    condition=pair.elts[1]
                    own=refine(remaining,condition,True)
                    rest=refine(remaining,condition,False)
                    if own is not None and own!=domains:return expression,{}
                    if rest is not None and rest!=domains:return expression,{}
                    if rest is None:break
                    remaining=rest
        return current,protected

    def stabilize(self, expression, domains, path=()):
        if len(expression)>self.max_characters:raise ValueError("String expression budget exceeded; no partial result admitted")
        context=self.context(domains)
        key=(expression,context)
        cached=self._stable.get(key)
        if cached is not None:
            result,view=cached
            # Every occurrence still passes factor then simplify. Previously
            # certified descendants need not be parsed/proved again under
            # the identical context. Copies in the literal string remain.
            sp.simplify(sp.factor(view))
            self.events.append((path,0,"factor","simplify",len(expression),len(result),"cached-fixed-point"))
            return result
        compact,regions=self.compact_regions(expression,context)
        stabilized=self._stabilize(compact,domains,path)
        view,_=cas_view(stabilized,keep_piecewise=True)
        result=re.sub(r"\bCASStableRegion[0-9]+\b",lambda match:regions[match.group(0)],stabilized) if regions else stabilized
        if len(result)>self.max_characters:raise ValueError("Factored string exceeds output budget")
        if regions:
            # Annotate the actual CAS pass; do not count the restoration as
            # another factor/simplify invocation in producer growth metrics.
            self.events[-1]=self.events[-1][:6]+("completed-regions-envelope",)
        footprint=len(expression)+len(result)
        if footprint<=4*self.max_characters:
            while self._stable and self._cache_characters+footprint>4*self.max_characters:
                old_key=next(iter(self._stable));old_value=self._stable.pop(old_key)
                self._cache_characters-=len(old_key[0])+len(old_value[0])
            self._stable[key]=(result,symbolic(view))
            self._cache_characters+=footprint
        return result

    def _stabilize(self, expression, domains, path=()):
        if len(expression) > self.max_characters:
            raise ValueError("String expression budget exceeded; no partial result admitted")
        current = expression
        for iteration in range(self.max_passes):
            node = syntax(current)
            if not (isinstance(node,ast.Call) and node.func.id=="Piecewise"):
                compiler = self
                class NestedBranches(ast.NodeTransformer):
                    def visit_Call(self, child):
                        if isinstance(child.func,ast.Name) and child.func.id=="Piecewise":
                            return syntax(compiler.stabilize(ast.unparse(child),domains,path+("nested",)))
                        return self.generic_visit(child)
                transformed = NestedBranches().visit(node)
                # Reprinting preserves the AST's operation order; it performs
                # no algebra and retains unary minus on signed-zero literals.
                current = ast.unparse(transformed)
                node = syntax(current)
            # Piecewise guards are ordered. Each following arm inherits the
            # negation of preceding guards, and owns its separate context.
            if isinstance(node, ast.Call) and node.func.id == "Piecewise":
                remaining = dict(domains)
                arms = []
                for index, pair in enumerate(node.args):
                    if not isinstance(pair, ast.Tuple) or len(pair.elts)!=2:
                        raise ValueError("Piecewise requires (expression, condition) pairs")
                    branch, condition = pair.elts
                    own = refine(remaining,condition,True)
                    if own is not None:
                        body = self.stabilize(ast.unparse(branch),own,path+(index,))
                        arms.append((body,ast.unparse(condition)))
                    remaining = refine(remaining,condition,False)
                    if remaining is None:
                        break
                if not arms:
                    raise ValueError("No reachable Piecewise branch in the certified domain")
                candidate = "Piecewise("+", ".join("(("+body+"), "+guard+")" for body,guard in arms)+")"
                # Mandatory factor then simplify at the branch envelope too.
                envelope,_=cas_view(candidate,keep_piecewise=True)
                factored = sp.factor(symbolic(envelope))
                sp.simplify(factored)
                self.events.append((path,iteration,"factor","simplify",len(current),len(candidate),"branch-contexts"))
            else:
                # Opaque numerical/bitwise boundaries are compiler-only CAS
                # atoms. Avoid simplify recursively inspecting a rounded
                # function's huge arguments as if they were real arithmetic.
                # Restore the original mathematical string before proof and
                # emission: no atom identifiers survive the replacement.
                # The transformer mutates its tree; retain the original AST
                # for certification and parse a separate CAS view.
                cas_node,protected=cas_view(current)
                value = symbolic(cas_node)
                factored = sp.factor(value)
                simplified = sp.simplify(factored)
                # simplify may distribute a numerical common factor again.
                # Restore the factored form after mandatory factor/simplify;
                # repeating the complete cycle must leave it unchanged.
                candidate = sp.sstr(sp.factor(simplified))
                if protected:
                    candidate=re.sub(r"\bCASBoundary[0-9]+\b",lambda match:"("+protected[match.group(0)]+")",candidate)
                original = certificate(node,domains,self.dtype)
                replacement = certificate(syntax(candidate),domains,self.dtype)
                permitted = original is not None and replacement is not None and original.excludes_negative_zero and replacement.excludes_negative_zero
                # Unknown/non-exact IEEE regions retain original operation
                # order and boundaries even if CAS proposes a smaller form.
                if len(candidate)>self.max_characters:
                    raise ValueError("Factored string exceeds output budget")
                if not permitted:
                    candidate = current
                self.events.append((path,iteration,"factor","simplify",len(current),len(candidate),"certified" if permitted else "IEEE-barrier"))
            if candidate == current or ast.dump(syntax(candidate))==ast.dump(syntax(current)):
                return candidate
            current = candidate
        raise ValueError("factor/simplify did not stabilize; next substitution forbidden")

    def substitute(self, expression, name, replacement, domains, path=()):
        if not re.fullmatch(r"X[1-9][0-9]*",name):
            raise ValueError("Substitution variables must be Xn")
        # Identifier boundaries: X1 must not modify X10 or function names.
        # Replacement always carries its own parentheses before CAS parsing.
        enclosed = "("+expression+")"
        pattern = r"\b"+re.escape(name)+r"\b"
        occurrences = len(re.findall(pattern,enclosed))
        estimated = len(enclosed)+occurrences*(len(replacement)+2-len(name))
        event=(name,len(expression),len(replacement),occurrences,estimated)
        index=len(self.substitution_events)
        self.substitution_events.append(event+(None,"pending"))
        if estimated>self.max_characters:
            self.substitution_events[index]=event+(None,"budget")
            self.failed_substitution=(expression,name,replacement)
            raise ValueError(f"Substitution exceeds string budget before allocation: variable={name} templateCharacters={len(expression)} replacementCharacters={len(replacement)} occurrences={occurrences} estimatedCharacters={estimated} limit={self.max_characters}")
        if (self.context(domains),replacement) not in self._regions:syntax(replacement)
        result = re.sub(pattern,lambda _:"("+replacement+")",enclosed)
        stabilized=self.stabilize(result,domains,path)
        self.substitution_events[index]=event+(len(stabilized),"admitted")
        return stabilized

    def compile(self, expression, definitions, domains):
        """Backward substitution. An entire fixed point precedes next Xn."""
        current = self.stabilize("("+expression+")",domains)
        expansions = 0
        while True:
            names = [n.id for n in ast.walk(syntax(current)) if isinstance(n,ast.Name) and n.id in definitions]
            if not names:
                return current
            name = names[0]
            # Reusing a dependency in a DAG is valid. Check dependency cycles,
            # not whether the same name has been substituted previously.
            def acyclic(variable,active,done):
                if variable in active:raise ValueError("Cyclic dependency")
                if variable in done:return
                active=active|{variable}
                replacement=self.stabilize(definitions[variable],domains)
                for child in ast.walk(syntax(replacement)):
                    if isinstance(child,ast.Name) and child.id in definitions:
                        acyclic(child.id,active,done)
                done.add(variable)
            acyclic(name,set(),set())
            expansions+=1
            if expansions>10000:raise ValueError("Substitution step budget exceeded")
            current = self.substitute(current,name,definitions[name],domains)

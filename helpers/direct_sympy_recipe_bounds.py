"""Prefix-local forward/backward enclosures of ordered numerical recipes.

Only proof intervals move. The original arithmetic, conversion boundaries
and zero signs remain in the generated expressions. Work limits retain all
proved enclosures and never discard an input or path.
"""
import ast,math,re,struct
from fractions import Fraction as F
from direct_sympy_strings import syntax
from direct_sympy_conversions import ConversionSession,FiniteSource


def neighbor(value,kind,up):
    fmt,word,width=('e','H',16) if kind=='R16' else ('f','I',32)
    if value==0:return (1 if up else -1)*2**(-24 if width==16 else -149)
    raw=struct.unpack(word,struct.pack(fmt,value))[0]
    raw+=(1 if up else -1) if value>0 else (-1 if up else 1)
    return struct.unpack(fmt,struct.pack(word,raw))[0]


def conversion_preimage(bound,kind):
    """Closed conservative RN-even cell envelope, including both zero signs."""
    fmt='e' if kind=='R16' else 'f';result=[]
    for value,up in ((bound.minimum,False),(bound.maximum,True)):
        try:stored=struct.unpack(fmt,struct.pack(fmt,value))[0]
        except (OverflowError,ValueError):return None
        if not math.isfinite(stored):return None
        # Enclose nonrepresentable certificate endpoints before choosing
        # adjacent cells. Equality at a midpoint stays conservatively inside.
        if stored>value and not up:stored=neighbor(stored,kind,False)
        if stored<value and up:stored=neighbor(stored,kind,True)
        if not math.isfinite(stored):return None
        other=neighbor(stored,kind,up)
        if math.isfinite(other):endpoint=(F(stored)+F(other))/2
        else:endpoint=F(65520 if kind=='R16' else 2**128-2**103)*(1 if up else -1)
        rounded=float(endpoint)
        if (F(rounded)<endpoint if up else F(rounded)>endpoint):rounded=math.nextafter(rounded,math.inf if up else -math.inf)
        result.append(rounded)
    return FiniteSource(*result)


def propagate_recipe_bounds(registry,refined,stats=None,*,assumptions=(),max_passes=64,max_nodes=20000):
    """None means an actual proof contradiction; unsupported nodes are barriers."""
    if not refined and not assumptions:return dict(refined)
    stats={} if stats is None else stats
    model=registry.model;session=ConversionSession(model.compiler,model.domains)
    source=getattr(model,'conversions',None)
    proofs={};nodes={};visited=0;literal_keys={}
    for index,text in enumerate(getattr(registry,'definitions',())):
        literal_keys.setdefault(session.key(syntax(text)),[]).append(index)
    def marker(key):return syntax(f'CompileValue{key}()') if type(key)is int else syntax(key)
    def put(key,bound,kind=None):
        node=marker(key);structural=session.key(node);session.completed[structural]=bound
        if kind=='half':session.half_values.add(structural)
        if kind in ('half','f32'):session.f32_values.add(structural)
        proofs[key]=bound
        if type(key)is int and key<len(getattr(registry,'definitions',())):
            session.completed[session.key(syntax(registry.definitions[key]))]=bound
    for name,domain in model.domains.items():
        put(name,FiniteSource(float(domain.minimum),float(domain.maximum),domain.quantum))
        if source is not None:
            old=source.key(syntax(name));own=session.key(syntax(name))
            if old in source.half_values:session.half_values.add(own)
            if old in source.f32_values:session.f32_values.add(own)
    for key,(bound,kind,_,_) in enumerate(getattr(registry,'definition_proofs',())):
        if bound is not None:put(key,bound,kind)
    for key,text in getattr(registry,'definition_recipes',{}).items():
        tree=syntax(text);count=sum(1 for _ in ast.walk(tree))
        if visited+count>max_nodes:
            stats['recipeConstraintBudgetStops']=stats.get('recipeConstraintBudgetStops',0)+1;break
        dependencies=[int(m[1])for m in re.finditer(r'\bCompileValue([0-9]+)\s*\(\s*\)',text)]
        if all(dependency<key for dependency in dependencies):nodes[key]=tree;visited+=count
    contradiction=False;changed=False
    def narrow(key,bound):
        nonlocal contradiction,changed
        old=proofs.get(key)
        if old is None or bound is None:return
        low,high=max(old.minimum,bound.minimum),min(old.maximum,bound.maximum)
        quanta=[q for q in (old.quantum,bound.quantum)if q is not None]
        q=max(quanta)if quanta else None
        if q is not None:
            step=F(2)**q;a,b=F(low)/step,F(high)/step
            low=float((-(-a.numerator//a.denominator))*step)
            high=float((b.numerator//b.denominator)*step)
        magnitude=max(old.minimum_magnitude,bound.minimum_magnitude)
        if low>high or magnitude>max(abs(low),abs(high)):
            contradiction=True;return
        new=FiniteSource(low,high,q,magnitude)
        if new!=old:
            put(key,new);changed=True
            stats['recipeConstraintRefinements']=stats.get('recipeConstraintRefinements',0)+1
    def scalar(node):
        if isinstance(node,ast.Name) and node.id in model.domains:return node.id
        if isinstance(node,ast.Call) and not node.args:
            match=re.fullmatch('CompileValue([0-9]+)',node.func.id)
            if match:return int(match[1])
        return None
    def outward(low,high):
        try:a,b=float(low),float(high)
        except OverflowError:return None
        if not all(math.isfinite(x)for x in (a,b)):return None
        if F(a)>low:a=math.nextafter(a,-math.inf)
        if F(b)<high:b=math.nextafter(b,math.inf)
        return FiniteSource(a,b)
    def backward(node,bound):
        if bound is None:return
        key=scalar(node)
        if key is not None:narrow(key,bound);return
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            backward(node.operand,bound if isinstance(node.op,ast.UAdd)else FiniteSource(-bound.maximum,-bound.minimum));return
        if isinstance(node,ast.Call) and len(node.args)==1 and node.func.id in ('R16','R32'):
            raw=conversion_preimage(bound,node.func.id)
            if raw is not None:
                stats['backwardRoundingSteps']=stats.get('backwardRoundingSteps',0)+1
                backward(node.args[0],raw)
            return
        if not isinstance(node,ast.BinOp):return
        a,b=session.bounds(node.left),session.bounds(node.right)
        if a is None or b is None:return
        # Original F64 sum/product error; no reassociation or exact-real
        # replacement enters the generated calculation. Fractions prevent
        # rounding a proof error down during its construction.
        if isinstance(node.op,(ast.Add,ast.Sub)):
            total=F(max(abs(a.minimum),abs(a.maximum)))+F(max(abs(b.minimum),abs(b.maximum)))
            error=total*F(1,2**52)+F(1,2**1074)
            low,high=F(bound.minimum)-error,F(bound.maximum)+error
            if isinstance(node.op,ast.Add):
                backward(node.left,outward(low-F(b.maximum),high-F(b.minimum)))
                backward(node.right,outward(low-F(a.maximum),high-F(a.minimum)))
            else:
                backward(node.left,outward(low+F(b.minimum),high+F(b.maximum)))
                backward(node.right,outward(F(a.minimum)-high,F(a.maximum)-low))
        elif isinstance(node.op,ast.Mult):
            for operand,constant,own in ((node.left,b,a),(node.right,a,b)):
                minimum=max(constant.minimum_magnitude,constant.minimum if constant.minimum>0 else -constant.maximum if constant.maximum<0 else 0)
                if not minimum:continue
                total=F(max(abs(constant.minimum),abs(constant.maximum)))*F(max(abs(own.minimum),abs(own.maximum)))
                error=total*F(1,2**52)+F(1,2**1074)
                lower,upper=F(bound.minimum)-error,F(bound.maximum)+error
                if constant.minimum<=0<=constant.maximum:
                    peak=max(abs(lower),abs(upper))/F(minimum);low,high=-peak,peak
                else:
                    values=[x/F(c)for x in (lower,upper)for c in (constant.minimum,constant.maximum)]
                    low,high=min(values),max(values)
                backward(operand,outward(low,high))
    for key,bound in refined.items():narrow(key,bound)
    if contradiction:return None
    for _ in range(max_passes):
        if contradiction:return None
        changed=False;stats['recipeConstraintPasses']=stats.get('recipeConstraintPasses',0)+1
        for key,tree in sorted(nodes.items()):narrow(key,session.bounds(tree))
        for condition,truth in assumptions:
            if not isinstance(condition,ast.Compare) or len(condition.ops)!=1 or len(condition.comparators)!=1:continue
            constrained=session.magnitude_guard_bounds(condition,truth,session.completed)
            if constrained is None:contradiction=True;break
            for key,bound in constrained.items():
                if bound==session.completed.get(key):continue
                session.completed[key]=bound;changed=True
                for alias in literal_keys.get(key,()):narrow(alias,bound)
                # Only recognized unsigned magnitude predicates can add
                # these entries. Their raw source stays a numerical recipe,
                # not an assumed dtype or a replacement for its calculation.
                backward(condition.left.args[0].args[0],bound)
        for key,tree in sorted(nodes.items(),reverse=True):
            if key in proofs:backward(tree,proofs[key])
        if contradiction:return None
        if not changed:return {key:bound for key,bound in proofs.items()if bound!=
            (FiniteSource(float(model.domains[key].minimum),float(model.domains[key].maximum),model.domains[key].quantum)
             if type(key)is str else registry.definition_proofs[key][0]) or key in refined}
    stats['recipeConstraintBudgetStops']=stats.get('recipeConstraintBudgetStops',0)+1
    return dict(proofs)

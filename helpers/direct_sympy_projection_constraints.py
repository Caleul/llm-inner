"""Compiler-only coupled projection exclusions; never alter model arithmetic."""
import ast,itertools,math,re,struct
from fractions import Fraction as F
from direct_sympy_strings import syntax
from direct_sympy_layer_bounds import sqrt_outward


def norm_squared_floor(sources,epsilon):
    """Lower norm of a finite Half RMS vector before exact +/-1 gamma.

    Use the original F32 error envelope in both directions. Half storage
    loses at most relative 2^-11 plus sqrt(n)*2^-25 in Euclidean norm.
    Zero-admitting inputs give floor zero and cannot exclude small outputs.
    """
    from direct_sympy_checkpoint import rms_half_bound
    n=len(sources)
    if not sources or rms_half_bound(n,epsilon) is None:return F(0)
    small=[]
    for s in sources:
        if s is None or max(abs(s.minimum),abs(s.maximum))>65504:return F(0)
        small.append(F(max(s.minimum_magnitude,s.minimum if s.minimum>0 else -s.maximum if s.maximum<0 else 0)))
    total=sum((x*x for x in small),F(0))
    if not total:return F(0)
    u=F(1,2**23)
    loss=(1+u)**3/((1-u)**3*sqrt_outward(1-n*u,False))
    lower=sqrt_outward(n*total/(total+n*F(epsilon)),False)/loss*(1-F(1,2**11))-sqrt_outward(F(n),True)*F(1,2**25)
    return max(F(0),lower)**2


def norm_error_bound(width,epsilon):
    """Euclidean error against real RMS, including original F32/Half steps."""
    from direct_sympy_checkpoint import rms_half_bound
    if rms_half_bound(width,epsilon) is None:return None
    u=F(1,2**23)
    loss=(1+u)**3/((1-u)**3*sqrt_outward(1-width*u,False))
    return sqrt_outward(F(width),True)*(loss-1+loss*F(1,2**11)+F(1,2**25))


def rms_branch_bounds(registry,assumptions):
    """Refine actual Half RMS components from their own tiny-value guard.

    For the original rounded mean m <= K*T/n+(1+u)*eps, a guard
    x_i^2 < c*m bounds the real RMS component by sqrt(c*K). Its source
    magnitude also bounds T from below. In two dimensions this bounds
    the other component, sometimes inside a single Half storage cell.
    On the complementary guard, m >= (1-u)^(n+3)*(T/n+eps)
    supplies a lower component bound. Neither the rounded mean nor the
    numerical RMS operations are replaced.
    """
    from direct_sympy_conversions import FiniteSource
    result={};context=registry.model.compiler.context(registry.model.domains)
    def same(a,b):return ast.dump(a)==ast.dump(b)
    def coefficient(node):
        if (isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow)
            and isinstance(node.left,ast.Constant) and type(node.left.value)is int and node.left.value==2
            and isinstance(node.right,ast.UnaryOp) and isinstance(node.right.op,ast.USub)
            and isinstance(node.right.operand,ast.Constant) and type(node.right.operand.value)is int
            and 28<=node.right.operand.value<=64):return F(1,2**node.right.operand.value)
        return None
    for vector in getattr(registry.model,'norm_vectors',{}).values():
        n=vector['width'];sources=vector.get('sources',());bounds=vector.get('sourceBounds',())
        if (n!=2 or len(sources)!=n or len(bounds)!=n or len(vector['components'])!=n
            or vector.get('context')!=context or 'mean' not in vector
            or any(g not in (-1,1) for g in vector.get('gamma',())) or len(vector.get('gamma',()))!=n):continue
        if any(b is None or not all(math.isfinite(x) for x in (b.minimum,b.maximum)) or max(abs(b.minimum),abs(b.maximum))>65504 for b in bounds):continue
        eps=vector['epsilon'];u=F(1,2**23);K=(1+u)**(n+3)
        if eps<=0:continue
        loss=(1+u)**3/((1-u)**3*sqrt_outward(1-n*u,False))
        mean=syntax(vector['mean']);inputs=[syntax(s) for s in sources]
        minimum=lambda b:F(max(b.minimum_magnitude,b.minimum if b.minimum>0 else -b.maximum if b.maximum<0 else 0))
        for predicate,truth in assumptions:
            if not isinstance(predicate,ast.Compare) or len(predicate.ops)!=1 or len(predicate.comparators)!=1:continue
            if not isinstance(predicate.ops[0],(ast.Lt,ast.GtE)):continue
            tiny_branch=bool(truth) if isinstance(predicate.ops[0],ast.Lt) else not truth
            left,right=predicate.left,predicate.comparators[0]
            if not isinstance(right,ast.BinOp) or not isinstance(right.op,ast.Mult):continue
            c=coefficient(right.right) if same(right.left,mean) else coefficient(right.left) if same(right.right,mean) else None
            if c is None:continue
            for i,source in enumerate(inputs):
                square=isinstance(left,ast.BinOp) and ((isinstance(left.op,ast.Pow) and same(left.left,source)
                    and isinstance(left.right,ast.Constant) and left.right.value==2)
                    or (isinstance(left.op,ast.Mult) and same(left.left,source) and same(left.right,source)))
                if not square:continue
                total=max(sum((minimum(b)**2 for b in bounds),F(0)),n*(minimum(bounds[i])**2/c-(1+u)*eps)/K)
                tiny=c*K
                other=max(F(0),n*total/(total+n*eps)-tiny)
                for j,b in enumerate(bounds):
                    if not tiny_branch and j!=i:continue
                    low=(F(0) if j==i else sqrt_outward(other,False)/loss) if tiny_branch else sqrt_outward(c*(1-u)**(n+3),False)/loss
                    high=sqrt_outward(tiny if tiny_branch and j==i else F(n),True)*loss
                    lo=float(low);hi=float(high)
                    if F(lo)>low:lo=math.nextafter(lo,-math.inf)
                    if F(hi)<high:hi=math.nextafter(hi,math.inf)
                    lo=struct.unpack('e',struct.pack('e',lo))[0];hi=struct.unpack('e',struct.pack('e',hi))[0]
                    pair=(lo,hi) if b.minimum>0 else (-hi,-lo) if b.maximum<0 else (-hi,hi)
                    g=vector['gamma'][j];a,z=sorted(float(g)*x for x in pair)
                    component=vector['components'][j];match=re.fullmatch(r'CompileValue([0-9]+)\(\)',component)
                    if match is None:continue
                    alias=int(match[1])
                    proofs=getattr(registry,'definition_proofs',())
                    if alias<len(proofs) and proofs[alias][0] is not None:
                        original=proofs[alias][0]
                        a,z=max(a,original.minimum),min(z,original.maximum)
                        if a==original.minimum and z==original.maximum and lo<=original.minimum_magnitude:continue
                    previous=result.get(alias)
                    if previous is not None:a,z=max(a,previous.minimum),min(z,previous.maximum)
                    if a>z:continue  # Contradictions are handled separately; retain unsupported paths.
                    result[alias]=FiniteSource(a,z,-24,lo)
    return result



def rms_source_branch_bounds(registry,assumptions):
    """Bound actual Half sources when all share a certified small-mean arm.

    Sum x_i^2 < sum(c_i)*m and use the original F32 upper envelope
    m <= K*T/n+(1+u)*eps. Solve the inequality before substituting any
    source. Bounds belong only to this path; neither m nor its order changes.
    Unsupported predicates/contexts keep their original domains.
    """
    from direct_sympy_conversions import FiniteSource
    from direct_sympy_input_partitions import value,MAX_RANK
    result={};context=registry.model.compiler.context(registry.model.domains)
    for vector in getattr(registry.model,'norm_vectors',{}).values():
        n=vector['width'];sources=vector.get('sources',());bounds=vector.get('sourceBounds',())
        if (n!=2 or len(sources)!=n or len(bounds)!=n or vector.get('context')!=context
            or 'mean' not in vector or vector.get('epsilon',0)<=0):continue
        if any(b is None or max(abs(b.minimum),abs(b.maximum))>65504 for b in bounds):continue
        mean=syntax(vector['mean']);coefficients={}
        for predicate,truth in assumptions:
            if not isinstance(predicate,ast.Compare) or len(predicate.ops)!=1 or len(predicate.comparators)!=1:continue
            if not (isinstance(predicate.ops[0],ast.Lt) and truth or isinstance(predicate.ops[0],ast.GtE) and not truth):continue
            right=predicate.comparators[0]
            if not isinstance(right,ast.BinOp) or not isinstance(right.op,ast.Mult):continue
            coefficient=right.right if ast.dump(right.left)==ast.dump(mean) else right.left if ast.dump(right.right)==ast.dump(mean) else None
            if not (isinstance(coefficient,ast.BinOp) and isinstance(coefficient.op,ast.Pow)
                and isinstance(coefficient.left,ast.Constant) and type(coefficient.left.value)is int and coefficient.left.value==2
                and isinstance(coefficient.right,ast.UnaryOp) and isinstance(coefficient.right.op,ast.USub)
                and isinstance(coefficient.right.operand,ast.Constant) and type(coefficient.right.operand.value)is int
                and 28<=coefficient.right.operand.value<=64):continue
            c=F(1,2**coefficient.right.operand.value);left=predicate.left
            for i,source in enumerate(sources):
                node=syntax(source)
                if isinstance(left,ast.BinOp) and (
                    isinstance(left.op,ast.Pow) and ast.dump(left.left)==ast.dump(node) and isinstance(left.right,ast.Constant) and left.right.value==2
                    or isinstance(left.op,ast.Mult) and ast.dump(left.left)==ast.dump(node) and ast.dump(left.right)==ast.dump(node)):
                    coefficients[i]=min(c,coefficients.get(i,c))
        if len(coefficients)!=n:continue
        u=F(1,2**23);C=sum(coefficients.values(),F(0));denominator=1-C*(1+u)**(n+3)/n
        if denominator<=0:continue
        cap=sqrt_outward(C*(1+u)*vector['epsilon']/denominator,True)
        # Sources already are Half. Select the largest representable endpoint
        # below the conservative real cap, without rounding it up to a cell.
        low,high=0,MAX_RANK
        while low<high:
            middle=(low+high+1)//2
            if value(middle)<=cap:low=middle
            else:high=middle-1
        endpoint=float(value(low))
        for source,b in zip(sources,bounds):
            match=re.fullmatch(r'CompileValue([0-9]+)\(\)',source)
            key=int(match[1]) if match else source if source in registry.model.domains else None
            if key is None:continue
            a,z=max(b.minimum,-endpoint),min(b.maximum,endpoint)
            if a>z:continue  # Reachability remains the separate guard proof's job.
            previous=result.get(key)
            if previous is not None:a,z=max(a,previous.minimum),min(z,previous.maximum)
            if a<=z:result[key]=FiniteSource(a,z,b.quantum,b.minimum_magnitude)
    return result


def projection_branch_bounds(registry, refined, *, max_products=2**20):
    """Propagate branch enclosures through actual whole-producer projections.

    Read weights on demand and reuse the original ordered-reduction bound.
    Frames are published in dependency order; no runtime arithmetic changes.
    A singleton nonzero Half cell can eliminate its full dependency tree.
    """
    from direct_sympy_conversions import FiniteSource
    from direct_sympy_layer_bounds import dot_interval
    result=dict(refined)
    if not result:return result
    model=registry.model;session=getattr(model,'conversions',None)
    if session is None:return result
    context=model.compiler.context(model.domains);spent=0
    for producer,frame in getattr(model,'linear_frames',{}).items():
        alias=registry.names.get(producer,'');match=re.fullmatch(r'CompileValue([0-9]+)\(\)',alias)
        if match is None or frame['context']!=context:continue
        operands=frame['operands']
        if spent+len(operands)>max_products:break
        spent+=len(operands);intervals=[];supported=True
        for text in operands:
            if text is None:intervals.append((0.0,0.0));continue
            node=syntax(text)
            if session.value_kind(node)!='half':supported=False;break
            bound=session.bounds(node)
            own=re.fullmatch(r'CompileValue([0-9]+)\(\)',text)
            local=result.get(int(own[1])) if own else None
            if bound is None:bound=local
            elif local is not None:
                low,high=max(bound.minimum,local.minimum),min(bound.maximum,local.maximum)
                if low>high:supported=False;break
                bound=FiniteSource(low,high,-24,0)
            if bound is None:supported=False;break
            intervals.append((bound.minimum,bound.maximum))
        if not supported:continue
        enclosed=dot_interval(model,frame['name'],frame['row'],intervals)
        if enclosed is None:continue
        low,high=enclosed;index=int(match[1]);previous=result.get(index)
        if previous is not None:low,high=max(low,previous.minimum),min(high,previous.maximum)
        if low>high:continue
        # Signed zero singleton identity is deliberately not asserted here.
        result[index]=FiniteSource(low,high,-24,min(abs(low),abs(high)) if low*high>0 else 0)
    return result


def inverse(matrix):
    """Exact small-system elimination; singular systems provide no exclusion."""
    n=len(matrix)
    if not n or any(len(row)!=n for row in matrix):return None
    rows=[[F(x) for x in row]+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for j in range(n):
        pivot=next((i for i in range(j,n) if rows[i][j]),None)
        if pivot is None:return None
        rows[j],rows[pivot]=rows[pivot],rows[j]
        value=rows[j][j];rows[j]=[x/value for x in rows[j]]
        for i in range(n):
            if i==j:continue
            value=rows[i][j]
            rows[i]=[a-value*b for a,b in zip(rows[i],rows[j])]
    return [row[n:] for row in rows]


def integer_grid_source(node):
    """Recognize only the emitted integer-grid rounding composition.

    This is a numerical error envelope, not an assertion that a normal
    kernel implements target subnormal semantics outside its own context.
    """
    def call(value,name):return isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id==name and len(value.args)==1
    if not call(node,'Float64') or not call(node.args[0],'U64FromF64'):return None
    value=node.args[0].args[0];kinds=[]
    while isinstance(value,ast.BinOp) and isinstance(value.op,ast.Sub) and isinstance(value.left,ast.BinOp) and isinstance(value.left.op,ast.Add):
        offset=value.right
        if ast.dump(offset)!=ast.dump(value.left.right):return None
        if not (isinstance(offset,ast.BinOp) and isinstance(offset.op,ast.Pow)
            and isinstance(offset.left,ast.Constant) and offset.left.value==2
            and isinstance(offset.right,ast.Constant) and offset.right.value in (81,94)):return None
        kinds.append('R32' if offset.right.value==81 else 'R16');value=value.left.left
    if not kinds or len(kinds)>2 or not call(value,'F64FromU64') or not call(value.args[0],'Bits64'):return None
    return value.args[0].args[0],tuple(reversed(kinds))


class LinearProof:
    def __init__(self,registry,vector,*,additional_bases=None,bounded_unknowns=False):
        self.registry=registry;self.vector=vector;self.n=len(vector['components'])
        self.base={text:i for i,text in vector['components'].items()}
        self.bounds=vector['magnitudes'];self.memo={};self.visiting=set()
        self.additional_bases=additional_bases or {};self.bounded_unknowns=bounded_unknowns
        self.closed_aliases={}

    def unknown(self,node):
        if not self.bounded_unknowns:return None
        session=getattr(self.registry.model,'conversions',None)
        if session is None:return None
        # Only completed scalar aliases or fundamental inputs may terminate
        # this analysis. Never guess a bound for an unsupported operation.
        if not (isinstance(node,ast.Name) or isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
            and re.fullmatch(r'CompileValue[0-9]+',node.func.id) and not node.args):return None
        bound=session.bounds(node)
        if bound is None:return None
        return ((F(0),)*self.n,F(0),F(max(abs(bound.minimum),abs(bound.maximum))))

    def magnitude(self,form):
        coefficients,bias,error=form
        return sum((abs(c)*self.bounds[i] for i,c in enumerate(coefficients)),abs(bias))+error

    def scale(self,form,value,rounded=True):
        c,b,e=form;result=(tuple(value*x for x in c),value*b,abs(value)*e)
        if rounded:
            if self.magnitude(result)>F(2)**1023:return None
            result=(*result[:2],result[2]+self.magnitude(result)*F(1,2**52)+F(1,2**1074))
        return result

    def form(self,node):
        text=ast.unparse(node)
        if text in self.additional_bases:return self.additional_bases[text]
        if text in self.base:
            return (tuple(F(i==self.base[text]) for i in range(self.n)),F(0),F(0))
        if text in self.closed_aliases:
            return self.form(syntax('CompileValue'+str(self.closed_aliases[text])+'()'))
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if not math.isfinite(node.value) or F(float(node.value))!=F(node.value):return None
            return ((F(0),)*self.n,F(node.value),F(0))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            own=self.form(node.operand);return self.scale(own,F(-1),False) if own is not None else None
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            rounded=integer_grid_source(node)
            if rounded is not None:
                source,kinds=rounded;own=self.form(source)
                if own is None:return None
                # The initial integer-word/F64 conversion may lose up to
                # eleven low word bits. Its numerical error is bounded by
                # relative 2^-40, plus an absolute F32-subnormal quantum.
                # Subsequent positive-offset additions round on the same
                # significand grids; keep every independent error term.
                own=(*own[:2],own[2]+self.magnitude(own)*F(1,2**40)+F(1,2**149))
                for kind in kinds:
                    magnitude=self.magnitude(own)
                    if magnitude>(65504 if kind=='R16' else F(2)**127):return None
                    own=(*own[:2],own[2]+magnitude*F(1,2**(11 if kind=='R16' else 23))+F(1,2**(25 if kind=='R16' else 149)))
                return own
            match=re.fullmatch(r'CompileValue([0-9]+)',node.func.id)
            if match and not node.args:
                index=int(match[1])
                if index in self.memo:return self.memo[index]
                if index in self.visiting:return None
                recipe=self.registry.definition_recipes.get(index)
                if recipe is None:return self.unknown(node)
                self.visiting.add(index);own=self.form(syntax(recipe));self.visiting.remove(index)
                if own is None:own=self.unknown(node)
                self.memo[index]=own;return own
            if node.func.id in ('R16','R32') and len(node.args)==1:
                own=self.form(node.args[0])
                if own is None:return None
                magnitude=self.magnitude(own)
                if magnitude>(65504 if node.func.id=='R16' else F(2)**127):return None
                error=(magnitude*F(1,2**11)+F(1,2**25) if node.func.id=='R16' else magnitude*F(1,2**23)+F(1,2**149))
                return (*own[:2],own[2]+error)
            return None
        if isinstance(node,ast.BinOp):
            a,b=self.form(node.left),self.form(node.right)
            if a is None or b is None:return None
            if isinstance(node.op,(ast.Add,ast.Sub)):
                sign=-1 if isinstance(node.op,ast.Sub) else 1
                result=(tuple(x+sign*y for x,y in zip(a[0],b[0])),a[1]+sign*b[1],a[2]+b[2])
                if self.magnitude(a)+self.magnitude(b)>F(2)**1023:return None
                error=(self.magnitude(a)+self.magnitude(b))*F(1,2**52)+F(1,2**1074)
                return (*result[:2],result[2]+error)
            if isinstance(node.op,ast.Mult):
                for constant,other in ((a,b),(b,a)):
                    if not any(constant[0]) and not constant[2]:return self.scale(other,constant[1])
        return self.unknown(node)


def norm_distance(registry,old,new):
    """Bound stored RMS-vector change, without changing its calculation.

    For y=x+d, real RMS has Jacobian norm <=sqrt(n)/||x||. Along the
    segment, ||x+t*d||>=r-D. Add both independent numerical error bounds.
    The source difference itself includes every original F32/Half rounding.
    """
    n=old['width']
    if new['width']!=n or not old.get('sources') or not new.get('sources'):return None
    if old.get('epsilon')!=new.get('epsilon') or old.get('context')!=new.get('context'):return None
    r=sqrt_outward(old['sourceNormFloor'],False)
    if not r:return None
    inputs={'components':dict(enumerate(old['sources'])),'magnitudes':old['inputMagnitudes']}
    proof=LinearProof(registry,inputs,bounded_unknowns=True);errors=[]
    for i,source in enumerate(new['sources']):
        own=proof.form(syntax(source))
        if own is None or own[0]!=tuple(F(i==j) for j in range(n)) or own[1]:return None
        errors.append(own[2])
    delta=sqrt_outward(sum((e*e for e in errors),F(0)),True)
    if delta>=r:return None
    return delta*sqrt_outward(F(n),True)/(r-delta)+old['roundingError']+new['roundingError']


def norm_links(registry,*,max_links=32):
    """Compiler-only correlated bases; a work limit never excludes inputs."""
    vectors=list(getattr(registry.model,'norm_vectors',{}).values())
    identity=(len(getattr(registry,'definitions',())),tuple((id(v),len(v['components'])) for v in vectors),max_links)
    previous=getattr(registry,'_norm_links',None)
    if previous is not None and previous[0]==identity:return previous[1]
    links={};checked=0
    for old in vectors:
        if len(old['components'])!=old['width'] or not old.get('floor'):continue
        related={}
        for new in vectors:
            if old is new or len(new['components'])!=new['width']:continue
            checked+=1
            if checked>max_links:break
            distance=norm_distance(registry,old,new)
            if distance is None:continue
            for i,text in new['components'].items():
                sign=new['gamma'][i]/old['gamma'][i]
                related[text]=(tuple(sign*F(i==j) for j in range(old['width'])),F(0),distance)
        links[id(old)]=related
    registry._norm_links=(identity,links)
    return links


def small_source(guard):
    """Recognize only an unsigned IEEE magnitude guard with an upper bound."""
    node=syntax(guard.expression)
    if not isinstance(node,ast.Compare) or len(node.ops)!=1:return None
    upper=isinstance(node.ops[0],(ast.Lt,ast.LtE)) and guard.truth or isinstance(node.ops[0],(ast.Gt,ast.GtE)) and not guard.truth
    if not upper:return None
    raw,limit=node.left,node.comparators[0]
    if not isinstance(raw,ast.Call) or not isinstance(raw.func,ast.Name) or raw.func.id!='U64And' or len(raw.args)!=2:return None
    bits,mask=raw.args
    if not isinstance(mask,ast.Constant) or mask.value!=0x7fffffffffffffff:return None
    if not isinstance(bits,ast.Call) or not isinstance(bits.func,ast.Name) or bits.func.id!='Bits64' or len(bits.args)!=1:return None
    if not isinstance(limit,ast.Constant) or type(limit.value)is not int or not 0<=limit.value<0x7ff0000000000000:return None
    threshold=struct.unpack('d',struct.pack('Q',limit.value))[0]
    return bits.args[0],F(threshold)


def closed_aliases(guard):
    """Recover exact selected definitions from this guard's frozen prefix.

    Selected word-closed scalars may have been inlined into a later guard.
    Equality is a whole AST expression, never a fragment or real-algebra
    identity. Original recipes still supply all numerical error bounds.
    """
    view=getattr(guard,'view',None)
    if view is None:return {}
    previous=getattr(view,'_linear_closed_aliases',None)
    if previous is not None:return previous
    result={}
    for i,text in enumerate(view.definitions):
        node=syntax(text)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Float64':
            result[ast.unparse(node)]=i
    view._linear_closed_aliases=result
    return result


def impossible_projections(registry,guards,*,max_systems=32,max_links=32):
    """Exclude only certified coupled small linear forms of one RMS vector.

    For |A*z+b|<=threshold+rounding_error, exact inversion bounds each |z|.
    A resulting squared-norm upper bound strictly below the certified RMS
    lower bound proves this whole context unreachable. Unsupported forms,
    zero floors, singular systems and exhausted proof work remain reachable.
    """
    constraints=[(g,*value) for g in guards if (value:=small_source(g)) is not None]
    checked=0;links=norm_links(registry,max_links=max_links)
    for vector in getattr(registry.model,'norm_vectors',{}).values():
        n=vector['width']
        if not 1<=n<=8 or len(vector['components'])!=n or len(constraints)<n or not vector['floor']:continue
        proof=LinearProof(registry,vector,additional_bases=links.get(id(vector)));forms=[]
        for guard,node,threshold in constraints:
            proof.closed_aliases=closed_aliases(guard);proof.memo.clear()
            own=proof.form(node)
            if own is not None:forms.append((own[0],threshold+abs(own[1])+own[2]))
        for selected in itertools.combinations(forms,n):
            checked+=1
            if checked>max_systems:return False
            inv=inverse([row[0] for row in selected])
            if inv is None:continue
            upper=[sum((abs(c)*row[1] for c,row in zip(line,selected)),F(0)) for line in inv]
            if sum((x*x for x in upper),F(0))<vector['floor']:return True
    return False


def rms_output_source_bounds(registry,proofs):
    """Invert a small whole-vector RMS norm using its original error bound.

    If stored outputs have Euclidean norm <=H, the real RMS norm is <=H+E.
    With K=(H+E)^2<n and T=sum(input_i^2), n*T/(T+n*eps)<=K implies
    T<=n*eps*K/(n-K). Only exact +/-1 gamma and certified finite Half
    sources are admitted. These are prefix-local enclosures, not numerical
    replacements for RMS or its mean. Zero signs remain unconstrained.
    """
    from direct_sympy_conversions import FiniteSource
    result={};model=registry.model;context=model.compiler.context(model.domains)
    def scalar(text):
        match=re.fullmatch(r'CompileValue([0-9]+)\(\)',text)
        return int(match[1])if match else text if text in model.domains else None
    for vector in getattr(model,'norm_vectors',{}).values():
        n=vector.get('width',0);sources=vector.get('sources',());stored=vector.get('sourceBounds',())
        components=vector.get('components',{});gamma=vector.get('gamma',());error=vector.get('roundingError')
        eps=vector.get('epsilon',0)
        if (not n or len(sources)!=n or len(stored)!=n or len(components)!=n or len(gamma)!=n
            or vector.get('context')!=context or eps<=0 or error is None or error<0
            or any(g not in (-1,1)for g in gamma)
            or any(b is None or max(abs(b.minimum),abs(b.maximum))>65504 for b in stored)):continue
        caps=[];keys=[]
        for i in range(n):
            key=scalar(components[i]);b=proofs.get(key)
            if b is None:break
            caps.append(F(max(abs(b.minimum),abs(b.maximum))))
            own=scalar(sources[i])
            if own is None or own not in proofs:break
            keys.append(own)
        if len(caps)!=n or len(keys)!=n:continue
        upper=sqrt_outward(sum((c*c for c in caps),F(0)),True)+F(error)
        k=upper*upper
        if k>=n:continue
        cap=sqrt_outward(n*F(eps)*k/(n-k),True)
        endpoint=float(cap)
        if F(endpoint)<cap:endpoint=math.nextafter(endpoint,math.inf)
        for key in keys:
            b=proofs[key];previous=result.get(key)
            low,high=-endpoint,endpoint
            if previous is not None:low,high=max(low,previous.minimum),min(high,previous.maximum)
            # Return the independent constraint, including empty intersections
            # with an old source gap. The caller detects those contradictions
            # and clips to the exact source grid without erasing zero signs.
            result[key]=FiniteSource(low,high,b.quantum)
    return result

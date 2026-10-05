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

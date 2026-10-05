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


class LinearProof:
    def __init__(self,registry,vector):
        self.registry=registry;self.vector=vector;self.n=len(vector['components'])
        self.base={text:i for i,text in vector['components'].items()}
        self.bounds=vector['magnitudes'];self.memo={};self.visiting=set()

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
        if text in self.base:
            return (tuple(F(i==self.base[text]) for i in range(self.n)),F(0),F(0))
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if not math.isfinite(node.value) or F(float(node.value))!=F(node.value):return None
            return ((F(0),)*self.n,F(node.value),F(0))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            own=self.form(node.operand);return self.scale(own,F(-1),False) if own is not None else None
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            match=re.fullmatch(r'CompileValue([0-9]+)',node.func.id)
            if match and not node.args:
                index=int(match[1])
                if index in self.memo:return self.memo[index]
                if index in self.visiting:return None
                recipe=self.registry.definition_recipes.get(index)
                if recipe is None:return None
                self.visiting.add(index);own=self.form(syntax(recipe));self.visiting.remove(index)
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
        return None


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


def impossible_projections(registry,guards,*,max_systems=32):
    """Exclude only certified coupled small linear forms of one RMS vector.

    For |A*z+b|<=threshold+rounding_error, exact inversion bounds each |z|.
    A resulting squared-norm upper bound strictly below the certified RMS
    lower bound proves this whole context unreachable. Unsupported forms,
    zero floors, singular systems and exhausted proof work remain reachable.
    """
    constraints=[value for g in guards if (value:=small_source(g)) is not None]
    checked=0
    for vector in getattr(registry.model,'norm_vectors',{}).values():
        n=vector['width']
        if not 1<=n<=8 or len(vector['components'])!=n or len(constraints)<n or not vector['floor']:continue
        proof=LinearProof(registry,vector);forms=[]
        for node,threshold in constraints:
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

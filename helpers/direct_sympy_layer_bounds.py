"""Compiler-only Half update bounds for dependency elimination.

Bounds contain no expressions and never execute in the generated artifact.
Every estimated projection is finite Half. Half products are exact F32;
their ordered reductions use a certified gamma bound rather than doubling
the absolute sum at every projection. Fraction arithmetic and outward F64
conversion keep the proof bound above the exact rational value.
"""
import math
import struct
from fractions import Fraction


def half(value):
    if value is None or not math.isfinite(value) or value<0:return None
    try:return struct.unpack('e',struct.pack('e',value))[0]
    except OverflowError:return None


def silu_bound(bound):
    """Stored-Half magnitude bound for the existing certified SiLU kernel.

    For Half |x|<=B<=1/16, the quartic's inner coefficient lies in [0,1/4].
    Ordered F64 operations therefore give |x*(.5+x*c)|<=B*(.5+B/4).
    The endpoint is an exact F64 dyadic (at most 37 significant bits).
    Apply BOTH original storage boundaries monotonically: omitting F32
    changes the bound at the smallest positive Half, among other ties.
    This is compiler-only interval arithmetic, not a change to SiLU.
    """
    if bound is None or not math.isfinite(bound) or bound<0 or half(bound)!=bound:return None
    if bound>1/16:return bound  # Existing finite-Half |silu(x)|<=|x| contract.
    upper=bound*(0.5+bound*0.25)
    return half(struct.unpack('f',struct.pack('f',upper))[0])


def dot(model,name,inputs):
    shape=model.shape(name)
    if len(shape)!=2 or shape[1]!=len(inputs) or len(inputs)>2**20 or any(
        x is None or not math.isfinite(x) or x<0 or half(x)!=x for x in inputs):return None
    result=[]
    for row in range(shape[0]):
        total=Fraction(0)
        for col in range(shape[1]):
            weight=abs(float(model.weight(name,row,col)))
            if not math.isfinite(weight) or half(weight)!=weight:return None
            total+=Fraction(inputs[col])*Fraction(weight)
        # Four zero-initialized lanes followed by three F32 additions use
        # at most n+3 rounded additions. Products and all nonzero sums are
        # multiples of 2**-48, so F32 underflow cannot occur. The largest
        # possible absolute sum (n<=2**20 finite Half products) is far below
        # F32 overflow. With u=2**-24, 1+gamma_k=1/(1-k*u).
        bound=total*Fraction(2**24,2**24-(len(inputs)+3))
        upper=float(bound)
        if Fraction(upper)<bound:upper=math.nextafter(upper,math.inf)
        # Stored Half rounding is monotone. Do not round the proof bound
        # down to F32 before applying this final storage boundary.
        result.append(half(upper))
    return None if any(x is None for x in result) else result


def dot_interval(model,name,row,inputs):
    """Signed enclosure of one original four-lane F32/Half projection.

    Inputs are certified stored-Half intervals. Weights and endpoint
    products are exact F32; only the original reduction incurs gamma.
    Use this proof before requesting producers, never as a runtime dot.
    """
    shape=model.shape(name)
    if (len(shape)!=2 or type(row)is not int or not 0<=row<shape[0]
        or not 0<len(inputs)==shape[1]<=2**20):return None
    low=high=magnitude=Fraction(0)
    for col,pair in enumerate(inputs):
        if pair is None or len(pair)!=2:return None
        a,b=pair
        if any(not math.isfinite(x) or half(abs(x))!=abs(x) for x in pair) or a>b:return None
        weight=float(model.weight(name,row,col))
        if not math.isfinite(weight) or half(abs(weight))!=abs(weight):return None
        a,b=sorted((Fraction(a)*Fraction(weight),Fraction(b)*Fraction(weight)))
        low+=a;high+=b;magnitude+=max(abs(a),abs(b))
    error=magnitude*Fraction(len(inputs)+3,2**24-len(inputs)-3)
    result=[]
    for value,up in ((low-error,False),(high+error,True)):
        rounded=float(value)
        if (Fraction(rounded)<value if up else Fraction(rounded)>value):
            rounded=math.nextafter(rounded,math.inf if up else -math.inf)
        try:result.append(struct.unpack('e',struct.pack('e',rounded))[0])
        except OverflowError:return None
    # Ordered zero-initialized F32 lanes produce +0 for an exact zero
    # reduction. Negative nonzero endpoints retain possible Half -0.
    return tuple(result)


def sqrt_outward(value,up):
    """Directed square root of an exact nonnegative proof rational."""
    result=math.sqrt(float(value))
    while (Fraction(result)**2<value if up else Fraction(result)**2>value):
        result=math.nextafter(result,math.inf if up else -math.inf)
    return Fraction(result)


def norm_linear_bound(model,name,coefficients,inputs):
    """Absolute linear-form bound on this adapter's actual Half RMS output.

    Caller certifies that inputs bound the named RMS vector, not arbitrary
    independent scalars. The same common F32 loss factor as rms_half_bound
    encloses the pre-Half vector's Euclidean norm by sqrt(n)*loss. Half
    storage contributes relative 2**-11 plus Euclidean absolute error
    sqrt(n)*2**-25. Account separately for the learned Half gamma storage.
    All roots/coefficients are compiler-only outward proof arithmetic.
    """
    from direct_sympy_checkpoint import rms_half_bound
    width=model.width
    epsilon=struct.unpack('f',struct.pack('f',model.config['rms_norm_eps']))[0]
    if len(coefficients)!=width or len(inputs)!=width or model.shape(name)!=[width] or rms_half_bound(width,epsilon) is None:return None
    weights=[float(model.weight(name,i)) for i in range(width)]
    if any(not math.isfinite(w) or half(abs(w))!=abs(w) for w in weights):return None
    u=Fraction(1,2**23);root=sqrt_outward(Fraction(width),True)
    loss=(1+u)**3/((1-u)**3*sqrt_outward(1-width*u,False))
    magnitude=root*(loss*(1+Fraction(1,2**11))+Fraction(1,2**25))
    squared=sum(((c*Fraction(w))**2 for c,w in zip(coefficients,weights)),Fraction(0))
    result=magnitude*sqrt_outward(squared,True)
    for c,w,bound in zip(coefficients,weights,inputs):
        if bound is None or not math.isfinite(bound) or bound<0 or half(bound)!=bound:return None
        if w in (-1.0,0.0,1.0):continue # These Half products store exactly.
        exponent=math.frexp(bound)[1]-1 if bound else -14
        result+=abs(c)*Fraction(2)**max(-25,exponent-11)
    return result


def norm_dot(model,name,norm_name,inputs):
    """Bound an original projection of the named actual RMS output.

    Caller attaches the named normalization to its actual producer. The
    Euclidean constraint complements independent component bounds; it
    never reassociates generated arithmetic. Include the original F32
    reduction error before monotone final Half storage.
    """
    independent=dot(model,name,inputs)
    if independent is None:return None
    shape=model.shape(name)
    if shape[1]!=model.width:return independent
    result=[]
    for row,bound in enumerate(independent):
        coefficients=[Fraction(float(model.weight(name,row,col))) for col in range(shape[1])]
        correlated=norm_linear_bound(model,norm_name,coefficients,inputs)
        if correlated is None:return independent
        total=sum((abs(c)*Fraction(x) for c,x in zip(coefficients,inputs)),Fraction(0))
        upper=correlated+total*Fraction(shape[1]+3,2**24-shape[1]-3)
        value=float(upper)
        if Fraction(value)<upper:value=math.nextafter(value,math.inf)
        rounded=half(value)
        result.append(min(bound,rounded) if rounded is not None else bound)
    return result


def composed_dot(model,first_name,second_name,inputs,mapping,*,max_products=2**20,input_norm=None):
    """Enclose two ordered stored-Half projections without reassociating them.

    Compose weights only in exact proof arithmetic. The actual first
    projection remains rounded F32 then Half, and the actual second
    projection retains its four-lane reduction and final Half boundary.
    The first projection's reduction/storage error and the second's
    reduction error are added to the composed exact sum before final
    monotone Half storage. This exposes coefficient cancellations which
    independent absolute sums lose. Large proof workloads retain dot().
    """
    a,b=model.shape(first_name),model.shape(second_name)
    if (len(a)!=2 or len(b)!=2 or any(type(n)is not int or n<1 for n in a+b)
        or a[1]!=len(inputs) or b[1]!=len(mapping) or a[1]>2**20 or b[1]>2**20
        or any(type(i)is not int or not 0<=i<a[0] for i in mapping)
        or type(max_products)is not int or max_products<1):return None
    if (2*a[0]+b[0]*b[1])*a[1]+2*b[0]*b[1]>max_products:return None
    first=dot(model,first_name,inputs)
    if first is None:return None
    old=dot(model,second_name,[first[i] for i in mapping])
    if old is None:return None
    gamma=lambda n:Fraction(n+3,2**24-n-3)
    errors=[]
    for row,stored_bound in enumerate(first):
        total=sum((Fraction(inputs[col])*abs(Fraction(float(model.weight(first_name,row,col)))) for col in range(a[1])),Fraction(0))
        # A rounded magnitude bound cannot lie below the binade of its
        # unrounded upper endpoint. Include the fixed subnormal half-ULP
        # even if that endpoint rounded to zero; powers of two use the
        # larger outgoing spacing, which is conservative on both sides.
        exponent=math.frexp(stored_bound)[1]-1 if stored_bound else -14
        storage=Fraction(2)**max(-25,exponent-11)
        errors.append(gamma(a[1])*total+storage)
    result=[]
    for row in range(b[0]):
        weights=[Fraction(float(model.weight(second_name,row,j))) for j in range(b[1])]
        total=Fraction(0);coefficients=[]
        for col in range(a[1]):
            coefficient=sum((weights[j]*Fraction(float(model.weight(first_name,mapping[j],col))) for j in range(b[1])),Fraction(0))
            coefficients.append(coefficient)
            total+=Fraction(inputs[col])*abs(coefficient)
        if input_norm is not None:
            correlated=norm_linear_bound(model,input_norm,coefficients,inputs)
            if correlated is not None:total=min(total,correlated)
        total+=sum((abs(w)*errors[mapping[j]] for j,w in enumerate(weights)),Fraction(0))
        total+=gamma(b[1])*sum((abs(w)*Fraction(first[mapping[j]]) for j,w in enumerate(weights)),Fraction(0))
        upper=float(total)
        if Fraction(upper)<total:upper=math.nextafter(upper,math.inf)
        stored=half(upper)
        result.append(old[row] if stored is None else min(old[row],stored))
    return result


def composed_dot_intervals(model,first_name,second_name,intervals,mapping,*,max_products=2**20):
    """Signed bounds for two original rounded projections, proof only.

    Compose coefficients in exact arithmetic, retaining both reductions
    and first Half storage. Apply final Half storage monotonically to
    outward endpoints. No numerical computation is reassociated.
    """
    a,b=model.shape(first_name),model.shape(second_name)
    if (len(a)!=2 or len(b)!=2 or any(type(n)is not int or n<1 for n in a+b)
        or a[1]!=len(intervals) or b[1]!=len(mapping)
        or type(max_products)is not int or max_products<1
        or (2*a[0]+b[0]*b[1])*a[1]+2*b[0]*b[1]>max_products
        or any(type(i)is not int or not 0<=i<a[0] for i in mapping)):return None
    if any(pair is None or len(pair)!=2 or pair[0]>pair[1] or any(
        not math.isfinite(x) or half(abs(x))!=abs(x) for x in pair) for pair in intervals):return None
    inputs=[max(abs(x),abs(y)) for x,y in intervals]
    first=dot(model,first_name,inputs)
    if first is None or dot(model,second_name,[first[i] for i in mapping]) is None:return None
    gamma=lambda n:Fraction(n+3,2**24-n-3)
    errors=[]
    for row,bound in enumerate(first):
        total=sum((Fraction(inputs[col])*abs(Fraction(float(model.weight(first_name,row,col)))) for col in range(a[1])),Fraction(0))
        exponent=math.frexp(bound)[1]-1 if bound else -14
        errors.append(gamma(a[1])*total+Fraction(2)**max(-25,exponent-11))
    result=[]
    for row in range(b[0]):
        weights=[Fraction(float(model.weight(second_name,row,j))) for j in range(b[1])]
        low=high=Fraction(0)
        for col,(x,y) in enumerate(intervals):
            coefficient=sum((weights[j]*Fraction(float(model.weight(first_name,mapping[j],col))) for j in range(b[1])),Fraction(0))
            lo,hi=sorted((coefficient*Fraction(x),coefficient*Fraction(y)));low+=lo;high+=hi
        error=sum((abs(w)*errors[mapping[j]] for j,w in enumerate(weights)),Fraction(0))
        error+=gamma(b[1])*sum((abs(w)*Fraction(first[mapping[j]]) for j,w in enumerate(weights)),Fraction(0))
        endpoints=[]
        for value,up in ((low-error,False),(high+error,True)):
            rounded=float(value)
            if (Fraction(rounded)<value if up else Fraction(rounded)>value):rounded=math.nextafter(rounded,math.inf if up else -math.inf)
            try:endpoints.append(struct.unpack('e',struct.pack('e',rounded))[0])
            except OverflowError:return None
        result.append(tuple(endpoints))
    return result


def norm(model,name):
    from direct_sympy_checkpoint import rms_half_bound
    bound=rms_half_bound(model.width,struct.unpack('f',struct.pack('f',model.config['rms_norm_eps']))[0])
    base=half(bound)
    if base is None:return None
    return [half(base*abs(float(model.weight(name,i)))) for i in range(model.width)]


def layer(model,prefix):
    if model.config.get('attention_bias') or model.config.get('mlp_bias') or model.config.get('hidden_act','silu') not in ('silu','swish'):return None
    pre=norm(model,prefix+'input_layernorm.weight')
    post=norm(model,prefix+'post_attention_layernorm.weight')
    if pre is None or post is None or any(x is None for x in pre+post):return None
    q=dot(model,prefix+'self_attn.q_proj.weight',pre)
    k=dot(model,prefix+'self_attn.k_proj.weight',pre)
    v=dot(model,prefix+'self_attn.v_proj.weight',pre)
    if q is None or k is None or v is None:return None
    heads=model.config['num_attention_heads'];kv_heads=model.config['num_key_value_heads']
    dimension=model.config.get('head_dim',model.width//heads)
    if heads<=0 or kv_heads<=0 or not dimension or model.width!=heads*dimension or heads%kv_heads or len(q)!=heads*dimension or len(k)!=kv_heads*dimension or len(v)!=len(k):return None
    # A single causal key has probability one only for finite scores.
    for h in range(heads):
        kv=h//(heads//kv_heads)
        score=half(2*math.fsum(q[h*dimension+i]*k[kv*dimension+i] for i in range(dimension)))
        if score is None:return None
    context=[v[(i//dimension//(heads//kv_heads))*dimension+i%dimension] for i in range(model.width)]
    attention=dot(model,prefix+'self_attn.o_proj.weight',context)
    # At position zero a single finite causal key has probability one.
    # Mapping retains grouped-query replication in the proof composition.
    mapping=[(i//dimension//(heads//kv_heads))*dimension+i%dimension for i in range(model.width)]
    composed=composed_dot(model,prefix+'self_attn.v_proj.weight',prefix+'self_attn.o_proj.weight',pre,mapping,
        input_norm=prefix+'input_layernorm.weight')
    if composed is not None:attention=composed
    gate=norm_dot(model,prefix+'mlp.gate_proj.weight',prefix+'post_attention_layernorm.weight',post)
    up=norm_dot(model,prefix+'mlp.up_proj.weight',prefix+'post_attention_layernorm.weight',post)
    if gate is None or up is None or len(gate)!=len(up):return None
    # Tighten the activation before expanding its producer, using the same
    # certified Half polynomial as the expression compiler where admitted.
    # Both operands are stored Half. Their product is exact F32; the final
    # Half conversion is monotone, with no extra factor-of-two error bound.
    gated=[half(silu_bound(a)*b) for a,b in zip(gate,up)]
    mlp=dot(model,prefix+'mlp.down_proj.weight',gated)
    return {'attention':attention,'mlp':mlp} if attention is not None and mlp is not None else None

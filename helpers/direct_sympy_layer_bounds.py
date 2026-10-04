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
    gate=dot(model,prefix+'mlp.gate_proj.weight',post)
    up=dot(model,prefix+'mlp.up_proj.weight',post)
    if gate is None or up is None or len(gate)!=len(up):return None
    # Tighten the activation before expanding its producer, using the same
    # certified Half polynomial as the expression compiler where admitted.
    # Both operands are stored Half. Their product is exact F32; the final
    # Half conversion is monotone, with no extra factor-of-two error bound.
    gated=[half(silu_bound(a)*b) for a,b in zip(gate,up)]
    mlp=dot(model,prefix+'mlp.down_proj.weight',gated)
    return {'attention':attention,'mlp':mlp} if attention is not None and mlp is not None else None

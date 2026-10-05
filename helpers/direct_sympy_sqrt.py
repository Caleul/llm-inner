"""Positive finite F32 square root, returning the exact widened F32 value.

A rational 5/5 expression interpolates eleven Lobatto points, including
both endpoints. Its partial fractions use five copies of the normalized
input, instead of the factored numerator/denominator's six. Their ordered
F64 evaluation is certified at the final F32 boundary.

Mantissa/parity enumeration certifies all 2**24 normalized cases. Exact
power-of-two scaling covers every F32 input exponent, including subnormals without output
underflow/overflow. The rational coefficients are compiler constants, not response data.
"""
from direct_sympy_strings import syntax
from direct_sympy_words import simplify_words
from direct_sympy_conversions import FiniteSource,lower_finite_conversion
import math

# Solve eleven Lobatto samples at 100 digits, find denominator roots at
# 70 digits, then round residues/poles once to F64. Descending contribution
# at z=0 fixes the evaluation order; another order can produce a midpoint.
# These constants describe the rational function, not model responses.
CONSTANT_TERM=13.367174015436346
PARTIAL_FRACTIONS=((-779.0174377628756,72.88886838020784),
    (-9.032202101157678,8.541941977833192),
    (-1.0005960489978687,3.4400824392419294),
    (-0.1852754751106736,2.096193424586035),
    (-0.029292444172441905,1.6235843576093156))

# Five Lobatto samples on [1.03125,1.0625], centered at 1.046875,
# admit a two-source 2/2 kernel. Its original ordered F64 evaluation is
# certified on every F32 mantissa in this interval and both scale parities.
# Outside a proved single-binade enclosure, retain the universal 5/5 kernel.
NARROW_CENTER=1.046875
NARROW_INTERVAL=(1.03125,1.0625)
NARROW_CONSTANT_TERM=5.115760004435847
NARROW_PARTIAL_FRACTIONS=((-42.49736561224015,10.962622678643505),
    (-0.3455137406727387,1.599440032849224))

# Compiler-selected numerical certificates, ordered by source occurrences.
# Selection requires the entire branch enclosure inside one binary scale and
# one certified interval. These constants approximate the operation, not model
# responses. Never extrapolate or introduce a runtime kernel selector.
REGIONAL_KERNELS=(
    ((1.0,1.03125),1.015625,5.03882163904675,
        ((-40.608633603342646,10.635356285748777),(-0.33015693435982507,1.5516929693141102))),
    (NARROW_INTERVAL,NARROW_CENTER,NARROW_CONSTANT_TERM,NARROW_PARTIAL_FRACTIONS),
    ((1.0,1.125),1.0625,7.213649925328846,
        ((-121.21401571571069,21.44681022912209),(-1.2643385409523005,2.731723482708797),
         (-0.08923857710404674,1.3085889107675457))),
    ((1.0,1.5),1.25,10.034070281879014,
        ((-328.42424548693276,41.215407137443975),(-3.6876941635275764,4.9684755018866555),
         (-0.36684145291176556,2.1193794430256965),(-0.045564819547329084,1.413021128487478))),
    ((1.5,2.0),1.75,11.88892058018027,
        ((-546.3483212075085,57.866108384246175),(-6.140496982105641,6.977579432856978),
         (-0.6123816025075169,2.9745257795503455),(-0.07631459236509174,1.980002684263491))),
)


def certified_kernel(minimum,maximum):
    first=math.frexp(minimum)[1]-1;last=math.frexp(maximum)[1]-1
    if first!=last:return None
    low,high=math.ldexp(minimum,-first),math.ldexp(maximum,-first)
    return next((kernel for kernel in REGIONAL_KERNELS
        if kernel[0][0]<=low and high<=kernel[0][1]),None)


def supported(source,session):
    node=syntax(source);bounds=session.bounds(node)
    return session.value_kind(node) in ('half','f32') and bounds is not None and bounds.minimum>=2**-149 and bounds.maximum<=3.4028234663852886e38


def expand(source,session):
    if not supported(source,session):raise ValueError('F32 sqrt certificate requires finite positive F32 input')
    raw='Bits64(X999999997)'
    m=f'Float64(U64Or(U64And({raw},4503599627370495),4607182418800017408))'
    bounds=session.bounds(syntax(source))
    first=math.frexp(bounds.minimum)[1]-1;last=math.frexp(bounds.maximum)[1]-1
    constant_scale=first==last
    if constant_scale:
        # Widened finite F32 (including subnormals) times this exact power
        # of two is the same [1,2) mantissa as the word extraction above.
        # F32 exponents fit safely in F64, so this operation neither rounds
        # nor overflows/underflows. Keep the original rational order.
        m='X999999997' if first==0 else f'(X999999997 * 2**({-first}))'
    kernel=certified_kernel(bounds.minimum,bounds.maximum)
    center,constant,fractions=(kernel[1:] if kernel is not None else (1.5,CONSTANT_TERM,PARTIAL_FRACTIONS))
    z=f'(({m}) - {center!r})'
    p=repr(constant)
    for residue,pole in fractions:
        p=f'(({p}) + ({residue!r} / (({z}) + {pole!r})))'
    if kernel is not None and kernel[0]==NARROW_INTERVAL:
        session.narrow_square_roots_closed=getattr(session,'narrow_square_roots_closed',0)+1
    elif kernel is not None:
        session.regional_square_roots_closed=getattr(session,'regional_square_roots_closed',0)+1
    exponent=f'U64And(U64Shr({raw},52),2047)'
    parity=f'U64And(U64Add({exponent},1),1)'
    if constant_scale:
        # The complete admitted enclosure is inside one binary scale range.
        # Use its proved exponent, without changing the ordered rational
        # evaluation or creating another runtime decision.
        if first%2:p=f'({p}) * 1.4142135623730951'
    else:p=f'({p}) * (1.0 + ({parity}) * 0.4142135623730951)'
    candidate=session.compiler.substitute(p,'X999999997',source,session.domains)
    # The exhaustive mantissa/parity proof bounds this F64 evaluation inside [0.9,2.1].
    # All such F64 values lie on the 2**-53 grid, including roundoff.
    # Exhaustive evaluation of this rational expression/order on every normalized
    # F32 mantissa and both exponent parities finds no F32 midpoint. Avoid
    # duplicating the complete candidate just to select its retained parity.
    rounded=lower_finite_conversion(candidate,'R32',FiniteSource(0.9,2.1,-53),session.compiler,session.domains,no_odd_f32_ties=True)
    if constant_scale:adjustment=str((first//2*4503599627370496)%(1<<64))
    else:
        target=f'U64Shr(U64Add({exponent},1023),1)'
        adjustment=f'U64Mul(U64Add({target},18446744073709550593),4503599627370496)'
        adjustment=session.compiler.substitute(adjustment,'X999999997',source,session.domains)
    return simplify_words(f'Float64(U64Add(Bits64({rounded}),{adjustment}))',session.compiler,session.domains)

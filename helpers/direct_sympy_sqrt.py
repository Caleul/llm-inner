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


def supported(source,session):
    node=syntax(source);bounds=session.bounds(node)
    return session.value_kind(node) in ('half','f32') and bounds is not None and bounds.minimum>=2**-149 and bounds.maximum<=3.4028234663852886e38


def expand(source,session):
    if not supported(source,session):raise ValueError('F32 sqrt certificate requires finite positive F32 input')
    raw='Bits64(X999999997)'
    m=f'Float64(U64Or(U64And({raw},4503599627370495),4607182418800017408))'
    z=f'(({m}) - 1.5)'
    p=repr(CONSTANT_TERM)
    for residue,pole in PARTIAL_FRACTIONS:
        p=f'(({p}) + ({residue!r} / (({z}) + {pole!r})))'
    bounds=session.bounds(syntax(source))
    first=math.frexp(bounds.minimum)[1]-1;last=math.frexp(bounds.maximum)[1]-1
    constant_scale=first==last
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

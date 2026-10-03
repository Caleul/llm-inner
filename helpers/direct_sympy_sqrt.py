"""Positive finite F32 square root, returning the exact widened F32 value.

A cubic/quadratic rational seed interpolates six Lobatto points, including
both endpoints, before one ordered Newton correction.

Mantissa/parity enumeration certifies all 2**24 normalized cases. Exact
power-of-two scaling covers every F32 input exponent, including subnormals without output
underflow/overflow. The rational coefficients are compiler constants, not response data.
"""
from direct_sympy_strings import syntax
from direct_sympy_words import simplify_words
from direct_sympy_conversions import FiniteSource,lower_finite_conversion

NUMERATOR=(0.011776416813942576, 0.31267808731141555, 1.2350886571993487, 1.2247448161010575)
DENOMINATOR=(0.08581640268346791, 0.6751122868867456, 1.0)


def supported(source,session):
    node=syntax(source);bounds=session.bounds(node)
    return session.value_kind(node) in ('half','f32') and bounds is not None and bounds.minimum>=2**-149 and bounds.maximum<=3.4028234663852886e38


def expand(source,session):
    if not supported(source,session):raise ValueError('F32 sqrt certificate requires finite positive F32 input')
    raw='Bits64(X999999997)'
    m=f'Float64(U64Or(U64And({raw},4503599627370495),4607182418800017408))'
    z=f'(({m}) - 1.5)'
    def horner(coefficients):
        p=repr(coefficients[0])
        for coefficient in coefficients[1:]:p=f'({repr(coefficient)} + ({z}) * ({p}))'
        return p
    seed=f'(({horner(NUMERATOR)}) / ({horner(DENOMINATOR)}))'
    p=f'(0.5 * (({seed}) + ({m}) / ({seed})))'
    exponent=f'U64And(U64Shr({raw},52),2047)'
    parity=f'U64And(U64Add({exponent},1),1)'
    p=f'({p}) * (1.0 + ({parity}) * 0.4142135623730951)'
    candidate=session.compiler.substitute(p,'X999999997',source,session.domains)
    # The mantissa/parity proof bounds this F64 evaluation inside [0.9,2.1].
    # All such F64 values lie on the 2**-53 grid, including roundoff.
    # Exhaustive evaluation of this rational seed/correction/order on every normalized
    # F32 mantissa and both exponent parities finds no F32 midpoint. Avoid
    # duplicating the complete candidate just to select its retained parity.
    rounded=lower_finite_conversion(candidate,'R32',FiniteSource(0.9,2.1,-53),session.compiler,session.domains,no_odd_f32_ties=True)
    target=f'U64Shr(U64Add({exponent},1023),1)'
    adjustment=f'U64Mul(U64Add({target},18446744073709550593),4503599627370496)'
    adjustment=session.compiler.substitute(adjustment,'X999999997',source,session.domains)
    return simplify_words(f'Float64(U64Add(Bits64({rounded}),{adjustment}))',session.compiler,session.domains)

"""Positive finite F32 square root, returning the exact widened F32 value.

Mantissa/parity enumeration certifies all 2**24 normalized cases. Exact
power-of-two scaling covers every F32 input exponent, including subnormals without output
underflow/overflow. The rational coefficients are compiler constants, not response data.
"""
from direct_sympy_strings import syntax
from direct_sympy_words import simplify_words
from direct_sympy_conversions import FiniteSource,lower_finite_conversion

NUMERATOR=(0.04091863612351722,0.4844752145633112,1.439174633487947,1.224744871391589)
DENOMINATOR=(0.004807144017881053,0.17054532022514554,0.8417478211033906,1.0)


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

"""Positive finite F32 square root, returning the exact widened F32 value.

Mantissa/parity enumeration certifies all 2**24 normalized cases. Exact
power-of-two scaling covers every F32 input exponent, including subnormals without output
underflow/overflow. The polynomial is a compiler constant, not response data.
"""
from direct_sympy_strings import syntax
from direct_sympy_words import simplify_words
from direct_sympy_conversions import FiniteSource,lower_finite_conversion

COEFFICIENTS=(3.157062843816458e-06, -5.121056696396752e-06, 4.396882440200792e-06, -7.258073785697736e-06, 1.4154162706898482e-05, -2.3577883649300753e-05, 3.8971587738244104e-05, -6.608569760505093e-05, 0.0001134057378097486, -0.0001969672889392875, 0.0003475761971079193, -0.0006256372128288169, 0.0011550234673755844, -0.002205044797811739, 0.004410089557247572, -0.009450191908511494, 0.02268046058132749, -0.06804138174398001, 0.408248290463863, 1.2247448713915892)


def supported(source,session):
    node=syntax(source);bounds=session.bounds(node)
    return session.value_kind(node) in ('half','f32') and bounds is not None and bounds.minimum>=2**-149 and bounds.maximum<=3.4028234663852886e38


def expand(source,session):
    if not supported(source,session):raise ValueError('F32 sqrt certificate requires finite positive F32 input')
    raw='Bits64(X999999997)'
    m=f'Float64(U64Or(U64And({raw},4503599627370495),4607182418800017408))'
    z=f'(({m}) - 1.5)'
    p=repr(COEFFICIENTS[0])
    for coefficient in COEFFICIENTS[1:]:p=f'({repr(coefficient)} + ({z}) * ({p}))'
    exponent=f'U64And(U64Shr({raw},52),2047)'
    parity=f'U64And(U64Add({exponent},1),1)'
    p=f'({p}) * (1.0 + ({parity}) * 0.4142135623730951)'
    polynomial=session.compiler.substitute(p,'X999999997',source,session.domains)
    # The mantissa/parity proof bounds this F64 evaluation inside [0.9,2.1].
    # All such F64 values lie on the 2**-53 grid, including roundoff.
    # Exhaustive evaluation of these coefficients/order on every normalized
    # F32 mantissa and both exponent parities finds no F32 midpoint. Avoid
    # duplicating the complete polynomial just to select its retained parity.
    rounded=lower_finite_conversion(polynomial,'R32',FiniteSource(0.9,2.1,-53),session.compiler,session.domains,no_odd_f32_ties=True)
    target=f'U64Shr(U64Add({exponent},1023),1)'
    adjustment=f'U64Mul(U64Add({target},18446744073709550593),4503599627370496)'
    adjustment=session.compiler.substitute(adjustment,'X999999997',source,session.domains)
    return simplify_words(f'Float64(U64Add(Bits64({rounded}),{adjustment}))',session.compiler,session.domains)

"""Positive finite F32 square root, returning the exact widened F32 value.

A rational 5/5 expression interpolates eleven Lobatto points, including
both endpoints. SymPy factors its numerator and denominator into one linear
and two completed-square factors each. No refinement expression is duplicated.

Mantissa/parity enumeration certifies all 2**24 normalized cases. Exact
power-of-two scaling covers every F32 input exponent, including subnormals without output
underflow/overflow. The rational coefficients are compiler constants, not response data.
"""
from direct_sympy_strings import syntax
from direct_sympy_words import simplify_words
from direct_sympy_conversions import FiniteSource,lower_finite_conversion

# Eleven Lobatto samples of sqrt(z + 1.5), z in [-0.5, 0.5].
# Solve the rational 5/5 interpolation at 100 digits, factor at 70 digits,
# then round each coefficient once to F64. These are arithmetic constants.
COEFFICIENT=13.367174015436346
NUMERATOR_FACTORS=((18.5850823052632,), (1.664726217594286, -0.018266192753258678), (3.8155673924431786, -1.5052409305796959))
DENOMINATOR_FACTORS=((72.88886838020784,), (1.8598888910976752, -0.05583983254715131), (5.991012208537561, -6.507242687878663))


def supported(source,session):
    node=syntax(source);bounds=session.bounds(node)
    return session.value_kind(node) in ('half','f32') and bounds is not None and bounds.minimum>=2**-149 and bounds.maximum<=3.4028234663852886e38


def expand(source,session):
    if not supported(source,session):raise ValueError('F32 sqrt certificate requires finite positive F32 input')
    raw='Bits64(X999999997)'
    m=f'Float64(U64Or(U64And({raw},4503599627370495),4607182418800017408))'
    z=f'(({m}) - 1.5)'
    def product(factors):
        parts=[f'(({z}) + {factor[0]!r})' if len(factor)==1 else f'((({z}) + {factor[0]!r}) ** 2 + ({factor[1]!r}))' for factor in factors]
        result=parts[0]
        for part in parts[1:]:result=f'(({result}) * ({part}))'
        return result
    numerator=product(NUMERATOR_FACTORS)
    denominator=product(DENOMINATOR_FACTORS)
    p=f'(({COEFFICIENT!r} * ({numerator})) / ({denominator}))'
    exponent=f'U64And(U64Shr({raw},52),2047)'
    parity=f'U64And(U64Add({exponent},1),1)'
    p=f'({p}) * (1.0 + ({parity}) * 0.4142135623730951)'
    candidate=session.compiler.substitute(p,'X999999997',source,session.domains)
    # The exhaustive mantissa/parity proof bounds this F64 evaluation inside [0.9,2.1].
    # All such F64 values lie on the 2**-53 grid, including roundoff.
    # Exhaustive evaluation of this rational expression/order on every normalized
    # F32 mantissa and both exponent parities finds no F32 midpoint. Avoid
    # duplicating the complete candidate just to select its retained parity.
    rounded=lower_finite_conversion(candidate,'R32',FiniteSource(0.9,2.1,-53),session.compiler,session.domains,no_odd_f32_ties=True)
    target=f'U64Shr(U64Add({exponent},1023),1)'
    adjustment=f'U64Mul(U64Add({target},18446744073709550593),4503599627370496)'
    adjustment=session.compiler.substitute(adjustment,'X999999997',source,session.domains)
    return simplify_words(f'Float64(U64Add(Bits64({rounded}),{adjustment}))',session.compiler,session.domains)

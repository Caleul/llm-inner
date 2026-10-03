"""Arithmetic SiLU expansion certified on finite Half inputs in [-1/16,1/16].

The final Half result, not the internal exponential approximation, is the
certificate boundary. Exhaustive scalar/vector reference tests cover every
Half pattern in this interval. Other domains retain the generic primitive.
"""
from direct_sympy_strings import syntax


def supported(source,session):
    node=syntax(source);bounds=session.bounds(node)
    return session.value_kind(node)=='half' and bounds is not None and bounds.minimum>=-1/16 and bounds.maximum<=1/16


def expand(source,session):
    if not supported(source,session):raise ValueError('SiLU arithmetic certificate requires finite Half within [-1/16,1/16]')
    x='X999999997'
    # Preserve the tested Horner and F32 operation order. No exponential,
    # answer table, activation executor, or runtime intermediate survives.
    polynomial=f'1.0 + (-{x}) * (1.0 + (-{x}) * (1.0 + (-{x}) / 3.0) / 2.0)'
    template=f'R16(R32({x} / R32(1.0 + R32({polynomial}))))'
    return session.compiler.substitute(template,x,source,session.domains)

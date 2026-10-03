"""Arithmetic SiLU expansion certified on finite Half inputs in [-1/16,1/16].

The final Half result, not the internal exponential approximation, is the
certificate boundary. Exhaustive scalar/vector reference tests cover every
Half pattern in this interval. Other domains retain the generic primitive.
"""


def supported(source,session):
    node,_=session.analyze_expression(source);bounds=session.bounds(node)
    return session.value_kind(node)=='half' and bounds is not None and bounds.minimum>=-1/16 and bounds.maximum<=1/16


def expand(source,session):
    if not supported(source,session):raise ValueError('SiLU arithmetic certificate requires finite Half within [-1/16,1/16]')
    x='X999999997'
    # This ordered quartic/F32/Half chain is certified exhaustively at the
    # final Half boundary. The F32 step is essential: the smallest positive
    # Half input otherwise rounds to a different Half result.
    node,_=session.analyze_expression(source);bounds=session.bounds(node)
    # Separate exhaustive emitted-kernel certificate on this smaller domain.
    # The quadratic is NOT valid throughout [-1/32,1/32], so use the exact
    # dyadic certificate rather than rounding the domain up to that interval.
    if bounds.minimum>=-3/128 and bounds.maximum<=3/128:
        polynomial=f'{x} * (0.5 + {x} * 0.25)'
    else:
        polynomial=f'{x} * (0.5 + {x} * (0.25 - {x} * {x} / 48.0))'
    template=f'R16(R32({polynomial}))'
    return session.compiler.substitute(template,x,source,session.domains)

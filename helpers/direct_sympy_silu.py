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


def quadratic_subnormal_guard(node,session):
    """Exact preimage of the F32 normal threshold for the Half quadratic.

    On Half x in [-3/128,3/128], x*(0.5+x*0.25) is increasing.
    Positive x reaches the threshold at 2**-13; negative x remains below
    it at -2**-13, and crosses at the next negative Half. The shifted
    square tests exactly this discrete interval with one occurrence of x.
    Its addition and square are exact F64 operations on this Half domain.
    """
    import ast
    if not isinstance(node,ast.BinOp) or not isinstance(node.op,ast.Mult):return None
    x=node.left;factor=node.right
    if not isinstance(factor,ast.BinOp) or not isinstance(factor.op,ast.Add) or session.constant(factor.left)!=0.5:return None
    term=factor.right
    if not isinstance(term,ast.BinOp) or not isinstance(term.op,ast.Mult) or session.constant(term.right)!=0.25 or session.key(term.left)!=session.key(x):return None
    bounds=session.bounds(x)
    if session.value_kind(x)!='half' or bounds is None or bounds.minimum < -3/128 or bounds.maximum > 3/128:return None
    return '('+ast.unparse(x)+' + 2**-25)**2 < 2**-26'

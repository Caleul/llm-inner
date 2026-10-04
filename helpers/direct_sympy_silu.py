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


def quadratic_source(node,session):
    """Recognize only the certified Half quadratic, not real-algebra patterns."""
    import ast
    # The exact-arithmetic factor pass can move the dyadic denominator
    # outside the product. Recognize those ordered forms too; each step
    # remains exact on the admitted Half interval and preserves x's zero
    # sign. Do not recognize an arbitrary real polynomial with different
    # zero signs or unproved intermediate rounding.
    divided=isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and session.constant(node.right)==4
    if divided:node=node.left
    if not isinstance(node,ast.BinOp) or not isinstance(node.op,ast.Mult):return None
    x=node.left;factor=node.right
    if divided:
        if not isinstance(factor,ast.BinOp) or not isinstance(factor.op,ast.Add):return None
        if session.key(factor.left)!=session.key(x) or session.constant(factor.right)!=2:return None
    else:
        if isinstance(factor,ast.BinOp) and isinstance(factor.op,ast.Div) and session.constant(factor.right)==4:
            term=factor.left
            if not isinstance(term,ast.BinOp) or not isinstance(term.op,ast.Add) or session.key(term.left)!=session.key(x) or session.constant(term.right)!=2:return None
        else:
            if not isinstance(factor,ast.BinOp) or not isinstance(factor.op,ast.Add) or session.constant(factor.left)!=0.5:return None
            term=factor.right
            scaled=(isinstance(term,ast.BinOp) and
                ((isinstance(term.op,ast.Mult) and session.constant(term.right)==0.25) or
                 (isinstance(term.op,ast.Div) and session.constant(term.right)==4)))
            if not scaled or session.key(term.left)!=session.key(x):return None
    bounds=session.bounds(x)
    if session.value_kind(x)!='half' or bounds is None or bounds.minimum < -3/128 or bounds.maximum > 3/128:return None
    return x


def quadratic_subnormal_guard(node,session):
    """Exact preimage of the F32 normal threshold for the Half quadratic.

    On Half x in [-3/128,3/128], x*(0.5+x*0.25) is increasing.
    Positive x reaches the threshold at 2**-13; negative x remains below
    it at -2**-13, and crosses at the next negative Half. The shifted
    square tests exactly this discrete interval with one occurrence of x.
    Its addition and square are exact F64 operations on this Half domain.
    """
    import ast
    x=quadratic_source(node,session)
    if x is None:return None
    return '('+ast.unparse(x)+' + 2**-25)**2 < 2**-26'


def quadratic_tandem_source(node,session):
    """One-occurrence magnitude, certified only at the tandem Half boundary.

    For Half |x| <= 3/128, (x+1)^2 and the subtraction are exact F64
    dyadics, equal to x*(.5+x*.25). The nonzero product has <=37
    significant bits (11-bit x times a <=26-bit factor), admitting the
    exact integer-word tandem route. At -0 this polynomial becomes +0:
    the caller MUST preserve the original x sign in the subnormal arm.
    This is not a generic raw-expression or same-sign rewrite.
    """
    import ast
    x=quadratic_source(node,session)
    if x is None:return None
    source=ast.unparse(x)
    return '((('+source+') + 1.0)**2 - 1.0) * 0.25',source

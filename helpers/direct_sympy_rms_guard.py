"""Exact Half dispatch for the ordered F32 RMS inverse.

For positive finite F32 s, sqrt(s), its reciprocal, and their rounded
results are normal F32. Their relative error and the final F32 product's
error together are less than 2**-22. A nonzero finite Half x times this
inverse cannot underflow F32 (its magnitude is at least 2**-89).

The exact dyadic test x*x < s*2**-28 selects the side of 2**-14 for
x/sqrt(s). A disagreement with the rounded product can only occur within
2**-36 of 2**-14. Both Half magnitude kernels agree on that interval:
the normal kernel rounds to 2**-14 from 2**-14-2**-26 upward, while the
fixed-subnormal-grid kernel remains valid up to the next binade 2**-13.
Thus the dispatch can omit the inverse without changing ANY output bit.
The evaluated normalization, dtype and numerical order stay unchanged.

The caller owns the ordered inverse provenance R32(1/R32(sqrt(s))).
This is not a general rule for an arbitrary positive factor.
"""


def bindings(input_value,mean_value,inverse_value,session):
    nodes=[session.analyze_expression(value)[0] for value in (input_value,mean_value,inverse_value)]
    x,mean,inverse=nodes
    if session.value_kind(x)!='half' or session.bounds(x) is None:return None
    for node in (mean,inverse):
        bounds=session.bounds(node)
        if session.value_kind(node)!='f32' or bounds is None or bounds.minimum<=0:return None
    # Fundamental Half inputs or immutable completed literals can be bound
    # without rebuilding their descendants. Unknown provenance is a barrier.
    if input_value not in session.domains and input_value not in session.closed_literals:return None
    if mean_value not in session.closed_literals or inverse_value not in session.closed_literals:return None
    return ('X999999995 ** 2 < X999999994 * 2**(-28)',
            {'X999999995':input_value,'X999999994':mean_value})

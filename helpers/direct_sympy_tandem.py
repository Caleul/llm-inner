"""Certified F64 -> F32 -> F16 composition using elementary word operations.

F32 must round a finite normal-or-zero source, without overflow. No real
arithmetic reassociation is used. Other sources retain the generic route.
"""
from direct_sympy_words import simplify_words
from direct_sympy_strings import syntax


def supported(certificate):
    return certificate is not None and certificate.quantum is not None and certificate.quantum>=-126 and max(abs(certificate.minimum),abs(certificate.maximum))<2**128-2**103


def lower_tandem(source,certificate,compiler,domains,*,integer_word_exact=False,sign_expression=None,small_condition=None,frontier=None,facts=()):
    if not supported(certificate):raise ValueError("Tandem conversion requires finite normal-or-zero F32 source")
    source=compiler.stabilize("("+source+")",domains,facts=facts)
    raw="Bits64(X999999997)"
    mag=f"U64And({raw}, 9223372036854775807)"
    # Caller may certify a smaller expression with the identical IEEE sign,
    # including both zeros. Only the sign field uses it; magnitude, rounding
    # and every branch threshold still use the original producer.
    sign_raw=raw if sign_expression is None else "Bits64(X999999996)"
    sign=f"U64And({sign_raw}, 9223372036854775808)"
    # At a Half midpoint the F32 significand is even. Its whole closed
    # F64 tie cell [mid-2^28 ulps, mid+2^28 ulps] maps to that midpoint.
    # Even Half lower endpoints therefore switch above mid+2^28; odd ones
    # switch at mid-2^28. This word bias encodes both, including double ties.
    bias=(1<<41)-1-(1<<28)
    extra=(1<<29)+1
    parity=f"U64And(U64Shr({raw}, 42), 1)"
    normal=f"Float64(U64And(U64Add({raw}, U64Add({bias}, U64Mul({parity}, {extra}))), 18446739675663040512))"
    odd32=f"U64And(U64Shr({mag}, 29), 1)"
    rounded32=f"U64And(U64Add({mag}, U64Add(268435455, {odd32})), 18446744073172680704)"
    if integer_word_exact:
        # Exact UInt64 -> F64 admission comes from the source's significand
        # certificate. Fixed offsets round the word on 2**29 and 2**42 grids
        # in that order; the subtractions are exact by Sterbenz.
        numeric=f"F64FromU64({raw})"
        rounded_numeric=f"(({numeric} + 2**81) - 2**81)"
        normal=f"Float64(U64FromF64(({rounded_numeric} + 2**94) - 2**94))"
        rounded32=f"U64And(U64FromF64({rounded_numeric}), 9223372036854775807)"
    sub=f"Float64(U64Or(Bits64((Float64({rounded32}) + 2**28) - 2**28), {sign}))"
    infinity=f"Float64(U64Or(9218868437227405312, {sign}))"
    # Thresholds are preimages of representable F32 values, with even ties.
    small_bits=0x3f10000000000000-(1<<28)
    overflow_bits=0x40effe0000000000-(1<<28)
    small=2**-14-2**-39
    overflow=65520-2**-9
    maximum=max(abs(certificate.minimum),abs(certificate.maximum))
    minimum=certificate.minimum if certificate.minimum>0 else -certificate.maximum if certificate.maximum<0 else 0
    if maximum<overflow:above=normal
    else:above=f"Piecewise(({normal}, {mag} < {overflow_bits}), ({infinity}, True))"
    condition=f"{mag} < {small_bits}" if small_condition is None else small_condition
    classification=None
    if facts:
        guard=compiler.substitute(condition,"X999999997",source,domains)
        classification=compiler.branch_facts.truth(syntax(guard),facts)
        if classification is not None:compiler.condition_events.append(("numeric/tandem","small","path-proved-"+str(classification).lower()))
    mixed=False
    if minimum>=small or certificate.quantum>=-24 or classification is False:template=above
    elif maximum<small or classification is True:template=sub
    else:
        mixed=True
        template=f"Piecewise(({sub}, {condition}), ({above}, True))"
    if sign_expression is not None:
        template=compiler.substitute(template,"X999999996",sign_expression,domains)
    result=simplify_words(compiler.substitute(template,"X999999997",source,domains),compiler,domains)
    if mixed and frontier is not None:
        guard=compiler.substitute(condition,'X999999997',source,domains)
        frontier.append((guard,-2**-14,2**-14,-24))
    return result

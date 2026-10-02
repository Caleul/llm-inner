"""Certified F64 -> F32 -> F16 composition using elementary word operations.

F32 must round a finite normal-or-zero source, without overflow. No real
arithmetic reassociation is used. Other sources retain the generic route.
"""
from direct_sympy_words import simplify_words


def supported(certificate):
    return certificate is not None and certificate.quantum is not None and certificate.quantum>=-126 and max(abs(certificate.minimum),abs(certificate.maximum))<2**128-2**103


def lower_tandem(source,certificate,compiler,domains):
    if not supported(certificate):raise ValueError("Tandem conversion requires finite normal-or-zero F32 source")
    source=compiler.stabilize("("+source+")",domains)
    raw="Bits64(X999999997)"
    mag=f"U64And({raw}, 9223372036854775807)"
    sign=f"U64And({raw}, 9223372036854775808)"
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
    if maximum<small:template=sub
    elif minimum>=small or certificate.quantum>=-14:template=above
    else:template=f"Piecewise(({sub}, {mag} < {small_bits}), ({above}, True))"
    return simplify_words(compiler.substitute(template,"X999999997",source,domains),compiler,domains)

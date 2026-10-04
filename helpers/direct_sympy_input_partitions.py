"""Disjoint finite-Half input regions for compiler-only context refinement.

Ranks select interval boundaries, never model outputs. Every runtime guard
uses the original input value. Signed zeros share the same numeric interval;
no region containing zero assumes a positive zero.
"""
from fractions import Fraction as F
import math
import struct

from direct_sympy_strings import Domain

MAX_RANK=0x7bff


def rank(value):
    original=F(value)
    value=float(value)
    if not math.isfinite(value):raise ValueError('Finite Half boundary required')
    try:bits=struct.unpack('>H',struct.pack('>e',value))[0]
    except OverflowError:raise ValueError('Finite Half boundary required') from None
    represented=struct.unpack('>e',struct.pack('>H',bits))[0]
    if F(represented)!=original:raise ValueError('Exactly representable Half boundary required')
    magnitude=bits&0x7fff
    return -magnitude if bits&0x8000 else magnitude


def value(index):
    if type(index) is not int or not -MAX_RANK<=index<=MAX_RANK:
        raise ValueError('Finite Half rank required')
    bits=abs(index)|(0x8000 if index<0 else 0)
    return F(struct.unpack('>e',struct.pack('>H',bits))[0])


def interval(low,high):
    a,b=rank(low),rank(high)
    if a>b:raise ValueError('Ordered interval required')
    minimum=low if low>0 else -high if high<0 else 0
    q=max(-24,math.frexp(float(minimum))[1]-11) if minimum else -24
    return Domain(F(low),F(high),q,False)


def validate_domains(domains,originals):
    if set(domains)!=set(originals):raise ValueError('All fundamental inputs must have a domain')
    result={}
    for name,domain in domains.items():
        original=originals[name]
        actual=interval(domain.minimum,domain.maximum)
        if domain.minimum<original.minimum or domain.maximum>original.maximum:
            raise ValueError('Partition extends outside admitted input domain')
        if domain.quantum>actual.quantum:
            raise ValueError('Partition grid excludes admitted Half inputs')
        if domain.excludes_negative_zero and domain.minimum<=0<=domain.maximum:
            raise ValueError('A zero-containing partition must retain both signed zeros')
        result[name]=domain
    return result


def split(domains,axis):
    source=domains[axis];low,high=rank(source.minimum),rank(source.maximum)
    if low==high:raise ValueError('Cannot subdivide a singleton input interval')
    midpoint=(low+high)//2;boundary=value(midpoint)
    left=dict(domains);right=dict(domains)
    left[axis]=interval(source.minimum,boundary)
    right[axis]=interval(value(midpoint+1),source.maximum)
    return f'{axis} <= {repr(float(boundary))}',left,right

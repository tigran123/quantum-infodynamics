# Physical model of relativistic harmonic oscillator
from numpy import sqrt,sign

params = {'$m$': 1.0, '$c$': 1.0, '$\omega$': 1.0}

_m = params['$m$']
_c = params['$c$']
_omega = params['$\omega$']

def T(p):
    global _m, _c
    return _c*sqrt(p**2 + _m**2*_c**2)

def dTdp(p):
    global _m, _c
    if _m == 0.0:
        return _c*sign(p)
    else:
        return _c*p/sqrt(p**2 + _m**2*_c**2)

def U(x):
    global _m, _omega
    return _m*_omega**2*x**2/2.

def dUdx(x):
    global _m, _omega
    return _m*_omega**2*x

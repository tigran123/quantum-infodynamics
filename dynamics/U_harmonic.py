# Physical model of non-relativistic harmonic oscillator

params = {'$m$': 1.0, '$\omega$': 1.0}

_m = params['$m$']
_omega = params['$\omega$']

def U(x):
    global _m, _omega
    return _m*_omega**2*x**2/2.

def dUdx(x):
    global _m, _omega
    return _m*_omega**2*x

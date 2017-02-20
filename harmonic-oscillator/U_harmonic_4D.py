# Physical model of non-relativistic harmonic oscillator in 4-dimensional phase space

params = {'$m$': 1.0, '$\omega$': 1.0}

_m = params['$m$']
_omega = params['$\omega$']

def U(x,y):
    global _m, _omega
    return _m*_omega**2*(x**2+y**2)/2.

def dUdx(x,y):
    global _m, _omega
    return _m*_omega**2*x

def dUdy(x,y):
    global _m, _omega
    return _m*_omega**2*y

# Physical model of non-relativistic potential barrier

from numpy import cosh, exp

params = {'$m$': 5.0, '$U_0$': 1.0, r'$\alpha$': 1.0}

_m = params['$m$']
_U0 = params['$U_0$']
_alpha = params[r'$\alpha$']

def T(p):
    global _m
    return p**2/(2.*_m)

def dTdp(p):
    global _m
    return p/_m

def U(x):
    global _U0, _alpha
    return _U0/(cosh(_alpha*x)**2)

def dUdx(x):
    global _U0, _alpha
    y = exp(_alpha*x)
    z = 1.0/y
    return -8.0*_U0*(y-z)/(y+z)**3

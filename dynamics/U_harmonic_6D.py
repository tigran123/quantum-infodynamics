# Physical model of harmonic oscillator in 6-dimensional phase space (3-dimensional space)

params = {'$m$': 1.0, '$\omega$': 1.0}

_m = params['$m$']
_omega = params['$\omega$']

def U(x,y,z):
    global _m, _omega
    return _m*_omega**2*(x**2 + y**2 + z**2)/2.

def dUdx(x,y,z):
    global _m, _omega
    return _m*_omega**2*x

def dUdy(x,y,z):
    global _m, _omega
    return _m*_omega**2*y

def dUdz(x,y,z):
    global _m, _omega
    return _m*_omega**2*z

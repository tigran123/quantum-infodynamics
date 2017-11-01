# Physical model of mathematical pendulum

from numpy import sin,cos

def U(x):
    return 1-cos(x)

def dUdx(x):
    return sin(x)

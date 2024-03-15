#mass = 1.0 # kg
#omega = 1.0 # rad/s

def U(x):
    #global mass, omega
    #return mass * omega**2 * x**2/2
    return x**2

def dUdx(x):
    #global mass, omega
    #return mass * omega**2 * x
    return 2.*x

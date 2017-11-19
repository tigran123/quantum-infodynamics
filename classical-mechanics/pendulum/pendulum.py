"""
  Mathematical Pendulum Simulator (Python Class Definition)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  This program was written in 2017 and placed in public domain.

  origin
   |\
     \
   |  \
       \
   |    \
         \ L
   |      \
           \
   |--phi-> \
             \
   |          \
               \
   |            \
                 O M
   |
"""

from numpy import sin, cos, pi
from scipy.integrate import odeint

class Pendulum:
    """Pendulum Class --- model of a mathematical pendulum
       We use Lagrangian dynamics in variables (phi, phidot = dphi/dt)
    """
    def __init__(self,
                 phi    = pi, # initial angle phi, in radians
                 phidot =  0.0,  # initial angular velocity = dphi/dt, in radian/s
                 L      =  1.0,  # length of pendulum in m
                 M      =  1.0,  # mass of pendulum in kg
                 G      =  9.81, # standard gravity in m/s^2
                 color  = 'k',   # boring black colour by default :)
                 origin = (0, 0)): # coordinates of the suspension point
        self.phi = phi
        self.phidot = phidot
        self.L = L
        self.M = M
        self.G = G
        self.color = color # colour to paint this pendulum
        self.origin = origin
        self.cs = None # matplotlib contour set artist for this pendulum
        self.line = None # matplotlib line artist for this pendulum
        self.energy_text = None # matplotlib text artist for the energy value
        self.t = 0.0 # elapsed time of simulation in s

    def position(self):
        """Return the current position of the pendulum"""
        L = self.L
        phi = self.phi
        x = self.origin[0] + L*sin(phi)
        y = self.origin[1] - L*cos(phi)
        return [[self.origin[0], x], [self.origin[1],y]]

    def Hamiltonian(self, phi, phidot):
        """Return the total energy (Kinetic+Potential) of the specified state"""
        M = self.M
        L = self.L
        G = self.G
        T = 0.5*M*L**2*phidot**2
        U = -M*G*L*cos(phi)
        return T + U

    def energy(self):
        """Return the total energy (Kinetic+Potential) of the current state"""
        return self.Hamiltonian(self.phi, self.phidot)

    def derivs(self, state, t):
        """Return the RHS of the ODEs of motion"""
        return [state[1], 0.0 if abs(state[0]) == pi else -self.G*sin(state[0])/self.L]

    def step(self, dt):
        """Evolve the system by time step dt"""
        t = self.t
        self.phi,self.phidot = odeint(self.derivs, [self.phi,self.phidot], [t, t + dt])[1]
        if self.phi > pi: self.phi -= 2*pi    # the phase space is a cylinder, so we must wrap ...
        elif self.phi < -pi: self.phi += 2*pi # ... phi around to remain within [-pi, pi]
        self.t += dt

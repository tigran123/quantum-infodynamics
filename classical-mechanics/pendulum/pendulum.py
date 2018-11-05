"""
  Mathematical Pendulum Simulator (Python Class Definition)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  Released under GPLv3, 2017.
"""

import numpy as np
import scipy.integrate

class Pendulum:
    """Pendulum Class --- model of a mathematical pendulum.
       Uses Lagrangian dynamics in variables (phi, phidot = dphi/dt)
       The motion of pendulum does not depend on the mass, so by "energy" we mean "energy per unit mass"
    """
    def __init__(self,
                 phi    =  np.pi,  # initial angle phi, in radians
                 phidot =  0.0,    # initial angular velocity = dphi/dt, in radian/s
                 L      =  1.0,    # length of pendulum in m
                 G      =  9.80665,# standard gravity in m/s^2
                 color  = 'k',     # boring black colour by default :)
                 origin = (0, 0)): # coordinates of the suspension point
        self.phi = phi
        self.phidot = phidot
        self.L = L
        self.G = G
        self.color = color # colour to paint this pendulum
        self.origin = origin
        self.cs = None # matplotlib contour set artist for this pendulum
        self.line = None # matplotlib line artist for this pendulum
        self.energy_text = None # matplotlib text artist for the energy value

    def position(self):
        """Return the current position of the pendulum"""
        L = self.L
        phi = self.phi
        x = self.origin[0] + L*np.sin(phi)
        y = self.origin[1] - L*np.cos(phi)
        return [[self.origin[0], x], [self.origin[1],y]]

    def Hamiltonian(self, phi, phidot):
        """Return the total (Kinetic+Potential) energy per unit mass of the specified state"""
        L = self.L
        G = self.G
        T = 0.5*L**2*phidot**2
        U = -G*L*np.cos(phi)
        return T + U

    def energy(self):
        """Return the total (Kinetic+Potential) energy per unit mass of the current state"""
        return self.Hamiltonian(self.phi, self.phidot)

    def derivs(self, state, t):
        """Return the RHS of the ODEs of motion"""
        return [state[1], 0.0 if abs(state[0]) == np.pi else -self.G*np.sin(state[0])/self.L]

    def evolve(self, t1, t2):
        """Evolve the pendulum from the moment of time t1 to t2"""
        self.phi,self.phidot = scipy.integrate.odeint(self.derivs, [self.phi,self.phidot], [t1, t2])[1]
        # the phase space is a cylinder, so we must wrap phi around to remain within [-pi, pi]
        if self.phi > np.pi:
            self.phi -= 2*np.pi
        elif self.phi < -np.pi:
            self.phi += 2*np.pi

    def free(self):
        """Free the resources held by this instance"""
        self.line.remove()
        del(self.line)
        self.energy_text.remove()
        del(self.energy_text)
        while True:
            try: self.cs.pop_label()
            except IndexError: break
        for c in self.cs.collections: c.remove()

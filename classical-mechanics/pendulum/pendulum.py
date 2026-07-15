"""
  Mathematical Pendulum Simulator (Python Class Definition)
  Author: Tigran Aivazian <aivazian.tigran@gmail.com>
  Released under GPLv3, 2017.
"""

from numpy import pi, sin, cos
from scipy.integrate import odeint

class Pendulum:
    """Pendulum Class --- model of a mathematical pendulum.
       Uses Lagrangian dynamics in variables (phi, phidot = dphi/dt)
       The motion of pendulum does not depend on the mass, so by "energy" we mean "energy per unit mass"
    """
    def __init__(self,
                 phi    =  0.0,  # initial angle phi in radians
                 phidot =  0.0,    # initial angular velocity = dphi/dt in radian/s
                 L      =  1.0,    # length of pendulum in m
                 G      =  9.80665,# standard gravity in m/s^2
                 color  = 'k',     # plain black colour by default
                 live   = True):   # simulated (True) or staged, i.e. drawn frozen (False)
        self.phi = phi
        self.phidot = phidot
        self.L = L
        self.G = G
        self.color = color # colour to paint this pendulum with
        self.live = live

    def position(self):
        """Return the current position of the pendulum"""
        L = self.L
        phi = self.phi
        return [[0, L*sin(phi)], [0, -L*cos(phi)]]

    def Hamiltonian(self, phi, phidot):
        """Return the total (Kinetic+Potential) energy per unit mass of the specified state"""
        L = self.L
        G = self.G
        return 0.5*L**2*phidot**2 - G*L*cos(phi)

    def energy(self):
        """Return the total (Kinetic+Potential) energy per unit mass of the current state"""
        return self.Hamiltonian(self.phi, self.phidot)

    def __derivs(self, state, t):
        """Return the RHS of the ODEs of motion, used by .evolve() method"""
        # zero the torque at phi = +-pi to pin the unstable equilibrium there: sin(pi) in
        # floating point is ~1.2e-16, not 0, and this residual grows as exp(sqrt(G/L)*t)
        # near the hyperbolic point, toppling the balanced pendulum in ~10 s at default G/L
        return [state[1], 0.0 if abs(state[0]) == pi else -self.G*sin(state[0])/self.L]

    def evolve(self, t1, t2):
        """Evolve the pendulum from the moment of time t1 to t2"""
        self.phi,self.phidot = odeint(self.__derivs, [self.phi,self.phidot], [t1, t2])[1]
        # the phase space is a cylinder, so we must wrap phi around to remain within [-pi, pi];
        # modulo, not a single +-2*pi correction: one step may advance phi by more than 2*pi
        # when |phidot*(t2-t1)| is large; strict '>' leaves phi = +pi untouched, so the
        # equilibrium pinned in __derivs() stays at +pi instead of being relabelled as -pi
        if abs(self.phi) > pi:
            self.phi = (self.phi + pi) % (2*pi) - pi

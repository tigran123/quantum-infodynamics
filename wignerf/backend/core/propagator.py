"""
Spectral split-operator propagator of 2nd order for the Wigner function
W(x,p,t), after Cabrera, Bondar, Jacobs, Rabitz (2015). Direct port of the
validated batch implementation in dynamics/solve.py, reshaped as a class
with mutable physics parameters for interactive use.

Variants: quantum/classical x relativistic/non-relativistic, selected per
instance. Units are Hartree atomic units (hbar = m_e = e = 1); c is a
parameter (C_AU by default, c=1 reproduces the old natural-unit runs) and
hbar_eff scales the quantum differential for classical-limit studies.

State convention: W is (Nx, Np) float64, fftshifted along both axes.
One step is the Strang splitting expT * expU * expT where dT already
carries the factor 1/2 (exactly as in solve.py).

Note the Bopp arguments are real-valued: qd(U, X, 1j*Theta) evaluates U at
X -+ hbar*Theta/2 (complex dtype, zero imaginary part). For real U the
exponent dU is purely imaginary, so |expU| = 1 and the evolution is
unconditionally norm-stable; accuracy is governed by adjust_step.
"""

import logging

from .xp import C_AU

log = logging.getLogger(__name__)


class Propagator:
    def __init__(self, grid, *, quantum=True, relativistic=False,
                 mass=1.0, c=C_AU, hbar_eff=1.0, tol=1e-2, U=None, dUdx=None):
        if U is None:
            raise ValueError("U(x) callable is required")
        if not quantum and dUdx is None:
            raise ValueError("dUdx(x) callable is required for the classical propagator")
        self.grid = grid
        self.backend = grid.backend
        self.xp = grid.backend.xp
        self.quantum = bool(quantum)
        self.relativistic = bool(relativistic)
        self.mass = float(mass)
        self.c = float(c)
        self.hbar_eff = float(hbar_eff)
        self.tol = float(tol)
        self.U = U
        self.dUdx = dUdx

        self._fft0, self._ifft0 = self.backend.fft_pair((grid.Nx, grid.Np), axis=0)
        self._fft1, self._ifft1 = self.backend.fft_pair((grid.Nx, grid.Np), axis=1)
        self.rebuild()

    # -- physics construction ---------------------------------------------

    def qd(self, f, x, dx):
        """Quantum differential of f at x on the increment dx (solve.py:102)."""
        hbar = self.hbar_eff
        return (f(x + 1j*hbar*dx/2.) - f(x - 1j*hbar*dx/2.))/(1j*hbar)

    def _kinetic(self):
        m, c, xp = self.mass, self.c, self.xp
        if self.relativistic:
            T = lambda p: c*xp.sqrt(p**2 + m**2*c**2)
            if m == 0.0:
                dTdp = lambda p: c*xp.sign(p)
            else:
                dTdp = lambda p: c*p/xp.sqrt(p**2 + m**2*c**2)
        else:
            T = lambda p: p**2/(2.*m)
            dTdp = lambda p: p/m
        return T, dTdp

    def rebuild(self):
        """(Re)build the exponent generators dU, dT and the energy mesh H.
        Called on construction and after any change to U/mass/c/hbar_eff."""
        g, xp = self.grid, self.xp
        T, dTdp = self._kinetic()
        if self.quantum:
            dU = self.qd(self.U, g.X, 1j*g.Theta)
            dT = self.qd(T, g.P, -1j*g.Lam)/2.
        else:
            dU = self.dUdx(g.X)*1j*g.Theta
            dT = -dTdp(g.P)*1j*g.Lam/2.
        # Broadcast to full (Nx, Np) so expU/expT multiplications are plain
        # elementwise products regardless of how U/dUdx broadcast.
        shape = (g.Nx, g.Np)
        self.dU = xp.ascontiguousarray(xp.broadcast_to(xp.asarray(dU, dtype=xp.complex128), shape))
        self.dT = xp.ascontiguousarray(xp.broadcast_to(xp.asarray(dT, dtype=xp.complex128), shape))
        # Energy mesh on the shifted grid (display/observables). The rest
        # energy m*c^2 cancels identically inside dT (kinetic term enters
        # only as a difference) but dominates <H>; observables subtract it.
        self.H = xp.ascontiguousarray(
            xp.broadcast_to((T(g.P) + self.U(g.X)).real.astype(xp.float64), shape))
        self.rest_energy = self.mass*self.c**2 if self.relativistic else 0.0

    def set_physics(self, *, U=None, dUdx=None, mass=None, c=None,
                    hbar_eff=None, tol=None):
        if U is not None: self.U = U
        if dUdx is not None: self.dUdx = dUdx
        if mass is not None: self.mass = float(mass)
        if c is not None: self.c = float(c)
        if hbar_eff is not None: self.hbar_eff = float(hbar_eff)
        if tol is not None: self.tol = float(tol)
        self.rebuild()

    # -- stepping -----------------------------------------------------------

    def exponents(self, dt):
        xp = self.xp
        return xp.exp(dt*self.dU), xp.exp(dt*self.dT)

    def solve_spectral(self, W, expU, expT):
        """One Strang step (solve.py:130-140). W in shifted order; returns a
        fresh real array (never a view into an FFT plan buffer)."""
        xp = self.xp
        B = xp.asarray(W, dtype=xp.complex128)
        B = self._fft0(B)   # (x,p) -> (lambda,p)
        B *= expT
        B = self._ifft0(B)  # (lambda,p) -> (x,p)
        B = self._fft1(B)   # (x,p) -> (x,theta)
        B *= expU
        B = self._ifft1(B)  # (x,theta) -> (x,p)
        B = self._fft0(B)   # (x,p) -> (lambda,p)
        B *= expT
        B = self._ifft0(B)  # (lambda,p) -> (x,p)
        return xp.ascontiguousarray(B.real)

    def adjust_step(self, dt, W, maxtries=15):
        """Adaptive timestep control (solve.py:142-158): shrink |dt| until one
        full step and two half steps agree to relative tolerance tol.
        Returns (W_next, dt, expU, expT). Works for either sign of dt."""
        xp = self.xp
        tries = 0
        while True:
            tries += 1
            expU, expT = self.exponents(dt)
            W1 = self.solve_spectral(W, expU, expT)
            expUn, expTn = self.exponents(0.5*dt)
            W2 = self.solve_spectral(self.solve_spectral(W, expUn, expTn), expUn, expTn)
            rel = float(xp.sum(xp.abs(W1 - W2))/xp.sum(xp.abs(W1)))
            if rel < self.tol:
                break
            if tries > maxtries:
                log.warning("adjust_step: giving up after %d attempts (rel=%.3g)", maxtries, rel)
                break
            dt *= 0.7
        return W1, dt, expU, expT

"""
Phase-space grid and Fourier-conjugate meshes (port of dynamics/solve.py:82-100).

Conventions:
- x and p vectors exclude the right endpoint (periodic domain).
- `xv`, `pv` are 1D device arrays in natural order (for previews/marginal axes).
- `X`, `P`, `Theta`, `Lam` are fftshifted, shaped for broadcasting:
  X, Lam are (Nx, 1); P, Theta are (1, Np). All propagator state W(x, p) is
  kept fftshifted along both axes; the 2D unshift for display happens in the
  frontend shader as a half-period texture offset (requires even Nx, Np).
- theta is conjugate to p (theta_amp = pi/dp), lam is conjugate to x
  (lam_amp = pi/dx). The quantum propagator evaluates U on the extended real
  range [x1 - hbar*theta_amp/2, x2 + hbar*theta_amp/2] and T on
  [p1 - hbar*lam_amp/2, p2 + hbar*lam_amp/2].
"""

import warnings

from numpy import pi


class Grid:
    def __init__(self, x1, x2, Nx, p1, p2, Np, backend):
        if not (x2 > x1 and p2 > p1):
            raise ValueError("require x2 > x1 and p2 > p1")
        if Nx < 2 or Np < 2 or Nx % 2 or Np % 2:
            raise ValueError("Nx and Np must be even and >= 2")
        if Nx & (Nx - 1):
            warnings.warn("Nx=%d is not a power of 2, FFT may be slowed down" % Nx)
        if Np & (Np - 1):
            warnings.warn("Np=%d is not a power of 2, FFT may be slowed down" % Np)

        self.backend = backend
        xp = backend.xp
        self.x1, self.x2, self.Nx = float(x1), float(x2), int(Nx)
        self.p1, self.p2, self.Np = float(p1), float(p2), int(Np)
        self.dx = (self.x2 - self.x1) / self.Nx
        self.dp = (self.p2 - self.p1) / self.Np

        self.xv = self.x1 + self.dx * xp.arange(self.Nx, dtype=xp.float64)
        self.pv = self.p1 + self.dp * xp.arange(self.Np, dtype=xp.float64)

        dtheta = 2. * pi / (self.p2 - self.p1)
        self.theta_amp = dtheta * self.Np / 2.        # == pi/dp
        thetav = -self.theta_amp + dtheta * xp.arange(self.Np, dtype=xp.float64)

        dlam = 2. * pi / (self.x2 - self.x1)
        self.lam_amp = dlam * self.Nx / 2.            # == pi/dx
        lamv = -self.lam_amp + dlam * xp.arange(self.Nx, dtype=xp.float64)

        self.X = backend.fftshift(self.xv)[:, None]
        self.P = backend.fftshift(self.pv)[None, :]
        self.Theta = backend.fftshift(thetav)[None, :]
        self.Lam = backend.fftshift(lamv)[:, None]

    def shift2d(self, W):
        """Natural-order W(x,p) -> propagator (fftshifted) order."""
        return self.backend.fftshift(W, axes=(0, 1))

    def unshift2d(self, W):
        """Propagator order -> natural order (display/diagnostics only; the
        streaming path leaves this to the frontend shader)."""
        return self.backend.ifftshift(W, axes=(0, 1))

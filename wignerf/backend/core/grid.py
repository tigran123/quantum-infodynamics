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
from dataclasses import dataclass, replace

from numpy import pi


class Grid:
    def __init__(self, x1, x2, Nx, p1, p2, Np, backend, dx=None, dp=None,
                 x_anchor=None, p_anchor=None):
        """`dx`/`dp` override the derived cell sizes: an auto-expand regrid
        keeps the ORIGINAL lattice spacing bitwise-frozen (extents move by
        whole cells, so re-deriving (x2-x1)/Nx could differ by an ulp) —
        pass GridState.dx/dp there; the default derivation is unchanged for
        session-start grids. `x_anchor`/`p_anchor` = (anchor, offset_cells):
        materialize lattice points as anchor + d*(offset + i), the same
        float expression for the same GLOBAL cell in every window, so
        overlap points are bitwise-identical across regrids (an offset of 0
        reduces to the default x1 + d*i exactly)."""
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
        self.dx = float(dx) if dx is not None else (self.x2 - self.x1) / self.Nx
        self.dp = float(dp) if dp is not None else (self.p2 - self.p1) / self.Np

        if x_anchor is not None:
            x0a, oxa = x_anchor
            self.xv = float(x0a) + self.dx * (
                float(oxa) + xp.arange(self.Nx, dtype=xp.float64))
        else:
            self.xv = self.x1 + self.dx * xp.arange(self.Nx, dtype=xp.float64)
        if p_anchor is not None:
            p0a, opa = p_anchor
            self.pv = float(p0a) + self.dp * (
                float(opa) + xp.arange(self.Np, dtype=xp.float64))
        else:
            self.pv = self.p1 + self.dp * xp.arange(self.Np, dtype=xp.float64)

        # spectral spans from N*d (== the extent span up to an ulp) so the
        # conjugate amplitudes pi/dp, pi/dx are invariant across regrids
        dtheta = 2. * pi / (self.Np * self.dp) if dp is not None \
            else 2. * pi / (self.p2 - self.p1)
        self.theta_amp = dtheta * self.Np / 2.        # == pi/dp
        thetav = -self.theta_amp + dtheta * xp.arange(self.Np, dtype=xp.float64)

        dlam = 2. * pi / (self.Nx * self.dx) if dx is not None \
            else 2. * pi / (self.x2 - self.x1)
        self.lam_amp = dlam * self.Nx / 2.            # == pi/dx
        lamv = -self.lam_amp + dlam * xp.arange(self.Nx, dtype=xp.float64)

        self.X = backend.fftshift(self.xv)[:, None]
        self.P = backend.fftshift(self.pv)[None, :]
        self.Theta = backend.fftshift(thetav)[None, :]
        self.Lam = backend.fftshift(lamv)[:, None]

    def geom(self):
        """This grid's geometry as the wire/history record fact."""
        from .protocol import RecordGeom
        return RecordGeom(self.Nx, self.Np, self.x1, self.x2, self.p1, self.p2)

    def shift2d(self, W):
        """Natural-order W(x,p) -> propagator (fftshifted) order."""
        return self.backend.fftshift(W, axes=(0, 1))

    def unshift2d(self, W):
        """Propagator order -> natural order (display/diagnostics only; the
        streaming path leaves this to the frontend shader)."""
        return self.backend.ifftshift(W, axes=(0, 1))


@dataclass(frozen=True)
class GridState:
    """The session's live grid window on the FROZEN lattice — the exactness
    linchpin of auto-expand. dx/dp and the anchor (x0, p0) are fixed at
    session creation; every regrid is pure integer arithmetic on the window
    (ox, Nx, op, Np), and extents materialize as x0 + integer*dx — the same
    float expression every time, so lattice points are bitwise-reproducible
    across regrids and no state value is ever interpolated."""
    x0: float    # anchor: x1/p1 of the ORIGINAL grid
    p0: float
    dx: float    # frozen cell sizes (never re-derived from extents)
    dp: float
    ox: int      # window offset from the anchor, in cells (signed)
    op: int
    Nx: int
    Np: int

    @classmethod
    def from_spec(cls, spec):
        """Initial window from a GridSpec (dx/dp derived exactly as
        Grid.__init__ derives them, so the two agree bitwise)."""
        return cls(x0=spec.x1, p0=spec.p1,
                   dx=(spec.x2 - spec.x1)/spec.Nx,
                   dp=(spec.p2 - spec.p1)/spec.Np,
                   ox=0, op=0, Nx=spec.Nx, Np=spec.Np)

    @property
    def x1(self):
        return self.x0 + self.ox*self.dx

    @property
    def x2(self):
        return self.x0 + (self.ox + self.Nx)*self.dx

    @property
    def p1(self):
        return self.p0 + self.op*self.dp

    @property
    def p2(self):
        return self.p0 + (self.op + self.Np)*self.dp

    def geom(self):
        from .protocol import RecordGeom
        return RecordGeom(self.Nx, self.Np, self.x1, self.x2, self.p1, self.p2)

    def make_grid(self, backend):
        return Grid(self.x1, self.x2, self.Nx, self.p1, self.p2, self.Np,
                    backend, dx=self.dx, dp=self.dp,
                    x_anchor=(self.x0, self.ox), p_anchor=(self.p0, self.op))

    def x_extended(self, hbar_eff):
        """Same contract as GridSpec.x_extended: the quantum propagator
        evaluates U on this range (half-width depends only on dp, which is
        frozen — regrids move the interval, never widen the extension)."""
        h = hbar_eff*(pi/self.dp)/2.
        return (self.x1 - h, self.x2 + h)

    def union(self, other):
        """Covering window of two windows on the same lattice (validity
        checks while a regrid is pending; NOT power-of-2, never propagated)."""
        ox = min(self.ox, other.ox)
        op = min(self.op, other.op)
        return replace(self, ox=ox, op=op,
                       Nx=max(self.ox + self.Nx, other.ox + other.Nx) - ox,
                       Np=max(self.op + self.Np, other.op + other.Np) - op)


def embed_window(Wn, old, new, xp):
    """The exact-regrid primitive: natural-order W on window `old` ->
    natural-order W on window `new` (GridStates on one lattice). Overlap
    cells are COPIED by global cell index (bitwise, no interpolation);
    entering cells are zero; leaving cells are dropped (the trigger
    threshold bounds their mass)."""
    Wnew = xp.zeros((new.Nx, new.Np), dtype=xp.float64)
    gx0, gx1 = max(old.ox, new.ox), min(old.ox + old.Nx, new.ox + new.Nx)
    gp0, gp1 = max(old.op, new.op), min(old.op + old.Np, new.op + new.Np)
    if gx1 > gx0 and gp1 > gp0:
        Wnew[gx0 - new.ox:gx1 - new.ox, gp0 - new.op:gp1 - new.op] = \
            Wn[gx0 - old.ox:gx1 - old.ox, gp0 - old.op:gp1 - old.op]
    return Wnew

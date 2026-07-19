"""
Initial (Cauchy) data on the (x,p) grid: sums of Gaussians.

Two types:
- mixture_wigner: statistical mixture — weighted sum of Gaussian blobs,
  W >= 0 everywhere (the generalization of dynamics/initgauss.py);
  sigma_x and sigma_p are independent per component.
- cat_wigner: coherent superposition psi = sum_j c_j psi_j of minimal
  Gaussian packets (sigma_p = hbar_eff/(2 sigma_x), derived), built from
  the analytic pairwise cross-Wigner closed form — exact on the grid,
  normalized analytically by <psi|psi>. Interference cross-terms give
  the oscillatory fringes and negative regions.

Returned arrays are in NATURAL (unshifted) order; the caller applies
grid.shift2d() before propagation.
"""

import cmath
from dataclasses import dataclass
from math import sqrt as msqrt

from numpy import pi


@dataclass
class GaussianComponent:
    x0: float
    p0: float
    sigma_x: float
    sigma_p: float
    weight: float = 1.0
    phase: float = 0.0   # used only by the cat-state builder

    def validate(self):
        if self.sigma_x <= 0 or self.sigma_p <= 0:
            raise ValueError("sigma_x and sigma_p must be positive")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


def mixture_wigner(grid, components):
    """Weighted sum of Gaussian phase-space blobs, normalized so that the
    analytic integral of W over the plane is 1 (each blob carries
    Z = 1/(2*pi*sigma_x*sigma_p), weights are normalized to unit sum)."""
    if not components:
        raise ValueError("at least one Gaussian component is required")
    xp = grid.backend.xp
    x = grid.xv[:, None]
    p = grid.pv[None, :]
    wtot = sum(c.weight for c in components)
    W = xp.zeros((grid.Nx, grid.Np), dtype=xp.float64)
    for c in components:
        c.validate()
        Z = c.weight/(wtot*2.*pi*c.sigma_x*c.sigma_p)
        W += Z*xp.exp(-((x - c.x0)**2/(2.*c.sigma_x**2) + (p - c.p0)**2/(2.*c.sigma_p**2)))
    return W


def _overlap(cj, ck, hbar):
    """<psi_k|psi_j> = integral psi_j psi_k* dx for minimal packets
    psi_j = N_j exp(-a_j (x-x_j)^2 + i p_j (x-x_j)/hbar), closed form via
    Gaussian integral of exp(-A x^2 + B x + C)."""
    aj, ak = 1./(4.*cj.sigma_x**2), 1./(4.*ck.sigma_x**2)
    Nj = (2.*pi*cj.sigma_x**2)**(-0.25)
    Nk = (2.*pi*ck.sigma_x**2)**(-0.25)
    A = aj + ak
    B = 2.*aj*cj.x0 + 2.*ak*ck.x0 + 1j*(cj.p0 - ck.p0)/hbar
    C = -aj*cj.x0**2 - ak*ck.x0**2 - 1j*(cj.p0*cj.x0 - ck.p0*ck.x0)/hbar
    return Nj*Nk*msqrt(pi)/cmath.sqrt(A)*cmath.exp(B*B/(4.*A) + C)


def cat_wigner(grid, components, hbar_eff=1.0):
    """Wigner function of psi = (1/sqrt(S)) sum_j c_j psi_j with
    c_j = sqrt(weight_j) exp(i phase_j) and minimal packets (sigma_p is
    derived, component.sigma_p is ignored). Uses the pairwise closed form

      W_jk = (N_j N_k / 2 pi hbar) sqrt(pi/alpha)
             * exp(i (p_j u_j - p_k u_k)/hbar)
             * exp(-a_j u_j^2 - a_k u_k^2 + beta^2/(4 alpha)),

    u_j = x - x_j, alpha = (a_j + a_k)/4,
    beta = -a_j u_j + a_k u_k + i (p_j + p_k - 2 p)/(2 hbar),
    W = sum_j |c_j|^2 W_jj + 2 Re sum_{j<k} c_j c_k^* W_jk, normalized by
    S = <psi|psi> computed from the same closed-form overlaps (never by
    grid sum, which would hide truncation error)."""
    if not components:
        raise ValueError("at least one Gaussian component is required")
    for c in components:
        c.validate()
    xp = grid.backend.xp
    hbar = float(hbar_eff)
    x = grid.xv[:, None]
    p = grid.pv[None, :]

    a = [1./(4.*c.sigma_x**2) for c in components]
    N = [(2.*pi*c.sigma_x**2)**(-0.25) for c in components]
    amp = [msqrt(c.weight)*cmath.exp(1j*c.phase) for c in components]

    W = xp.zeros((grid.Nx, grid.Np), dtype=xp.float64)
    for j, cj in enumerate(components):
        uj = x - cj.x0
        for k in range(j, len(components)):
            ck = components[k]
            uk = x - ck.x0
            alpha = (a[j] + a[k])/4.
            beta = -a[j]*uj + a[k]*uk + 1j*(cj.p0 + ck.p0 - 2.*p)/(2.*hbar)
            core = (N[j]*N[k]/(2.*pi*hbar))*msqrt(pi/alpha) \
                * xp.exp(-a[j]*uj**2 - a[k]*uk**2 + beta**2/(4.*alpha)) \
                * xp.exp(1j*(cj.p0*uj - ck.p0*uk)/hbar)
            term = (amp[j]*amp[k].conjugate()*core).real
            W += term if j == k else 2.*term

    S = sum(amp[j]*amp[k].conjugate()*_overlap(components[j], components[k], hbar)
            for j in range(len(components)) for k in range(len(components)))
    return W/S.real


def minimal_sigma_p(sigma_x, hbar_eff=1.0):
    """sigma_p of a minimal (pure-state) packet; the UI shows this as the
    derived, read-only sigma_p for cat components."""
    return hbar_eff/(2.*sigma_x)


def from_spec(grid_spec, ic, hbar_eff, backend):
    """Build (Grid, W_natural, warnings) from protocol.GridSpec/ICSpec
    (duck-typed: anything with the same attributes works). Shared by the
    IC preview endpoint and by each SolverWorker at session start."""
    from .grid import Grid
    g = Grid(grid_spec.x1, grid_spec.x2, grid_spec.Nx,
             grid_spec.p1, grid_spec.p2, grid_spec.Np, backend)
    comps = []
    for c in ic.components:
        sp_ = (minimal_sigma_p(c.sigma_x, hbar_eff) if ic.type == "cat"
               else c.sigma_p)
        if sp_ is None:
            raise ValueError("sigma_p is required for mixture components")
        comps.append(GaussianComponent(c.x0, c.p0, c.sigma_x, sp_,
                                       c.weight, c.phase))
    if ic.type == "cat":
        W = cat_wigner(g, comps, hbar_eff)
    else:
        W = mixture_wigner(g, comps)
    return g, W, preview_warnings(g, comps, ic.type, hbar_eff, W)


def preview_warnings(grid, components, kind, hbar_eff=1.0, W=None):
    """Quality diagnostics for an initial condition (warnings, not blocks):
    - fringe Nyquist (cat): packets separated by dx_jk interfere with
      p-period 2*pi*hbar/|dx_jk|; require dp < pi*hbar/|dx_jk| with a
      safety factor of 2 (dually dx vs dp_jk);
    - packet mass within 4 sigma of a domain edge (tails wrap under the
      periodic spectral propagator);
    - quantum validity of the TOTAL W via the purity bound: any density
      operator satisfies Tr rho^2 = 2*pi*hbar * integral W^2 dx dp <= 1.
      This is a necessary condition on the complete state — individual
      components are only a decomposition and carry no physical meaning
      (a valid W may well be a sum of sub-Heisenberg blobs). Violation
      proves W is not a quantum state; it remains a perfectly good
      classical phase-space density.

    Messages may contain any Unicode; the preview endpoint percent-encodes
    them for HTTP-header transport."""
    warns = []
    hbar = float(hbar_eff)
    for i, c in enumerate(components):
        if (c.x0 - 4.*c.sigma_x < grid.x1 or c.x0 + 4.*c.sigma_x > grid.x2
                or c.p0 - 4.*c.sigma_p < grid.p1 or c.p0 + 4.*c.sigma_p > grid.p2):
            warns.append("component %d is within 4σ of a domain edge "
                         "(tails will wrap around)" % (i + 1))
    if W is not None:
        purity = 2.*pi*hbar*float((W*W).sum())*grid.dx*grid.dp
        if purity > 1. + 1e-6:
            warns.append("W is not a valid quantum state: Tr ρ² = "
                         "2πℏ∬W²dxdp = %.8g > 1 (fine for the classical "
                         "variants)" % purity)
    if kind == "cat":
        n = len(components)
        for j in range(n):
            for k in range(j + 1, n):
                dxjk = abs(components[j].x0 - components[k].x0)
                dpjk = abs(components[j].p0 - components[k].p0)
                if dxjk > 0 and grid.dp > pi*hbar/dxjk/2.:
                    warns.append("interference fringes of components %d and %d "
                                 "(p-period %.4g) are under-resolved by dp = %.4g"
                                 % (j + 1, k + 1, 2.*pi*hbar/dxjk, grid.dp))
                if dpjk > 0 and grid.dx > pi*hbar/dpjk/2.:
                    warns.append("interference fringes of components %d and %d "
                                 "(x-period %.4g) are under-resolved by dx = %.4g"
                                 % (j + 1, k + 1, 2.*pi*hbar/dpjk, grid.dx))
    return warns

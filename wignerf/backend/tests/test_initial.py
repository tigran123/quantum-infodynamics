"""Cat-state initial condition: closed-form correctness checks."""

from math import pi, sqrt

import numpy as np
import pytest

from core.xp import ArrayBackend
from core.grid import Grid
from core.initial import (GaussianComponent, cat_wigner, minimal_sigma_p,
                          mixture_wigner, preview_warnings)


@pytest.fixture(scope="module")
def grid():
    return Grid(-8.0, 8.0, 128, -8.0, 8.0, 128, ArrayBackend(device="cpu"))


def _comp(x0, p0, sx, weight=1.0, phase=0.0):
    return GaussianComponent(x0, p0, sx, minimal_sigma_p(sx), weight, phase)


def test_single_packet_reduces_to_coherent_gaussian(grid):
    sx = 0.6
    Wc = cat_wigner(grid, [_comp(1.0, -0.5, sx)])
    Wm = mixture_wigner(grid, [_comp(1.0, -0.5, sx)])
    np.testing.assert_allclose(Wc, Wm, atol=1e-12)


def test_cat_normalization_analytic(grid):
    """Normalization is by the analytic <psi|psi>; the grid sum then equals
    1 up to tail truncation only."""
    W = cat_wigner(grid, [_comp(-2.5, 0.0, 0.5), _comp(2.5, 0.0, 0.5)])
    assert abs(float(W.sum())*grid.dx*grid.dp - 1.0) < 1e-6


def test_cat_has_negative_interference_fringes(grid):
    W = cat_wigner(grid, [_comp(-2.5, 0.0, 0.5), _comp(2.5, 0.0, 0.5)])
    assert float(W.min()) < -0.05*float(W.max())
    # mixture of the same packets is non-negative
    Wm = mixture_wigner(grid, [_comp(-2.5, 0.0, 0.5), _comp(2.5, 0.0, 0.5)])
    assert float(Wm.min()) > -1e-15


def test_cat_rho_matches_wavefunction(grid):
    """rho(x) = integral W dp must equal |psi(x)|^2 built directly."""
    comps = [_comp(-2.0, 0.5, 0.5, weight=1.0),
             _comp(2.0, -1.0, 0.7, weight=0.5, phase=0.9)]
    W = cat_wigner(grid, comps)
    rho = np.asarray(W.sum(axis=1))*grid.dp

    x = np.asarray(grid.xv)
    psi = np.zeros_like(x, dtype=complex)
    for c in comps:
        a = 1./(4.*c.sigma_x**2)
        N = (2.*pi*c.sigma_x**2)**(-0.25)
        amp = sqrt(c.weight)*np.exp(1j*c.phase)
        psi += amp*N*np.exp(-a*(x - c.x0)**2 + 1j*c.p0*(x - c.x0))
    S = (np.abs(psi)**2).sum()*grid.dx
    np.testing.assert_allclose(rho, np.abs(psi)**2/S, atol=1e-6)


def test_cat_fringe_spacing(grid):
    """Packets at +-x0 produce phi(p) fringes with period 2*pi*hbar/(2*x0)."""
    x0 = 2.5
    W = cat_wigner(grid, [_comp(-x0, 0.0, 0.5), _comp(x0, 0.0, 0.5)])
    mid = np.asarray(W[np.asarray(grid.xv).searchsorted(0.0), :])
    p = np.asarray(grid.pv)
    # W(0, p) ~ cos(2*x0*p/hbar): check the oscillation period via FFT peak
    spec = np.abs(np.fft.rfft(mid - mid.mean()))
    freq = np.fft.rfftfreq(len(mid), d=grid.dp)[spec.argmax()]
    assert freq == pytest.approx(2.*x0/(2.*pi), rel=0.05)


def test_hbar_eff_scales_fringes(grid):
    """Smaller hbar_eff -> finer fringes and larger |W| extrema."""
    comps = lambda h: [GaussianComponent(-2.0, 0.0, 0.5, minimal_sigma_p(0.5, h)),
                       GaussianComponent(2.0, 0.0, 0.5, minimal_sigma_p(0.5, h))]
    W1 = cat_wigner(grid, comps(1.0), hbar_eff=1.0)
    W2 = cat_wigner(grid, comps(0.5), hbar_eff=0.5)
    assert abs(float(W2.min())) > abs(float(W1.min()))


def test_preview_warnings(grid):
    # near the edge
    w = preview_warnings(grid, [_comp(7.9, 0.0, 0.5)], "cat")
    assert any("edge" in s for s in w)
    # under-resolved fringes: huge separation
    c = [_comp(-7.0, 0.0, 0.3), _comp(7.0, 0.0, 0.3)]
    g2 = Grid(-8.0, 8.0, 64, -8.0, 8.0, 16, ArrayBackend(device="cpu"))
    assert any("under-resolved" in s for s in preview_warnings(g2, c, "cat"))


def test_quantum_validity_is_a_property_of_the_total_W(grid):
    """The purity bound Tr rho^2 = 2*pi*hbar*int W^2 <= 1 applies to the
    complete W, never to individual components (which are only a
    decomposition)."""
    # a single sub-Heisenberg blob IS an invalid state: purity > 1
    bad = [GaussianComponent(0, 0, 0.1, 0.1)]
    w = preview_warnings(grid, bad, "mixture", 1.0, mixture_wigner(grid, bad))
    assert any("not a valid quantum state" in s for s in w)
    # but a mixture OF sub-Heisenberg components (sigma_x*sigma_p = 0.4
    # < 1/2 each) can pass the bound: two equal-weight separated blobs
    # give Tr rho^2 ~ (0.25 + 0.25)*hbar/(2*0.4) = 0.625 <= 1 -> no warning
    ok = [GaussianComponent(-2.0, 0.0, 0.4, 1.0),
          GaussianComponent(2.0, 0.0, 0.4, 1.0)]
    w = preview_warnings(grid, ok, "mixture", 1.0, mixture_wigner(grid, ok))
    assert not any("not a valid" in s for s in w)
    # a cat state is pure: purity == 1 on a resolving grid -> no warning
    cat = [_comp(-2.0, 0.0, 0.5), _comp(2.0, 0.0, 0.5)]
    w = preview_warnings(grid, cat, "cat", 1.0, cat_wigner(grid, cat))
    assert not any("not a valid" in s for s in w)

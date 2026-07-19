"""
Physics-invariant tests for the core propagator (plan Phase 1).

Reference system: harmonic oscillator U = x^2/2 with m = 1 (omega = 1) in
atomic units, coherent-state Gaussian (sigma_x = sigma_p = 1/sqrt(2),
sigma_x*sigma_p = hbar/2) centred at (x0, p0) = (2, 0). For quadratic H the
Moyal corrections vanish, so quantum and classical evolution must agree to
roundoff — the strongest single correctness check of the port.
"""

from math import pi, sqrt

import numpy as np
import pytest

from core.xp import ArrayBackend
from core.grid import Grid
from core.propagator import Propagator
from core.initial import GaussianComponent, mixture_wigner
from core import observables

SIG = 1.0/sqrt(2.0)


@pytest.fixture(scope="module")
def backend():
    return ArrayBackend(device="cpu")


@pytest.fixture(scope="module")
def grid(backend):
    return Grid(-6.0, 6.0, 64, -7.0, 7.0, 64, backend)


def harmonic():
    return dict(U=lambda x: x**2/2., dUdx=lambda x: x)


def coherent_w0(grid):
    return mixture_wigner(grid, [GaussianComponent(2.0, 0.0, SIG, SIG)])


def evolve(prop, W_shifted, dt, nsteps):
    expU, expT = prop.exponents(dt)
    W = W_shifted
    for _ in range(nsteps):
        W = prop.solve_spectral(W, expU, expT)
    return W


def test_mixture_normalization(grid):
    # Bound set by Gaussian tail truncation at the domain edge (the +x tail
    # is ~5.7 sigma from x2=6 => ~1e-8 missing mass), not by quadrature.
    W = coherent_w0(grid)
    norm = float(W.sum())*grid.dx*grid.dp
    assert abs(norm - 1.0) < 1e-6


def test_exponents_unit_modulus(grid):
    for quantum in (True, False):
        prop = Propagator(grid, quantum=quantum, **harmonic())
        expU, expT = prop.exponents(0.01)
        assert float(np.max(np.abs(np.abs(expU) - 1.0))) < 1e-12
        assert float(np.max(np.abs(np.abs(expT) - 1.0))) < 1e-12


def test_quantum_equals_classical_exponents(grid):
    """For quadratic U and T the quantum differential reduces exactly to the
    classical derivative form: the dU/dT arrays must agree to roundoff."""
    q = Propagator(grid, quantum=True, **harmonic())
    c = Propagator(grid, quantum=False, **harmonic())
    assert float(np.max(np.abs(q.dU - c.dU))) < 1e-12
    assert float(np.max(np.abs(q.dT - c.dT))) < 1e-12


def test_quantum_equals_classical_evolution(grid):
    q = Propagator(grid, quantum=True, **harmonic())
    c = Propagator(grid, quantum=False, **harmonic())
    W0 = grid.shift2d(coherent_w0(grid))
    Wq = evolve(q, W0, 0.01, 200)
    Wc = evolve(c, W0, 0.01, 200)
    assert float(np.max(np.abs(Wq - Wc))) < 1e-10


def test_norm_conserved(grid):
    prop = Propagator(grid, quantum=True, **harmonic())
    W0 = grid.shift2d(coherent_w0(grid))
    n0 = float(W0.sum())*grid.dx*grid.dp
    W = evolve(prop, W0, 0.01, 200)
    n = float(W.sum())*grid.dx*grid.dp
    assert abs(n - n0) < 1e-12


def test_coherent_state_revival(grid):
    """After one period t = 2*pi the coherent state must return to itself
    (up to 2nd-order splitting error)."""
    prop = Propagator(grid, quantum=True, **harmonic())
    W0n = coherent_w0(grid)
    W0 = grid.shift2d(W0n)
    n = 2000
    W = evolve(prop, W0, 2.*pi/n, n)
    rel_l1 = float(np.sum(np.abs(W - W0)))/float(np.sum(np.abs(W0)))
    assert rel_l1 < 0.02


def test_center_follows_classical_trajectory(grid):
    """At t = pi/2 the packet centre must sit at (x, p) = (0, -2)."""
    prop = Propagator(grid, quantum=True, **harmonic())
    W = evolve(prop, grid.shift2d(coherent_w0(grid)), (pi/2.)/500, 500)
    obs = observables.compute(W, prop)
    assert abs(obs.x_mean - 0.0) < 1e-3
    assert abs(obs.p_mean - (-2.0)) < 1e-3


def test_time_reversal(grid):
    # Reversal is exact up to roundoff: each step's real() projection drops
    # an ~1e-16-relative imaginary residue, accumulating to ~1e-9 over
    # 600 steps. The bound is set well above that but far below any
    # physical signal.
    prop = Propagator(grid, quantum=True, **harmonic())
    W0 = grid.shift2d(coherent_w0(grid))
    W = evolve(prop, W0, 0.01, 300)
    W = evolve(prop, W, -0.01, 300)
    assert float(np.max(np.abs(W - W0))) < 1e-7


def test_energy_flat(grid):
    prop = Propagator(grid, quantum=True, **harmonic())
    W = grid.shift2d(coherent_w0(grid))
    E0 = observables.compute(W, prop).E
    # Coherent state at (2,0): E = (x0^2+p0^2)/2 + hbar*omega/2 = 2.5
    assert abs(E0 - 2.5) < 1e-6
    W = evolve(prop, W, 0.005, 400)
    assert abs(observables.compute(W, prop).E - E0)/abs(E0) < 1e-3


def test_observables_shift_consistency(grid, backend):
    """Observables computed on the shifted state with shifted meshes must
    equal the naive natural-order computation (catches fftshift bookkeeping
    bugs, the likeliest porting error)."""
    prop = Propagator(grid, quantum=True, **harmonic())
    Wn = coherent_w0(grid)
    obs = observables.compute(grid.shift2d(Wn), prop)

    x = backend.asnumpy(grid.xv)[:, None]
    p = backend.asnumpy(grid.pv)[None, :]
    Wh = backend.asnumpy(Wn)
    dxdp = grid.dx*grid.dp
    assert abs(obs.norm - Wh.sum()*dxdp) < 1e-12
    assert abs(obs.x_mean - (x*Wh).sum()*dxdp) < 1e-12
    assert abs(obs.p_mean - (p*Wh).sum()*dxdp) < 1e-12
    E_nat = ((p**2/2. + x**2/2.)*Wh).sum()*dxdp
    assert abs(obs.E - E_nat) < 1e-10
    np.testing.assert_allclose(obs.rho, Wh.sum(axis=1)*grid.dp, atol=1e-14)
    np.testing.assert_allclose(obs.phi, Wh.sum(axis=0)*grid.dx, atol=1e-14)


def test_relativistic_matches_nonrel_at_large_c(grid):
    """c = 1e4 makes (p/mc)^2 ~ 1e-8 while keeping the kinetic-difference
    cancellation error (~m*c^2*eps) far below the phase scale. (At c ~ 1e6
    the difference T(p+hL/2)-T(p-hL/2) of ~1e12-magnitude terms would lose
    precision to cancellation — deliberate choice, do not 'strengthen'.)"""
    nr = Propagator(grid, quantum=True, relativistic=False, **harmonic())
    re = Propagator(grid, quantum=True, relativistic=True, c=1e4, **harmonic())
    W0 = grid.shift2d(coherent_w0(grid))
    Wn = evolve(nr, W0, 0.01, 100)
    Wr = evolve(re, W0, 0.01, 100)
    assert float(np.max(np.abs(Wn - Wr))) < 1e-6
    # and the rest energy must be subtracted from E
    obs = observables.compute(Wr, re)
    assert abs(obs.E - 2.5) < 1e-3


def test_hbar_eff_to_zero_is_classical_limit(grid):
    """For anharmonic U = x^4/4 the Moyal corrections are O(hbar_eff^2):
    shrinking hbar_eff must drive the quantum evolution toward the
    classical one for the SAME (positive) Cauchy data."""
    quartic = dict(U=lambda x: x**4/4., dUdx=lambda x: x**3)
    W0 = grid.shift2d(mixture_wigner(
        grid, [GaussianComponent(1.5, 0.0, 0.7, 0.7)]))
    Wc = evolve(Propagator(grid, quantum=False, **quartic), W0, 0.002, 250)

    def err(h):
        q = Propagator(grid, quantum=True, hbar_eff=h, **quartic)
        Wq = evolve(q, W0, 0.002, 250)
        return float(np.sum(np.abs(Wq - Wc)))

    e1, e01 = err(1.0), err(0.1)
    assert e01 < 0.5*e1


def test_adjust_step_shrinks_dt(grid):
    """A deliberately huge dt must be shrunk by the adaptive controller."""
    prop = Propagator(grid, quantum=True, tol=1e-3, **harmonic())
    W0 = grid.shift2d(coherent_w0(grid))
    _, dt, _, _ = prop.adjust_step(1.0, W0)
    assert 0 < dt < 1.0


def test_adjust_step_negative_dt(grid):
    prop = Propagator(grid, quantum=True, tol=1e-3, **harmonic())
    W0 = grid.shift2d(coherent_w0(grid))
    _, dt, _, _ = prop.adjust_step(-1.0, W0)
    assert -1.0 < dt < 0

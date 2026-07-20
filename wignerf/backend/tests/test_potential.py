"""Safety and per-family validity tests for the U(x) compiler."""

import numpy as np
import pytest

from core.potential import PotentialError, compile_potential, sample_potential

XR = (-6.0, 6.0)
XEXT = (-60.0, 60.0)   # a typical extended quantum-probe range


def test_basic_compile_and_eval():
    cp = compile_potential("x^2/2", x_range=XR, x_extended=XEXT)
    assert cp.quantum_valid and cp.classical_valid
    xs = np.array([0.0, 2.0, -3.0])
    np.testing.assert_allclose(cp.U(xs), xs**2/2)
    np.testing.assert_allclose(cp.dUdx(xs), xs)
    assert cp.latex


def test_constant_broadcasts():
    cp = compile_potential("5", x_range=XR, x_extended=XEXT)
    xs = np.linspace(-1, 1, 7)
    assert cp.U(xs).shape == xs.shape
    np.testing.assert_allclose(cp.U(xs), 5.0)
    np.testing.assert_allclose(cp.dUdx(xs), 0.0)


def test_complex_dtype_evaluation():
    """The quantum propagator feeds complex-dtype (real-valued) arrays."""
    cp = compile_potential("cos(x) + x**4/4", x_range=XR, x_extended=XEXT)
    z = np.linspace(-60, 60, 11).astype(np.complex128)
    u = cp.U(z)
    assert u.dtype == np.complex128
    assert np.abs(u.imag).max() < 1e-12


@pytest.mark.parametrize("bad", [
    "__import__('os').system('id')",
    "x.__class__",
    "open('/etc/passwd')",
    "eval('1')",
    "lambda: 1",
    "'abc'",
    "x = 1",
    "y + 1",          # unknown free symbol
    "sin",            # not an expression in x -> parse/arity error or non-real
    "",
])
def test_rejected_expressions(bad):
    with pytest.raises(PotentialError):
        compile_potential(bad, x_range=XR)


def test_abs_is_quantum_and_classical_valid():
    """Abs(x) needs no analyticity: the Bopp arguments are real. Its
    classical force is sign(x) — perfectly valid."""
    cp = compile_potential("Abs(x)", x_range=XR, x_extended=XEXT)
    assert cp.quantum_valid
    assert cp.classical_valid
    np.testing.assert_allclose(cp.dUdx(np.array([-2.0, 3.0])), [-1.0, 1.0])


def test_heaviside_is_quantum_only():
    cp = compile_potential("3*Heaviside(x)", x_range=XR, x_extended=XEXT)
    assert cp.quantum_valid
    assert not cp.classical_valid
    assert any("Dirac" in r for r in cp.reasons)


def test_sqrt_branch_is_classical_invalid_quantum_warned():
    """sqrt(x-5) is NaN on most of the real grid (classical-invalid) and
    goes complex on the extended range (quantum warning: non-unitary)."""
    cp = compile_potential("sqrt(x - 5)", x_range=XR, x_extended=XEXT)
    assert not cp.classical_valid
    assert cp.quantum_valid
    assert any("complex" in w for w in cp.warnings)


def test_pole_is_quantum_invalid():
    cp = compile_potential("1/x", x_range=(1.0, 6.0), x_extended=(-60.0, 60.0))
    assert not cp.quantum_valid


def test_piecewise_compiles():
    cp = compile_potential("Piecewise((x**2, x > 0), (0, True))",
                           x_range=XR, x_extended=XEXT)
    assert cp.quantum_valid
    xs = np.array([-1.0, 2.0])
    np.testing.assert_allclose(cp.U(xs), [0.0, 4.0])


def test_soft_coulomb():
    """The workhorse 1D atomic model must be valid everywhere."""
    cp = compile_potential("-1/sqrt(x^2 + 2)", x_range=XR, x_extended=XEXT)
    assert cp.quantum_valid and cp.classical_valid


def test_astronomical_powers_rejected():
    """9**9**9 would make sympy materialize a ~4e8-digit integer during
    parse — the power screen must reject it (and towers like it) before
    any evaluation happens."""
    for bad in ("9**9**9", "x + 2^9999999", "2**(10**6)", "0.5**(-99**99)"):
        with pytest.raises(PotentialError):
            compile_potential(bad, x_range=XR)
    # sane powers are unaffected
    cp = compile_potential("x**8/8 + 2**10", x_range=XR, x_extended=XEXT)
    assert cp.quantum_valid and cp.classical_valid


def test_sample_potential_gaps():
    cp = compile_potential("sqrt(x)", x_range=XR)
    xs, us = sample_potential(cp, -1.0, 1.0, n=16)
    assert len(xs) == len(us) == 16
    assert us[0] is None            # sqrt of negative real -> NaN -> null
    assert us[-1] == pytest.approx(1.0)

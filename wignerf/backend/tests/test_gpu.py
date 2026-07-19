"""
GPU-backend tests (plan Phase 7). Skipped automatically when cupy or the
CUDA device is unavailable, so the suite still passes on the CPU-only VPS.
Device: cuda:1 (the RTX 3090; device 0 drives the displays) — override
with WIGNERF_GPU_DEVICE.
"""

import os
import threading

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from core.grid import Grid
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.quantize import dequantize, quantize
from core.xp import ArrayBackend

DEVICE = os.environ.get("WIGNERF_GPU_DEVICE", "cuda:1")

HARMONIC = dict(U=lambda x: x**2/2., dUdx=lambda x: x)
IC = [GaussianComponent(2.0, 0.0, 0.707, 0.707)]


@pytest.fixture(scope="module")
def gb():
    try:
        return ArrayBackend(device=DEVICE)
    except RuntimeError as e:
        pytest.skip(str(e))


def _evolve(backend, nsteps, N=256, dt=0.01):
    with backend.device():
        g = Grid(-6.0, 6.0, N, -7.0, 7.0, N, backend)
        prop = Propagator(g, quantum=True, **HARMONIC)
        W = g.shift2d(mixture_wigner(g, IC))
        expU, expT = prop.exponents(dt)
        for _ in range(nsteps):
            W = prop.solve_spectral(W, expU, expT)
        backend.synchronize()
        return backend.asnumpy(W)


def test_cpu_gpu_parity(gb):
    """100 harmonic steps must agree between cuFFT and FFTW to roundoff."""
    Wg = _evolve(gb, 100)
    Wc = _evolve(ArrayBackend(device="cpu"), 100)
    assert float(np.max(np.abs(Wg - Wc))) < 1e-10


def test_gpu_quantize_roundtrip(gb):
    with gb.device():
        g = Grid(-6.0, 6.0, 128, -7.0, 7.0, 128, gb)
        W = g.shift2d(mixture_wigner(g, IC))
        wq, wmin, wmax = quantize(W, gb)
    assert isinstance(wq, np.ndarray) and wq.dtype == np.uint16
    Wd = dequantize(wq, wmin, wmax)
    assert float(np.max(np.abs(Wd - gb.asnumpy(W)))) < (wmax - wmin)/65535.


def test_four_concurrent_gpu_workers(gb):
    """Four propagators stepping concurrently in threads on one device
    (the session worker model) must produce identical, finite results."""
    results = [None]*4
    errors = []

    def work(i):
        try:
            b = ArrayBackend(device=DEVICE)
            results[i] = _evolve(b, 60, N=128)
        except Exception as e:                      # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    assert not errors, errors
    for r in results:
        assert r is not None and np.isfinite(r).all()
        assert float(np.max(np.abs(r - results[0]))) == 0.0


def test_gpu_memory_stable(gb):
    """Repeated evolution must not grow the memory pool unboundedly."""
    with gb.device():
        pool = cupy.get_default_memory_pool()
        _evolve(gb, 30, N=128)
        used1 = pool.used_bytes()
        for _ in range(5):
            _evolve(gb, 30, N=128)
        used2 = pool.used_bytes()
    assert used2 <= used1*3

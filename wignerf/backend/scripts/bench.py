"""
Propagator throughput benchmark: steps/second per backend and grid size.

    .venv/bin/python scripts/bench.py [cpu] [cuda:1] ...

No arguments: benchmarks cpu and, when available, cuda:1.
"""

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.grid import Grid
from core.initial import GaussianComponent, mixture_wigner
from core.propagator import Propagator
from core.xp import ArrayBackend

HARMONIC = dict(U=lambda x: x**2/2., dUdx=lambda x: x)


def bench(device, N, nsteps=200):
    b = ArrayBackend(device=device, fft_threads=4 if device == "cpu" else 1)
    with b.device():
        g = Grid(-6.0, 6.0, N, -7.0, 7.0, N, b)
        prop = Propagator(g, quantum=True, **HARMONIC)
        W = g.shift2d(mixture_wigner(g, [GaussianComponent(2, 0, .707, .707)]))
        expU, expT = prop.exponents(0.01)
        for _ in range(10):                      # warm up plans/pool
            W = prop.solve_spectral(W, expU, expT)
        b.synchronize()
        t0 = perf_counter()
        for _ in range(nsteps):
            W = prop.solve_spectral(W, expU, expT)
        b.synchronize()
        dt = perf_counter() - t0
    print("%-18s %5dx%-5d %9.1f steps/s  (%.3f ms/step)"
          % (b.name, N, N, nsteps/dt, 1e3*dt/nsteps))


def main():
    devices = sys.argv[1:] or ["cpu", "cuda:1"]
    for dev in devices:
        for N in (256, 512, 1024):
            try:
                bench(dev, N)
            except Exception as e:
                print("%-18s %5dx%-5d unavailable: %s" % (dev, N, N, e))
                break


if __name__ == "__main__":
    main()

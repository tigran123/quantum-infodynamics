# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quantum Infodynamics — numerical simulation of dynamical systems on both the quantum and classical level, relativistic and non-relativistic, formulated on the Wigner function W(x,p,t) in phase space. Plain Python scripts (no package structure, build system, linter, or test suite). Core stack: numpy, scipy, matplotlib; GUI programs use Qt (PyQt6 for the pendulum simulator, PyQt5 elsewhere); movies are assembled with ffmpeg. Project website: http://quantuminfodynamics.com (source in `docs/`).

Tested versions and minimal prerequisites are listed in `README.md`; `classical-mechanics/pendulum/requirements.txt` has pip-style pins.

## dynamics/ — main solver pipeline

A three-stage pipeline, with stages connected by files:

1. `initgauss.py` — generates Gaussian Cauchy (initial) data on the (x,p) grid.
2. `solve.py` — spectral split-operator propagator of 2nd order with adaptive timestep control. Flags select the propagator: `-c` classical (default is quantum), `-r` relativistic (default non-relativistic). The physical model is loaded by module name: `-u U_harmonic` imports `U_harmonic.py` from the current directory, which must define `U(x)` and `dUdx(x)`.
3. `solanim.py` — renders solution frames to PNGs (parallelizable: `-P <nparts> -p <part>`), which ffmpeg then assembles into an mp4. `solplay.py` plays solutions interactively instead.

File convention: each dataset is a pair — `<name>.npz` (params dict with grid geometry, plus derived arrays) and `<name>_W.npy` (memory-mapped W(x,p,t) array; can be tens of GB, preallocated via `-mmsize` in GB). A solver output can be fed as the next run's `-i` input, so long time ranges are chained as sequential segments.

Orchestration scripts live in `dynamics/bin/` and must be run from `dynamics/`:

```sh
cd dynamics
bin/harmonic-oscillator.sh   # full run: solve (4 variants in parallel) then animate
```

The `*-solve.sh` scripts run quantum/classical × relativistic/non-relativistic variants as parallel background jobs chained over time segments; the `*-anim.sh` scripts fan `solanim.py` out over `nproc` and call ffmpeg. `solve4D.py`/`solve6D.py` are higher-dimensional variants.

Gotchas:
- Nx and Np should be powers of 2, or the FFT slows down (the scripts warn but proceed).
- Physical constants are in "natural units" (ħ = 1, c = 1) — SI and atomic-unit values are present but commented out in `solve.py`.

## classical-mechanics/pendulum/ — Mathematical Pendulum Simulator

PyQt6 + matplotlib GUI, run from its own venv (created with `uv venv`, deps from `requirements.txt`):

```sh
cd classical-mechanics/pendulum
.venv/bin/python psim.py           # interactive GUI (control window + plot window)
```

Recording to mp4 is interactive: the Record button on the Control tab starts capturing frames (via `FFMpegWriter`), and pausing the animation finalises the file.

`psim.py` is the main program, `pendulum.py` defines the `Pendulum` class (Lagrangian dynamics in (φ, φ̇), integrated with scipy `odeint`; phase space is a cylinder, φ wraps to [-π, π]). All Qt names are imported via the `qtapi.py` shim, not directly from PyQt6 — keep it that way when adding widgets, and use fully scoped Qt enums (e.g. `Qt.FocusPolicy.StrongFocus`). The animation uses matplotlib blitting; after any change to plot artists or axes, call `PlotWindow.refresh()` to recapture the blit background.

## Other directories

- `harmonic-oscillator/`, `rotator/` — older standalone solvers; run recipe (frames + ffmpeg) is in `harmonic-oscillator/README.txt`.
- `schrodinger/` — split-operator propagator for the 1D Schrödinger equation in coordinate representation (uses numba).
- `qcr/`, `skelqt/` — Qt-based GUI experiments sharing the same `qtapi.py` shim pattern as the pendulum simulator.
- `docs/` — static HTML/CSS/JS of the project website.

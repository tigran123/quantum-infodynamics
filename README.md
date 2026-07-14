# Quantum Infodynamics
## Theory of Motion in Real Space

This repository contains the programs and any other materials related to Quantum Infodynamics.

* dynamics [MAIN] --- Tools for simulating any [info]dynamical system on both the quantum and the classical level, relativistic and non-relativistic

* classical-mechanics/pendulum --- Mathematical Pendulum Simulator (GUI, using PyQt6 and matplotlib)

For more information see the project's website at http://quantuminfodynamics.com

### Mathematical Pendulum Simulator (classical-mechanics/pendulum/psim.py)

An interactive study of the mathematical pendulum: any number of pendulums are simulated
simultaneously and drawn, each in its own colour, both in real space and as points moving
along their energy level curves in the phase space portrait (φ, φ̇) — a cylinder unrolled
onto [-π, π], with the separatrix passing through the unstable equilibrium.

* Each pendulum has its own tab in the control window: initial conditions (φ and φ̇ — in
  radians or degrees — the length L and the colour) are edited there and applied when the
  pendulum is stopped; while it runs, the φ and φ̇ fields track its current state instead.
* Start/Stop freezes and resumes each pendulum individually; the '+' button creates a new
  pendulum, frozen until started; Delete removes one.
* Global time controls: play/pause, single stepping forwards and backwards in time and a
  slider for the ODE integration time step Δt.
* The Record button captures exactly what is shown on the screen into an mp4 file (piped
  to ffmpeg); stopping the animation finishes the recording.

Run it from its own virtual environment:

```sh
cd classical-mechanics/pendulum
uv venv && uv pip install -r requirements.txt
.venv/bin/python psim.py
```

### Versions currently in use

* Python 3.12.13

* numpy 2.5.1

* scipy 1.18.0

* matplotlib 3.11.0

* PyQt6 6.11.0

* ffmpeg 8.0 (assembles and records the mp4 videos)

* PyFFTW 0.12.0 (optional, for the dynamics/ solvers)

### Minimal pre-requisites for using our python programs

* Python >= 3.6.3

* numpy >= 1.13.3

* scipy >= 1.0.0

* matplotlib >= 2.1.0

* PyFFTW >= 0.10.4 (but the slower scipy.fftpack or even numpy.fft will be selected if PyFFTW is not available)

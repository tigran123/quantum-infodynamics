# Quantum Infodynamics
## Theory of Motion in Real Space

This repository contains the programs and any other materials related to Quantum Infodynamics.

* dynamics [MAIN] --- Tools for simulating any [info]dynamical system on both the quantum and the classical level, relativistic and non-relativistic

* classical-mechanics/pendulum --- Mathematical Pendulum Simulator (GUI, using PyQt6 and matplotlib)

* wignerf --- Interactive client–server simulator of the Wigner function W(x,p,t) in 1D phase space (FastAPI backend + Vue 3 SPA)

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

### Wigner Function Simulator (wignerf/)

Interactive client–server simulator of the Wigner function W(x,p,t) in 1D phase
space, evolved by the spectral split-operator method. A FastAPI backend (Python
3.12, uv-managed venv) streams frames over a WebSocket to a Vue 3 single-page
app; the backend serves the *built* SPA statically, so a running deployment is a
single process listening on port 8010.

`wignerf/start.sh` only **runs** the server — it does not install or build
anything. Create the backend venv and build the frontend once after cloning, and
rebuild them after pulling, as below. (This keeps the systemd service, which
just execs `start.sh`, sandboxed with a read-only home; installing/building
inside the service would need write access to `~/.cache/uv`, `node_modules`,
`frontend/dist`, and more.)

#### Install (after `git clone`)

Backend — create the venv, install the pinned dependencies, and precompile
bytecode. The service runs with a **read-only home**, so Python can never write
`__pycache__/*.pyc` at runtime; precompiling here (while the tree is writable)
keeps server startup fast. `--compile-bytecode` compiles the venv; `compileall`
compiles our own source:

```sh
cd wignerf/backend
uv venv
uv pip sync --compile-bytecode requirements.txt requirements-dev.txt   # add requirements-gpu.txt on a CUDA host
.venv/bin/python -m compileall main.py config.py core routers
```

Frontend — install node dependencies and build the SPA into `frontend/dist`.
For a deployment behind an nginx prefix, export `APP_ROOT_PATH` first so the
build bakes in the right base and API path (the runtime service's
`EnvironmentFile` does not reach the build); on the dev machine (prefix `/`) it
can be omitted:

```sh
cd wignerf/frontend
# export APP_ROOT_PATH=/wignerf     # only for a prefixed prod build
npm ci
npm run build
```

Then start the server:

```sh
cd wignerf
./start.sh                     # serves API + SPA at http://localhost:8010
```

To run it as a service, install `wignerf/wignerf.service` into
`/etc/systemd/system/`, then `sudo systemctl daemon-reload` and
`sudo systemctl enable --now wignerf`.

#### Upgrade (after `git pull`)

Re-sync the backend dependencies (in case the pins changed) and rebuild the SPA
from scratch — the old `frontend/dist` must be removed so a stale build is never
served:

```sh
cd wignerf/backend
uv pip sync --compile-bytecode requirements.txt requirements-dev.txt   # add requirements-gpu.txt on a CUDA host
.venv/bin/python -m compileall main.py config.py core routers

cd ../frontend
# export APP_ROOT_PATH=/wignerf     # only for a prefixed prod build
rm -rf dist
npm ci
npm run build
```

Then restart: `./start.sh` (or `sudo systemctl restart wignerf`).

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

# wignerf — Interactive Wigner Function Simulator

Live client-server simulator of W(x,p,t) in 1D phase space, evolved by the
spectral split-operator method of Cabrera, Bondar, Jacobs, Rabitz (2015)
(PDF at the repo root: `Efficient-Method-2015.pdf`). The validated batch
reference implementation is `../dynamics/solve.py` — the propagator here is
a direct port of its math; **never modify `dynamics/solve.py`** (it carries
uncommitted WIP).

Units are Hartree atomic units everywhere (ħ = mₑ = e = 1). `c` is a session
parameter (default 137.035999; `c=1` reproduces the old natural-unit toy
runs). `hbar_eff` (default 1) scales the quantum differential — a
classical-limit dial, not a unit change. SI (fs/Å/eV) appears in display
labels only (`frontend/src/lib/units.ts`).

## Layout and stack

- `backend/` — FastAPI (uv-managed Python 3.12 venv, port **8010**;
  urantia-library owns 8000). `main.py` is pure wiring; routes in
  `routers/`; all physics/infra in `core/` (no FastAPI imports there).
- `frontend/` — Vue 3 + TypeScript + Vite 8 + Tailwind 4, composables (no
  Pinia), hash routing. Built SPA is served statically by FastAPI when
  `frontend/dist` exists; in dev, Vite proxies `/api` (incl. WebSocket).
- `start.sh` — idempotent prod launcher (venv + deps + build + uvicorn).

Dependency workflow (same as urantia-library): edit `requirements*.in`,
then `uv pip compile requirements[-x].in -o requirements[-x].txt` and
`uv pip sync requirements.txt requirements-dev.txt [requirements-gpu.txt]`.

## Architecture (see the plan in git history / memory for rationale)

- **Streaming**: solver workers append records to an in-RAM byte-capped
  `FrameHistory`; the WS streamer (`routers/stream.py`) sends the newest
  lockstep-complete record (live, coalescing — slow clients skip frames) or
  exact sequential records (replay/scrub). Binary layout in
  `core/protocol.py`, mirrored by `frontend/src/lib/protocol.ts` and
  cross-checked via `scripts/gen_fixture.py` + the frontend vitest.
- **Record grid**: τ_k = t1 + k·record_dt. Each variant (1–4 worker
  threads: quantum/classical × rel/non-rel) integrates with its own
  adaptive dt (`adjust_step`, every 20 steps) but lands exactly on each τ_k
  by clamping the final substep. Same k ⇒ same physical t across variants.
- **State convention**: W is float64, fftshifted along both axes on the
  backend; frames stream in shifted order and the *shader* unshifts via a
  half-period texture offset (`render/WignerRenderer.ts`, R16UI texture,
  manual bilinear with periodic wrap, diverging LUT centered at W=0).
- **Parameter policy**: U(x), c, mass, hbar_eff, tol, dt_sign apply live at
  the frontier; grid/IC/variant-set changes require a session restart.
- **Sessions always start paused** (both modes): computation begins only on
  the explicit Solve/Play command. The transport button label predicts its
  effect: Solve = will compute, Play = pure history playback, Pause while
  running. Setup persists in browser localStorage.
- **Potentials** (`core/potential.py`): tokenize-screen (security boundary)
  → sympy parse → per-family validity. The Bopp arguments are REAL
  (x ∓ ħθ/2, complex dtype only): quantum needs U real+finite on the
  extended range [x1 − πħ/(2dp), x2 + πħ/(2dp)] (Abs is quantum-valid);
  classical needs DiracDelta-free dU/dx (Heaviside steps are quantum-only).
- **ICs** (`core/initial.py`): Gaussian mixtures (independent σx, σp) and
  cat states (analytic pairwise cross-Wigner; σp derived = ħ/(2σx)).

- **Purity** γ = 2πℏ_eff∬W²dxdp (= Tr ρ²) is computed per record and
  streamed/plotted. Both the Moyal flow (unitarity) and the classical
  Liouville flow (incompressibility) conserve it for closed systems, so
  until the Lindblad term exists it is a solver-fidelity diagnostic (a
  contained state holds it to ~1e-12); quantum validity of an IC is a
  property of the TOTAL W (γ ≤ 1 necessary), never of its components.

## GPU

`WIGNERF_DEVICE=auto|cpu|cuda:N` (config.py). `core/xp.py` pins
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match nvidia-smi, and `auto`
picks the largest-memory device (the idle RTX 3090 = cuda:1 on the main
workstation; the 2080 Ti on cuda:0 drives the displays). GPU deps:
`cupy-cuda13x[ctk]` — the `[ctk]` extra is REQUIRED (cupy JIT-compiles
kernels at runtime via NVRTC — never nvcc — and needs the PyPI CUDA
headers/libs; NO system CUDA Toolkit anywhere, only the driver). Note:
CUDA 13 dropped Maxwell/Pascal/Volta — the dev workstation's GTX 1060
(Pascal) needs `cupy-cuda12x[ctk]` instead. RTX 3090: ~2400 steps/s at 512², ~550 at 1024², ~134 at
2048²; CPU (pyfftw): ~75 at 512². Previews always run on CPU by design.
Workers release CuPy pool blocks back to the driver on session close
(nvidia-smi "used" while running is pool recycling, not a leak).

## Configuration (environment variables, read by backend/config.py)

| Variable | Default | Meaning |
|---|---|---|
| `WIGNERF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda:N`. `auto` = largest-memory CUDA device if cupy imports, else CPU. Indices are PCI order (match nvidia-smi). |
| `WIGNERF_PORT` | `8010` | Backend port (8000 belongs to urantia-library). Used by start.sh; `uvicorn --port` otherwise. |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session (scrub/replay window). 32 GiB ≈ 4000 four-variant records at 1024², ≈ 64000 at 256². On the VPS (32 GB RAM shared with urantia-library, Open WebUI, …) set `16384`. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto (ncores/(2·n_variants), capped at 4). Irrelevant on GPU. |

## Commands

```sh
# backend tests (GPU tests auto-skip without cupy/CUDA)
cd backend && .venv/bin/pytest

# live-server streaming smoke test (no browser)
.venv/bin/uvicorn main:app --port 8010 &
.venv/bin/python scripts/ws_smoke.py

# throughput benchmark
.venv/bin/python scripts/bench.py [cpu] [cuda:1]

# frontend: decoder golden test + typecheck + build
cd frontend && npm run test && npm run build

# dev loop: uvicorn (above) + `npm run dev`, open http://localhost:5173
# prod-style: ../start.sh, open http://localhost:8010
```

After changing the binary protocol: bump `VERSION` in BOTH protocol files,
regenerate the fixture (`scripts/gen_fixture.py`), and update the vitest.

UI debugging without touching the real display: drive the BUILT SPA with
headless Chrome via `puppeteer-core` (frontend devDep; system Chrome at
/usr/bin/google-chrome, flags `--no-sandbox --disable-gpu`). The series
plots expose `window.__wfSeries.<which>()` (poller state) and element
screenshots of `.wf-plot` reveal what uPlot actually painted — this is how
the "flat purity line camouflaged on a gridline" bug was found.

## Roadmap (v2, agreed 2026-07-19)

1. **Destructive forking**: resume computation from ANY record (end or
   intermediate; the abandoned branch is discarded), both modes. Requires
   periodic float64 checkpoints alongside the uint16 display history — a
   quantized frame must NEVER seed a propagator. "Continue past t2" is the
   fork-at-the-end special case.
2. **Save/load the whole simulation** to disk (config + history +
   checkpoints; own format, no legacy compatibility).
3. Multi-GPU: distribute variant workers across all CUDA devices
   (`WIGNERF_DEVICE` as comma list / `auto` = all, fastest first;
   relativistic variants to the faster card — lockstep gates on the
   slowest worker; ~+40% expected for 2- and 4-variant runs on the
   3090 + 2080 Ti pair). Workers are already fully independent per
   device — no propagator changes needed.
4. mp4 export. Later: Lindblad dissipation (the propagator's exponent
   construction is deliberately modular for it), multi-D.

## Conventions / gotchas

- Do not reference the old project website domain anywhere in wignerf —
  it expired (old code/comments elsewhere in the repo may keep theirs).
- Nx, Np must be even (shader unshift + fftshift symmetry); powers of 2
  for FFT speed. Grid warns, API schema enforces evenness.
- Physics invariants in `tests/test_propagator.py` are the correctness
  anchor — harmonic quantum ≡ classical (Moyal terms vanish for quadratic
  H) is the strongest single check; run them after touching propagator,
  grid or fftshift bookkeeping.
- pyFFTW plans are per-`ArrayBackend`-instance and must not be shared
  across threads; each worker owns its backend.
- Relativistic variants: mc² cancels inside the propagator; observables
  subtract it from displayed E.
- **Secular E drift + slow purity decay = boundary wrap, not a solver
  bug.** The spectral domain is a torus: when a state's orbit + ~5σ tails
  reach the x or p edge, mass wraps through the seam and the run faithfully
  evolves the WRONG (torus) problem. Tells: IC norm deficit >> 1e-6, the
  4σ edge warning, secular (not oscillatory-bounded) drifts. Fix: enlarge
  the domain. Verified: same cat state, [-6,6]x[-7,7] gives E drift 2e-3;
  [-12,12]² gives 4e-6 with purity conserved to 5e-12 — the discrete map
  is exactly unitary for contained states (healthy E behavior is a BOUNDED
  O(dt²) oscillation from Strang splitting, never a drift).

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
- `start.sh` — prod launcher: runs uvicorn only (guards that `backend/.venv`
  and `frontend/dist` exist, else errors). Install/build is manual and
  pre-service — see the wignerf section of the top-level `README.md`.

Dependency workflow (same as urantia-library): edit `requirements*.in`,
then `uv pip compile requirements[-x].in -o requirements[-x].txt` and
`uv pip sync requirements.txt requirements-dev.txt [requirements-gpu.txt]`.

## Architecture (see the plan in git history / memory for rationale)

- **Streaming**: solver workers append records to an in-RAM byte-capped
  `FrameHistory`; the WS streamer (`routers/stream.py`) sends the newest
  lockstep-complete record (live, coalescing — slow clients skip frames) or
  exact sequential records (replay/scrub). Computation ALWAYS runs at full
  speed in both modes — neither the dial nor a slow client ever throttles
  the workers; `delay` (seconds injected between played-back frames)
  paces only the display. The dial's "0" position (default) means one
  record per display refresh — the fastest speed at which every frame is
  still painted: the client measures its refresh interval (lib/perf.ts)
  and sends that as the delay, and every dial position is clamped to at
  least it, so delivery never outpaces painting. Replay never skips a
  record; it slips on WS backpressure when the client can't keep up. The
  UI dial is "0" plus a log range 20 ms–1.5 s. Client frame fan-out is
  rAF-timed (useSession: decode per message, paint one frame per
  animation frame; small FIFO with drop-to-newest as a burst safety
  valve), so texture uploads, uPlot updates and Vue reactivity run per
  PAINTED frame by construction. A playback-only run must never coalesce to the
  frontier while sequential records are unsent (that would teleport
  playback to the end), and its auto-pause is delivery-aware — it fires
  only after the frontier record was SENT. The transport must stay
  responsive under full frame backpressure: control JSON (status echoes)
  is flushed BEFORE frame sends each tick, play/pause are echoed
  immediately, replay batches are wall-clock-budgeted (~0.2 s) and
  preempted by pause/seek, and the client flips the transport button
  optimistically on play/pause. The delay dial is settable only while
  PAUSED (pause → change → resume) and its thumb is local UI state,
  re-synced from status when idle. Binary layout in
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
- **Boundary watch / auto-expand** (`core/boundary.py`): detection is
  ALWAYS on — every record, each worker sums the outer edge band of the
  ρ/φ marginals it already computed (host-side, O(Nx+Np), no extra device
  sync) and `session.report_edge` posts a `boundary` WS event on state
  change (band = max(4, N/32) cells/side, trigger 1e-6 — expansion
  prevents wrap, it cannot repair it, so it must fire while edge mass is
  negligible). The `auto_expand` toggle (SessionCreate field AND
  live-appliable via ParamChange) governs only the RESPONSE: an exact
  fixed-lattice regrid. dx/dp and the lattice anchor are FROZEN at session
  creation (`GridState`, integer window arithmetic; extents materialize as
  anchor + integer·dx, and `Grid` takes explicit dx/dp + anchors so overlap
  lattice points are bitwise-identical across regrids); move = whole-cell
  window shift, expand = double an axis (powers of 2, support centered,
  combined move+double; NO shrink, NO interpolation ever — norm/E/purity
  survive to machine precision minus the ≤threshold dropped tails). The
  session commits a `RegridPlan(epoch, k_star, state)` with k_star past
  every in-flight record; each worker applies it before computing its
  first record ≥ k_star (`embed_window` + `Propagator.set_grid`), so the
  switch is lockstep-uniform and records <k_star stay old-geometry. U is
  revalidated on the union extended Bopp range BEFORE commit (refusal ⇒
  `invalid_potential` warning, keep computing). **Plan commits and physics
  commits are mutually exclusive** (both hold `_edge_lock` for their whole
  body, and `apply_params` orders physics BEFORE any immediate schedule):
  U/hbar_eff move the Bopp range, a plan validated under stale physics
  would hit the deliberately-fatal non-finite check at k_star (a per-worker
  rollback there would desync lockstep geometry), so a pending plan's union
  window is revalidated under incoming physics and the change is REJECTED
  if it does not hold — this also closes the race of a plan committing
  during the streamer's ~ms validation compile. Expansion caps at
  `WIGNERF_MAX_GRID` (`capped` warning, keep computing; pure moves still
  work at the cap). Geometry is a PER-RECORD fact: protocol v3 headers
  carry Nx/Np/x1/x2/p1/p2, history stores geom per record, the streamer
  packs from the record (never the session), and the frontend follows the
  PAINTED frame (panels/overlays/marginal axes re-derive per frame;
  zoom windows remap to the same physical region) — so scrubbing across a
  regrid boundary just works. Each doubling ≈ 4× step cost and 4×
  bytes/record (the history cap then holds ¼ the records).
- **Export panel** (header button "⤓ export") carries two things: the mp4
  below, and the run's SETUP — `GET /sessions/{id}/setup` serves
  `describe.setup_document`, the config the session was CREATED with
  (`state_at(cfg, log, -1)` rewinds every live change; live changes are
  deliberately not part of a starting state — the video's metadata block is
  where they are recorded). Import fills the setup form and marks the
  session restart-dirty, never restarts by itself (`lib/config.importConfig`,
  in-place merge on the reactive cfg), and accepts that .json OR an exported
  .mp4: `lib/mp4meta.ts` scans the file's head for the same document in the
  `comment` tag (faststart keeps it there — byte ~3.5k), so a kept video is
  self-restoring.
  A render is destroyed by anything that moves the session on — Restart
  deletes the session (`close` → `videoexport.close_session`, file unlinked
  mid-write) and computing new records evicts the ones behind the renderer
  (`record N is no longer retained`). Both used to happen SILENTLY, so
  `SimulatorView.mayDiscardExport` confirms first (Restart, and a transport
  command whose action is `solve` — playback adds no records and is never
  gated) and cancels the job outright on "yes", instead of leaving it to die
  mid-file. The automatic restarts (first mount, backend recovery) never
  prompt.
- **mp4 export** (`core/videoexport.py` + `core/render_mpl.py` +
  `routers/export.py`): renders an ALREADY-COMPUTED record range on the
  BACKEND — matplotlib/Agg frames piped as raw RGBA into
  `ffmpeg -c:v libx264` (system ffmpeg, absence ⇒ 503). PAUSED-only (409
  while running): a running session evicts old records, and the feature is
  for filming a range you already played back. Two passes: a scan collects
  the E/ΔX·ΔP/γ series, the per-variant FIXED colour scale (no brightness
  flicker), the fixed marginal amplitudes and the widest window any record
  used, and proves every record is still retained before ffmpeg starts; then
  one figure update per frame. Only VALUE scales are export-wide — the
  SPATIAL axes follow each record's own geometry (`_apply_geom`, which also
  re-captures the blit background since ticks are static art), exactly as
  the SPA follows the painted frame; freezing them at the union rendered
  every frame before an auto-expansion as a stamp in the corner of its
  panel, and the union now only labels the metadata block. The figure is
  built ONCE and BLITTED (static background + ~15 animated artists): 465 →
  ~17-80 ms/frame measured at FHD (~320 ms at 4K, 4 variants), the
  difference between minutes and half an hour for a 1000-frame export.
  Sizes offered: FHD / QHD / 4K UHD. The figure is always 19.2×10.8 in and
  the RESOLUTION RIDES ON THE DPI (`FrameFigure.REF_WIDTH`) — font sizes
  are in points, so a fixed dpi would render every label at half its
  relative size at 4K. The downloaded name is descriptive
  (`wignerf-QN-QR-CN-CR-41rec-3840x2160-20260722-0107.mp4`, via
  `Content-Disposition`) while the on-disk path keeps session+job ids: two
  exports of the same range in one minute must not collide, least of all
  while one is being downloaded. Frame content mirrors the SPA (panels +
  marginals + series with a time cursor, variant colours/dashes from
  `lib/variants.ts`, the shader's symmetric bwr scale) plus a metadata
  block. The video must READ like the screen: plot titles are copied
  verbatim from `SeriesPlot.vue`/`MarginalsPlot.vue` (γ keeps the UI's
  "purity γ(t) = 2πℏ∬W²dxdp", never an equivalent like Tr ρ²), field labels
  match the Setup panel (ℏ, "run-ahead"), and the series y-window +
  tick decimals reproduce that component's `scales.y.range` rule
  (`render_mpl.series_ylim`); the "grid lines on plots" toggle rides along
  in `ExportSpec.show_grid` and governs EVERY plot in the frame — charts
  get uPlot's `#3f3f46`, the W panels get `GridOverlay.vue`'s
  rgba(120,120,120,.28/.55-at-zero) drawn AFTER the image (matplotlib puts
  the axes grid under it, which is why the heatmaps first had none; the
  lines are animated artists ordered behind the images in `_dynamic`) — matplotlib's own autoscale renders a 2e-5
  purity drift as a dramatic dive with a "×10⁻⁵+1" offset where the UI
  shows a flat line at 1.000000, from byte-identical data. Mirror any
  change to those rules on both sides. The block carries
  U(x), parameters, the IC as an analytic expression
  (`core/describe.py`; cat states print ψ(x,0), the compact complete form),
  and any live parameter change inside the range (`session.param_log`) —
  so one frame documents the whole run; the same facts go into the mp4
  `comment` tag as JSON. Progress: `export` events on the session WS plus a
  REST poll; the file lives in `WIGNERF_EXPORT_DIR` until downloaded, TTL
  (30 min), session close or shutdown. The header button stays ENABLED
  while computing (a disabled button explained only by a tooltip is how
  this feature first read as broken): the panel states the gate and
  "Pause & render" pauses, waits for the server to confirm and re-seeds an
  untouched range before posting. Rendering continues while the popover is
  CLOSED (the poll and the WS events keep updating), so the button IS the
  notification — "⤓ export 42%" while running, emerald "⤓ export ready"
  (red "failed") when finished, and reopening it collects the file; a
  finished job survives reopening and is dropped only by a new render or a
  session change (a restart deletes the old session's files). The panel re-reads the extent from
  `GET /sessions/{id}` when it opens — the streamed status lags a frame
  burst by up to seconds after a pause, and seeding the range from it
  silently exported half the history.
- **Parameter policy**: U(x), c, mass, hbar_eff, tol, dt_sign, auto_expand
  apply live at the frontier; grid/IC/variant-set changes require a session
  restart (auto-expand moves the LIVE grid; the Setup panel shows it and
  offers "adopt" to copy it into the form).
  `apply_params` compares against what is LIVE and drops the fields that
  did not change — no worker command, no `param_log` entry, no
  `params_applied`, and nothing at all if the whole message is a no-op (the
  UI sends complete fields; "Apply live" always carries the U(x) draft, so
  the log used to fill with U changes that never happened and an export's
  "how to reproduce this" block lied about its own frames). Entries carry
  `before` as well as `applied`, so the block renders "ℏ 1 → 2" and
  `describe.state_at` rewinds the header physics to the FIRST exported
  record instead of quoting the values the run ended with. Live changes are
  visible in the UI: the header flashes "✓ applied …", a Physics field
  whose form value differs from `status` renders amber (they apply on
  blur/Enter), and "Apply live" is greyed with an inline reason when the
  draft already IS the live U.
  The setup form gates the transport: while the potential draft is invalid
  for the active variant families or the IC preview errors, Solve (button
  AND Space) is disabled and "Use at restart"/"Apply live" are greyed —
  a computation must never run behind a visibly broken form.
- **Sessions always start paused** (both modes): computation begins only on
  the explicit Solve/Play command. The transport button label predicts its
  effect: Solve = will compute, Play = pure history playback, Pause while
  running. Playback-only runs (play pressed behind the frontier, or after a
  finished run-ahead) auto-pause AT the frontier — they never roll into
  computation; only an explicit Solve does (`SessionClock.stop_at_frontier`).
  A run-ahead that REACHES t2 ends the run too (`advance_cursor`, same
  delivery-aware condition): its workers already idle there, and leaving
  `running` set froze the transport on "Pause" forever and locked out
  every paused-only action (pinned by `test_runahead_starts_paused_and_
  stops_at_t2`).
  Setup persists in browser localStorage; "↺ defaults" (IC editor) and
  "Reset setup to defaults" (Setup panel) restore defaults in the form and
  mark the session restart-dirty.
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

`WIGNERF_DEVICE=auto|cpu|cuda:N|comma list` (config.py) names a device
POOL. `core/xp.resolve_devices` expands it fastest-first (`auto` = all
CUDA devices ranked by SM count; an explicit list like `cuda:1,cuda:0` is
trusted as written) and `core/session.assign_devices` spreads variant
workers over it: costliest variants (relativistic, then quantum) and the
larger share go to the fastest card; each worker owns its own
`ArrayBackend`, so no propagator code is device-aware. `core/xp.py` pins
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match nvidia-smi (RTX 3090 =
cuda:1, the display-driving 2080 Ti = cuda:0 on the main workstation).
GPU deps: `cupy-cuda13x[ctk]` — the `[ctk]` extra is REQUIRED (cupy
JIT-compiles kernels at runtime via NVRTC — never nvcc — and needs the
PyPI CUDA headers/libs; NO system CUDA Toolkit anywhere, only the
driver). Note: CUDA 13 dropped Maxwell/Pascal/Volta — the dev
workstation's GTX 1060 (Pascal) needs `cupy-cuda12x[ctk]` instead.
RTX 3090: ~2400 steps/s at 512², ~550 at 1024², ~134 at 2048²; 2080 Ti:
~390 at 1024²; CPU (pyfftw): ~75 at 512². Measured 4-worker lockstep at
1024²: 135 steps/s all-on-3090 vs 191 split 2+2 across the pair (+41%,
and 2+2 beats 3+1's 181 — the even chunk is right); 2 workers: 270 vs
376 (+39%). Previews always run on CPU by design. Workers release CuPy
pool blocks back to the driver on session close (nvidia-smi "used" while
running is pool recycling, not a leak).

## Configuration (environment variables, read by backend/config.py)

| Variable | Default | Meaning |
|---|---|---|
| `WIGNERF_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda:N` \| comma list (`cuda:1,cuda:0`). Names the device pool; sessions spread variant workers across it. `auto` = all CUDA devices fastest-first if cupy imports, else CPU; a list's order IS the speed ranking. Indices are PCI order (match nvidia-smi). |
| `WIGNERF_PORT` | `8010` | Backend port (8000 belongs to urantia-library). Used by start.sh; `uvicorn --port` otherwise. |
| `WIGNERF_HISTORY_MB` | `32768` | In-RAM frame-history cap per session (scrub/replay window). 32 GiB ≈ 4000 four-variant records at 1024², ≈ 64000 at 256². On the VPS (32 GB RAM shared with urantia-library, Open WebUI, …) set `16384`. |
| `WIGNERF_FFT_THREADS` | `0` | Threads per CPU FFT; `0` = auto (ncores/(2·n_variants), capped at 4). Irrelevant on GPU. |
| `WIGNERF_EXPORT_DIR` | `<tempdir>/wignerf-exports` | Where mp4 exports are written before download. Under systemd (`PrivateTmp=yes`) the default is a private tmpfs — i.e. RAM, wiped on restart; point it at a disk path for long 1440p exports. Files are removed after download, on session close, at shutdown, or 30 min after finishing. |
| `WIGNERF_MAX_GRID` | `4096` | Per-axis Nx/Np ceiling — enforced at session creation AND for auto-expand doublings; tunable BOTH ways (schema sanity rail: 16384). The UI's Nx/Np selects follow it (status carries `max_grid`). Lower it on VRAM-constrained hosts: a 4096² working set is ~1.3 GiB per variant worker. At the cap the session warns and keeps computing (moves still allowed). |

## Commands

```sh
# backend tests (GPU tests auto-skip without cupy/CUDA)
cd backend && .venv/bin/pytest

# live-server streaming smoke test (no browser)
.venv/bin/uvicorn main:app --port 8010 --ws-per-message-deflate false &
.venv/bin/python scripts/ws_smoke.py

# throughput benchmark
.venv/bin/python scripts/bench.py [cpu] [cuda:1]

# frontend: decoder golden test + typecheck + build
cd frontend && npm run test && npm run build

# mp4 export needs the system ffmpeg (libx264); its tests skip without it
ffmpeg -version

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
`window.__wfPerf.snapshot()/reset()` (lib/perf.ts) exposes frame-pipeline
counters: received/painted rates, MiB/s, queue drops, per-stage avg ms
(decode/upload/draw/plots/fanout), the GL renderer string (SwiftShader
here = software rendering, the classic cause of few-fps playback at large
grids) and the measured refresh interval.

## Roadmap (v2, agreed 2026-07-19)

1. **Destructive forking**: resume computation from ANY record (end or
   intermediate; the abandoned branch is discarded), both modes. Requires
   periodic float64 checkpoints alongside the uint16 display history — a
   quantized frame must NEVER seed a propagator. "Continue past t2" is the
   fork-at-the-end special case.
2. **Save/load the whole simulation** to disk (config + history +
   checkpoints; own format, no legacy compatibility).
3. ~~Multi-GPU~~ — DONE 2026-07-19: variant workers spread across the
   `WIGNERF_DEVICE` pool (see GPU section); measured +41% (4 variants)
   and +39% (2 variants) at 1024² on the 3090 + 2080 Ti pair.
4. ~~mp4 export~~ — DONE 2026-07-21: backend-rendered export of any
   computed range (see the mp4 export bullet above). Later: Lindblad
   dissipation (the propagator's exponent construction is deliberately
   modular for it), multi-D.

## Conventions / gotchas

- Do not reference the old project website domain anywhere in wignerf —
  it expired (old code/comments elsewhere in the repo may keep theirs).
- Nx, Np must be even (shader unshift + fftshift symmetry); powers of 2
  for FFT speed. Grid warns, API schema enforces evenness.
- Physics invariants in `tests/test_propagator.py` are the correctness
  anchor — harmonic quantum ≡ classical (Moyal terms vanish for quadratic
  H) is the strongest single check; run them after touching propagator,
  grid or fftshift bookkeeping.
- **Always run uvicorn with `--ws-per-message-deflate false`** (start.sh
  does). uvicorn's default permessage-deflate zlib-compresses every
  multi-MiB frame bundle on the asyncio event loop and caps the stream at
  ~10-25 records/s — measured 12x slower than uncompressed on localhost
  (browsers silently negotiate the extension, so the slowdown looks like
  a rendering problem; `__wfPerf` showing tiny stage times with a low
  received_per_s is the tell).
- pyFFTW plans are per-`ArrayBackend`-instance and must not be shared
  across threads; each worker owns its backend.
- Relativistic variants: mc² cancels inside the propagator; observables
  subtract it from displayed E.
- **Secular E drift + slow purity decay = boundary wrap, not a solver
  bug.** The spectral domain is a torus: when a state's orbit + ~5σ tails
  reach the x or p edge, mass wraps through the seam and the run faithfully
  evolves the WRONG (torus) problem. Tells: IC norm deficit >> 1e-6, the
  4σ edge warning, secular (not oscillatory-bounded) drifts. Fix: enlarge
  the domain — or enable auto-expand, which detects the approach (edge-band
  mass of the total sampled W, also checked at IC-preview time — the
  per-component 4σ boxes alone miss interference terms) and regrids
  exactly before mass wraps. Verified: same cat state, [-6,6]x[-7,7] gives E drift 2e-3;
  [-12,12]² gives 4e-6 with purity conserved to 5e-12 — the discrete map
  is exactly unitary for contained states (healthy E behavior is a BOUNDED
  O(dt²) oscillation from Strang splitting, never a drift).
- **Growing ΔX·ΔP in the RELATIVISTIC variants only = anharmonic shear, not
  a bug.** T = c√(p²+m²c²) carries a −p⁴/(8m³c²) term, so ω depends on E
  (δω = −3E/(8c²)) and the ensemble shears at k = t·r²·3/(8c²). The shear is
  symplectic: purity and det C are conserved and the LOWER envelope of ΔX·ΔP
  stays exactly at ħ/2, while the upper one grows ∝ t² (modulated at 2ω).
  Tells that it is physics: halving dt leaves it identical while the E(t)
  splitting oscillation drops 4×, it scales as 1/c⁴, purity stays flat.
  Non-relativistic harmonic H is exactly quadratic ⇒ no shear ⇒ flat.
  Measured: coherent state at (2,0) in x²/2 with c = 137.036 → 2e-5 at
  t = 100 (analytic σ²k²/2 = 1.6e-5). Pinned by
  `test_relativistic_uncertainty_shear`.

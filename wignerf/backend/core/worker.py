"""
SolverWorker: one thread per ticked variant. Owns its own ArrayBackend
(FFT plans and CUDA context are per-thread), integrates with its own
adaptive dt, and lands exactly on every record time tau_k of the session
clock by clamping its final substep. Emits quantized VariantFrames into the
session's FrameHistory and signals the asyncio streamer.

Commands arrive on cmd_q as dicts:
  {"kind": "params", "cp": CompiledPotential|None, "mass": ..., "c": ...,
   "hbar_eff": ..., "tol": ...}   (absent keys = unchanged)
"""

import logging
import queue
import threading
import traceback
from time import monotonic

from . import initial, observables
from .propagator import Propagator
from .protocol import VARIANTS, VariantFrame, variant_id
from .quantize import quantize
from .xp import ArrayBackend

log = logging.getLogger(__name__)

_EXP_CACHE_MAX = 8


class SolverWorker(threading.Thread):
    def __init__(self, session, key, slot):
        super().__init__(daemon=True, name="wignerf-%s-%s" % (session.id, key))
        self.session = session
        self.key = key
        self.slot = slot
        self.flavor = VARIANTS[key]
        self.cmd_q = queue.Queue()
        self.stop_evt = threading.Event()
        self.force_adjust = True
        self.dt = 0.0
        self.steps_total = 0
        self.steps_per_sec = 0.0
        self._exp_cache = {}
        self._rate_mark = (0, monotonic())

    def stop(self):
        self.stop_evt.set()
        self.session.clock.kick()

    # -- thread body --------------------------------------------------------

    def run(self):
        try:
            self._run()
        except Exception as e:
            log.exception("worker %s failed", self.name)
            self.session.post_error("variant '%s' solver died: %s" % (self.key, e),
                                    detail=traceback.format_exc())
        finally:
            self._release_gpu_pool()

    def _release_gpu_pool(self):
        """Return unused CuPy pool blocks to the driver when the session
        closes. CuPy pools freed memory per process, so nvidia-smi keeps
        showing it as 'used' even when idle — releasing here means closed
        sessions visibly give their VRAM back. The pool is per-device, so
        re-enter THIS worker's device; blocks still referenced by other
        live sessions are untouched (free_all_blocks frees only free
        blocks)."""
        backend = getattr(self, "_backend", None)
        if backend is None or not backend.is_gpu:
            return
        try:
            with backend.device():
                xp = backend.xp
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            log.debug("GPU pool release failed", exc_info=True)

    def _run(self):
        cfg = self.session.cfg
        backend = ArrayBackend(device=self.session.device,
                               fft_threads=self.session.fft_threads)
        self._backend = backend
        with backend.device():
            g, Wnat, _ = initial.from_spec(cfg.grid, cfg.ic, cfg.hbar_eff, backend)
            U, dUdx = self.session.compiled_potential.for_backend(backend)
            prop = Propagator(g, mass=cfg.mass, c=cfg.c, hbar_eff=cfg.hbar_eff,
                              tol=cfg.tol, U=U, dUdx=dUdx, **self.flavor)
            W = g.shift2d(Wnat)
            t = cfg.t1
            self.dt = cfg.record_dt/8.
            self._emit(0, t, W, prop, backend)      # record 0 = the Cauchy data
            frontier = 0
            while not self.stop_evt.is_set():
                self._drain_commands(prop, backend)
                tgt = self.session.clock.next_target(frontier)
                if tgt is None:
                    self.session.clock.wait_work(0.1)
                    continue
                k, t_tgt = tgt
                W, t = self._advance(prop, W, t, t_tgt)
                self._emit(k, t_tgt, W, prop, backend)
                frontier = k

    # -- stepping -----------------------------------------------------------

    def _exponents(self, prop, dts):
        pair = self._exp_cache.get(dts)
        if pair is None:
            if len(self._exp_cache) >= _EXP_CACHE_MAX:
                self._exp_cache.clear()
            pair = prop.exponents(dts)
            self._exp_cache[dts] = pair
        return pair

    def _advance(self, prop, W, t, t_tgt):
        eps = 1e-12*max(1.0, abs(t_tgt))
        while abs(t_tgt - t) > eps and not self.stop_evt.is_set():
            direction = 1.0 if t_tgt > t else -1.0
            if self.dt == 0.0 or (self.dt > 0) != (direction > 0):
                self.dt = direction*(abs(self.dt) or self.session.cfg.record_dt/8.)
                self._exp_cache.clear()
                self.force_adjust = True
            rem = t_tgt - t
            adjust_due = self.force_adjust or self.steps_total % 20 == 0
            if adjust_due and abs(self.dt) <= abs(rem):
                W, self.dt, eU, eT = prop.adjust_step(self.dt, W)
                self._exp_cache = {self.dt: (eU, eT)}
                self.force_adjust = False
                t += self.dt
            else:
                dts = direction*min(abs(self.dt), abs(rem))
                W = prop.solve_spectral(W, *self._exponents(prop, dts))
                t += dts
            self.steps_total += 1
        return W, t_tgt    # land exactly on the record time (no drift)

    def _emit(self, k, t, W, prop, backend):
        wq, wmin, wmax = quantize(W, backend)
        obs = observables.compute(W, prop)
        vf = VariantFrame(vid=variant_id(**self.flavor), wq=wq,
                          wmin=wmin, wmax=wmax, E=obs.E,
                          x_mean=obs.x_mean, x_std=obs.x_std,
                          p_mean=obs.p_mean, p_std=obs.p_std,
                          purity=obs.purity,
                          dt=self.dt, rho=obs.rho, phi=obs.phi)
        self.session.history.put(k, t, self.slot, vf)
        self.session.notify_frame()
        n, mark = self._rate_mark
        now = monotonic()
        if now - mark > 1.0:
            self.steps_per_sec = (self.steps_total - n)/(now - mark)
            self._rate_mark = (self.steps_total, now)

    # -- commands -----------------------------------------------------------

    def _drain_commands(self, prop, backend):
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            if cmd.get("kind") != "params":
                continue
            prev = dict(U=prop.U, dUdx=prop.dUdx, mass=prop.mass, c=prop.c,
                        hbar_eff=prop.hbar_eff, tol=prop.tol)
            try:
                kwargs = {}
                if cmd.get("cp") is not None:
                    U, dUdx = cmd["cp"].for_backend(backend)
                    kwargs.update(U=U, dUdx=dUdx)
                for f in ("mass", "c", "hbar_eff", "tol"):
                    if cmd.get(f) is not None:
                        kwargs[f] = cmd[f]
                prop.set_physics(**kwargs)
                xp = backend.xp
                if not (bool(xp.isfinite(prop.dU).all())
                        and bool(xp.isfinite(prop.dT).all())):
                    raise ValueError("non-finite propagator exponents")
            except Exception as e:
                prop.set_physics(**prev)   # roll back, keep evolving
                self.session.post_error(
                    "variant '%s': parameter change rejected (%s)" % (self.key, e))
            else:
                self._exp_cache.clear()
                self.force_adjust = True

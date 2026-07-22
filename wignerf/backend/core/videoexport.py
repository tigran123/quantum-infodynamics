"""
mp4 export of an already-computed record range: a background thread reads
records out of the session's FrameHistory, renders each one with
core.render_mpl and pipes the raw RGBA Agg buffer into
`ffmpeg -c:v libx264`.

Deliberately NOT a live recorder: export is a PAUSED-only action on history
that already exists. That is not just a scope decision — a running session
evicts its oldest records once the byte cap is reached, and an export
reading behind the frontier would lose them mid-file.

Two passes over the range:
  1. scan  — collects the E/ΔX·ΔP/γ series, the per-variant fixed colour
             scale, the fixed marginal amplitudes and the widest window any
             record used (all cheap scalars already stored in the records),
             and proves every record is still retained BEFORE ffmpeg is
             spawned. Only VALUE scales are export-wide: the spatial axes
             follow each record's own geometry (render_mpl._apply_geom), so
             a frame from before an auto-expansion still fills its panel;
  2. render — one figure update + one stdin write per frame.

This module must not import core.session (the session imports it back for
cleanup); the session object is duck-typed here.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from time import monotonic

from . import describe, render_mpl

log = logging.getLogger(__name__)

# how long a finished file stays downloadable before the sweeper unlinks it
FILE_TTL = 30*60.0
PROGRESS_PERIOD = 0.5

_JOBS = {}
_LOCK = threading.Lock()
# Matplotlib guarantees nothing about two figures rendering in parallel
# threads (shared font manager and image caches), and two exports would
# fight for the same cores anyway — so renders are serialized process-wide.
# A job waiting here honestly reports "queued".
_RENDER_LOCK = threading.Lock()


def ffmpeg_path():
    return shutil.which("ffmpeg")


class ExportJob(threading.Thread):
    def __init__(self, session, spec, k0, k1, outdir):
        super().__init__(daemon=True, name="wignerf-export-%s" % session.id)
        self.id = uuid.uuid4().hex[:12]
        self.session = session
        self.spec = spec
        self.k0, self.k1 = int(k0), int(k1)
        self.records = list(range(self.k0, self.k1 + 1, spec.stride))
        self.variants = list(spec.variants or session.cfg.variants)
        # On-disk name stays collision-proof (session + job id); the name the
        # BROWSER saves is the readable one below — two exports of the same
        # range in the same minute must not overwrite each other's file
        # while one of them is being downloaded.
        self.path = os.path.join(outdir, "wignerf-%s-%s.mp4"
                                 % (session.id, self.id))
        self.download_name = "wignerf-%s-%drec%s-%dx%d-%s.mp4" % (
            "-".join(v.upper() for v in self.variants),
            len(self.records),
            "" if spec.stride == 1 else "-every%d" % spec.stride,
            spec.width, spec.height,
            time.strftime("%Y%m%d-%H%M"))
        self.state = "queued"      # queued|running|done|error|cancelled
        self.done = 0
        self.total = len(self.records)
        self.error = None
        self.finished_at = None
        self.cancel_evt = threading.Event()

    # -- status -------------------------------------------------------------

    def status(self):
        return {"job_id": self.id, "session_id": self.session.id,
                "state": self.state, "done": self.done, "total": self.total,
                "bytes": (os.path.getsize(self.path)
                          if self.state == "done" and os.path.exists(self.path)
                          else 0),
                "error": self.error,
                "filename": self.download_name,
                "fps": self.spec.fps,
                "duration_s": self.total/float(self.spec.fps)}

    def _post(self):
        d = dict(self.status())
        d["type"] = "export"
        self.session.post_msg(d)

    def cancel(self):
        self.cancel_evt.set()

    # -- thread body --------------------------------------------------------

    def run(self):
        with _RENDER_LOCK:
            self._run()

    def _run(self):
        if self.cancel_evt.is_set():        # cancelled while queued
            self.state = "cancelled"
            self.finished_at = time.monotonic()
            self._post()
            return
        self.state = "running"
        self._post()
        fig = None
        proc = None
        try:
            stats, geom0 = self._scan()
            meta = render_mpl.meta_columns(
                self.session.cfg, geom0, stats, self.variants, self.k0,
                self.k1, self.total, self.spec.fps, self.session.param_log)
            fig = render_mpl.FrameFigure(self.variants, stats, meta,
                                         width=self.spec.width,
                                         height=self.spec.height,
                                         show_grid=self.spec.show_grid)
            proc = self._spawn_ffmpeg()
            last = 0.0
            for k in self.records:
                if self.cancel_evt.is_set():
                    raise _Cancelled()
                rec = self.session.history.get(k)
                if rec is None:
                    raise ValueError("record %d is no longer retained "
                                     "(history evicted)" % k)
                t, geom, vframes = rec
                try:
                    proc.stdin.write(fig.update(k, t, geom,
                                                self._order(vframes),
                                                self.k0, self.k1))
                except BrokenPipeError:
                    # ffmpeg died mid-stream (its diagnostics went to the
                    # server log); report that, not "broken pipe"
                    raise ValueError("ffmpeg exited early with code %s"
                                     % proc.wait(timeout=10)) from None
                self.done += 1
                now = monotonic()
                if now - last > PROGRESS_PERIOD:
                    last = now
                    self._post()
            proc.stdin.close()
            rc = proc.wait(timeout=120)
            proc = None
            if rc != 0:
                raise ValueError("ffmpeg exited with code %d" % rc)
            self.state = "done"
        except _Cancelled:
            self.state = "cancelled"
            self._unlink()
        except Exception as e:
            log.exception("export job %s failed", self.id)
            self.state = "error"
            self.error = str(e)
            self._unlink()
        finally:
            if proc is not None:
                _kill(proc)
            if fig is not None:
                fig.close()
            self.finished_at = time.monotonic()
            self._post()

    def _order(self, vframes):
        """Records carry every session variant in bundle order; an export of
        a subset picks its own, keeping the requested order."""
        by_key = {render_mpl.key_of_vid(vf.vid): vf for vf in vframes}
        return [by_key[k] for k in self.variants]

    def _scan(self):
        """Pass 1: series + fixed colour scales + the widest window (quoted
        in the metadata block; the plots follow each record)."""
        st = render_mpl.RangeStats()
        for key in self.variants:
            st.E[key], st.uncert[key], st.purity[key] = [], [], []
            st.scale[key] = 0.0
        x1 = p1 = float("inf")
        x2 = p2 = float("-inf")
        geom0 = None
        for k in self.records:
            if self.cancel_evt.is_set():
                raise _Cancelled()
            rec = self.session.history.get(k)
            if rec is None:
                raise ValueError("record %d is not retained (evicted, or the "
                                 "range is outside the computed history)" % k)
            t, geom, vframes = rec
            if geom0 is None:
                geom0 = geom
            st.t.append(t)
            x1, x2 = min(x1, geom.x1), max(x2, geom.x2)
            p1, p2 = min(p1, geom.p1), max(p2, geom.p2)
            for key, vf in zip(self.variants, self._order(vframes)):
                st.E[key].append(vf.E)
                st.uncert[key].append(vf.x_std*vf.p_std)
                st.purity[key].append(vf.purity)
                st.scale[key] = max(st.scale[key], vf.wmax, -vf.wmin)
                st.rho_max = max(st.rho_max, float(vf.rho.max()))
                st.phi_max = max(st.phi_max, float(vf.phi.max()))
        if geom0 is None:
            raise ValueError("no records in the requested range")
        for key in self.variants:
            if st.scale[key] <= 0.0:
                st.scale[key] = 1e-30
        st.x1, st.x2, st.p1, st.p2 = x1, x2, p1, p2
        return st, geom0

    def _spawn_ffmpeg(self):
        cfg = self.session.cfg
        comment = describe.config_json(
            cfg, self.session.param_log, at_record=self.k0,
            export={"records": [self.k0, self.k1], "stride": self.spec.stride,
                    "fps": self.spec.fps, "frames": self.total,
                    "variants": self.variants})
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgba",
               "-video_size", "%dx%d" % (self.spec.width, self.spec.height),
               "-framerate", str(self.spec.fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-metadata", "title=wignerf W(x,p,t) records %d-%d"
               % (self.k0, self.k1),
               "-metadata", "comment=%s" % comment,
               self.path]
        log.info("export %s: %d frames -> %s", self.id, self.total, self.path)
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _unlink(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def cleanup(self):
        self.cancel()
        self._unlink()


class _Cancelled(Exception):
    pass


def _kill(proc):
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def start(session, spec, k0, k1, outdir):
    os.makedirs(outdir, exist_ok=True)
    job = ExportJob(session, spec, k0, k1, outdir)
    with _LOCK:
        _JOBS[job.id] = job
    job.start()
    return job


def get(job_id):
    with _LOCK:
        return _JOBS.get(job_id)


def active_for(session_id):
    """The session's unfinished job, if any (one export at a time)."""
    with _LOCK:
        for j in _JOBS.values():
            if j.session.id == session_id and j.state in ("queued", "running"):
                return j
    return None


def drop(job_id):
    with _LOCK:
        job = _JOBS.pop(job_id, None)
    if job is not None:
        job.cleanup()
    return job


def close_session(session_id):
    """Cancel and clean every job of a session that is going away."""
    with _LOCK:
        ids = [j.id for j in _JOBS.values() if j.session.id == session_id]
    for jid in ids:
        drop(jid)


def sweep(now=None):
    """Drop finished jobs whose file has outlived FILE_TTL (called from the
    session TTL sweeper)."""
    now = time.monotonic() if now is None else now
    with _LOCK:
        stale = [j.id for j in _JOBS.values()
                 if j.finished_at is not None and now - j.finished_at > FILE_TTL]
    for jid in stale:
        drop(jid)


def close_all():
    with _LOCK:
        ids = list(_JOBS)
    for jid in ids:
        drop(jid)


def probe_json(path):
    """ffprobe helper (tests/diagnostics): stream info of an exported file."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    out = subprocess.run([exe, "-v", "error", "-print_format", "json",
                          "-show_streams", path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)

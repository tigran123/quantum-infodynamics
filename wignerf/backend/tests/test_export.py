"""mp4 export: the analytic description, the frame renderer and the job."""

import json
import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import describe, protocol, videoexport
from core.render_mpl import FrameFigure, RangeStats, meta_columns
from main import app

GRID = dict(x1=-6.0, x2=6.0, Nx=64, p1=-7.0, p2=7.0, Np=64)
IC = {"type": "mixture",
      "components": [{"x0": 2.0, "p0": 0.0, "sigma_x": 0.707, "sigma_p": 0.707}]}

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="ffmpeg is not installed")


# ---------------------------------------------------------------------------
# describe.py — the "how to reproduce this" text
# ---------------------------------------------------------------------------

def test_ic_expression_mixture_substitutes_numbers():
    ic = protocol.ICSpec(type="mixture", components=[
        {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5, "sigma_p": 1.0},
        {"x0": -2.0, "p0": 1.0, "sigma_x": 0.5, "sigma_p": 1.0, "weight": 3.0},
    ])
    text = " ".join(describe.ic_expression(ic, 1.0))
    assert "W(x,p,0)" in text
    assert "(x − 2)" in text and "(x + 2)" in text   # never "x−−2"
    assert "(p − 1)" in text
    # amplitudes carry the normalized weights: 1/4 and 3/4 over 2*pi*sx*sp
    assert "%.6g" % (0.25/(2*np.pi*0.5*1.0)) in text
    assert "%.6g" % (0.75/(2*np.pi*0.5*1.0)) in text


def test_ic_expression_cat_gives_psi_and_derived_sigma_p():
    ic = protocol.ICSpec(type="cat", components=[
        {"x0": -2.0, "p0": 0.0, "sigma_x": 0.5},
        {"x0": 2.0, "p0": 0.0, "sigma_x": 0.5, "phase": 3.14159},
    ])
    lines = describe.ic_expression(ic, 2.0)
    text = " ".join(lines)
    assert "ψ(x,0)" in text and "Wigner[ψ]" in text
    assert "e^(i3.14159)" in text
    # sigma_p is DERIVED for cat states: hbar/(2 sigma_x) = 2/(2*0.5) = 2
    assert "σp = 2" in text


def test_param_lines_report_live_changes_in_range():
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    log = [{"at_record": 5, "applied": {"U": "x^4/4"}},
           {"at_record": 900, "applied": {"mass": 2.0}}]
    text = " ".join(describe.param_lines(cfg, log, 0, 100))
    assert "U(x) = x^2/2" in text
    assert "live change at record 5" in text and "x^4/4" in text
    assert "record 900" not in text          # outside the exported range
    blob = json.loads(describe.config_json(cfg, log, export={"frames": 3}))
    assert blob["config"]["potential"] == "x^2/2"
    assert blob["param_log"][1]["applied"]["mass"] == 2.0
    assert blob["export"]["frames"] == 3


# ---------------------------------------------------------------------------
# render_mpl.py — the figure
# ---------------------------------------------------------------------------

def _vframe(seed, Nx=32, Np=32):
    rng = np.random.default_rng(seed)
    wq = (rng.random((Nx, Np))*65535).astype(np.uint16)
    return protocol.VariantFrame(
        vid=protocol.variant_id(True, False), wq=wq, wmin=-0.1, wmax=0.3,
        E=1.0 + seed, x_mean=0.1, x_std=0.7, p_mean=0.0, p_std=0.7,
        purity=1.0, dt=1e-3,
        rho=rng.random(Nx).astype("f4"), phi=rng.random(Np).astype("f4"))


def _stats(n=3):
    st = RangeStats(t=[0.0, 0.05, 0.1][:n], rho_max=1.0, phi_max=1.0,
                    x1=-6.0, x2=6.0, p1=-7.0, p2=7.0)
    st.E["qn"] = [1.0, 1.1, 1.2][:n]
    st.uncert["qn"] = [0.5]*n
    st.purity["qn"] = [1.0]*n
    st.scale["qn"] = 0.3
    return st


def test_video_labels_match_the_ui():
    """The video must NAME things as the SPA does: SeriesPlot.vue's titles
    verbatim (γ keeps 2πℏ∬W²dxdp, not the equivalent Tr ρ²), the Setup
    panel's ℏ, and the mode select's "run-ahead" — not the wire value."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"], mode="runahead", t2=5.0)
    text = " ".join(describe.param_lines(cfg))
    assert "ℏ = 1" in text and "ℏ_eff" not in text
    assert "mode = run-ahead" in text and "runahead" not in text

    geom = protocol.RecordGeom(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    fig = FrameFigure(["qn"], stats, meta_columns(cfg, geom, stats, ["qn"],
                                                  0, 2, 3, 30),
                      width=640, height=360)
    try:
        # the series plots carry their title at loc='right' (the axis
        # offset label owns the top-left corner)
        titles = [ax.get_title(loc=where) for ax in fig.fig.axes
                  for where in ("center", "right")]
    finally:
        fig.close()
    assert "purity γ(t) = 2πℏ∬W²dxdp" in titles
    assert "E(t)" in titles and "ΔX·ΔP(t)" in titles
    assert "ρ(x) = ∫W dp" in titles and "φ(p) = ∫W dx" in titles


def test_series_ylim_matches_uplot_rule():
    """SeriesPlot.vue: pad = max(15% of span, 1e-4 of |max|, 1e-12). A
    purity series that drifts 2e-5 must therefore keep the UI's ±1e-4
    window (a flat-looking line), not matplotlib's tight autoscale."""
    from core.render_mpl import series_ylim
    lo, hi = series_ylim([1.0 - 2e-5*i/100 for i in range(101)])
    assert hi == pytest.approx(1.0 + 1e-4, rel=1e-9)
    assert lo == pytest.approx(1.0 - 2e-5 - 1e-4, rel=1e-9)
    # a genuinely large span falls back to the 15% padding
    lo, hi = series_ylim([0.0, 10.0])
    assert (lo, hi) == pytest.approx((-1.5, 11.5))
    assert series_ylim([]) == (0.0, 1.0)


def test_show_grid_covers_charts_and_w_panels():
    """One setting, every plot: the SPA's "grid lines on plots" toggle must
    reach the W heatmaps too — matplotlib draws the axes grid UNDER the
    image, so the panels need their own lines on top (they had none)."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    out = {}
    for flag in (True, False):
        fig = FrameFigure(["qn"], stats, meta, width=320, height=240,
                          show_grid=flag)
        try:
            panel_ax = fig.images[0][0]
            chart_ax = [a for a in fig.fig.axes
                        if a.get_title(loc="right") == "E(t)"][0]
            out[flag] = (len(panel_ax.lines),
                         any(g.get_visible() for g in chart_ax.get_xgridlines()),
                         bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2)))
        finally:
            fig.close()
    on, off = out[True], out[False]
    assert on[0] > 0 and off[0] == 0, "W panel grid lines ignored the toggle"
    assert on[1] and not off[1], "chart grid ignored the toggle"
    assert on[2] != off[2], "the rendered frames are identical"


def test_frame_figure_renders_distinct_frames():
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    fig = FrameFigure(["qn"], stats, meta, width=640, height=360)
    try:
        # update() hands back a view of the Agg buffer (RGBA, fed straight
        # to ffmpeg): copy before comparing, the next update overwrites it
        a = bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2))
        b = bytes(fig.update(1, 0.05, geom, [_vframe(2)], 0, 2))
        assert len(a) == 640*360*4 and len(b) == len(a)
        assert a != b, "the frame did not change between records"
        # a regrid mid-video must be accepted (per-record geometry)
        big = protocol.RecordGeom(64, 32, -12.0, 12.0, -7.0, 7.0)
        c = bytes(fig.update(2, 0.1, big, [_vframe(3, Nx=64)], 0, 2))
        assert len(c) == len(a)
    finally:
        fig.close()


# ---------------------------------------------------------------------------
# the job, end to end
# ---------------------------------------------------------------------------

def _mk(client, **over):
    cfg = {"grid": GRID, "potential": "x^2/2", "ic": IC,
           "variants": ["qn", "cn"], "record_dt": 0.05, "delay": 0.0}
    cfg.update(over)
    r = client.post("/api/sessions", json=cfg)
    assert r.status_code == 200, r.text
    return r.json()


def _solve_a_few(client, ws, sid, n=6):
    import json as _json
    ws.send_text(_json.dumps({"type": "play"}))
    for _ in range(200):
        time.sleep(0.05)
        if client.get("/api/sessions/%s" % sid).json()["record_extent"][1] >= n:
            break
    ws.send_text(_json.dumps({"type": "pause"}))
    time.sleep(0.3)                     # let in-flight records land
    return client.get("/api/sessions/%s" % sid).json()["record_extent"]


@needs_ffmpeg
def test_export_end_to_end(tmp_path, monkeypatch):
    import config as appconfig
    monkeypatch.setattr(appconfig, "EXPORT_DIR", str(tmp_path))
    with TestClient(app) as client:
        info = _mk(client)
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            first, last = _solve_a_few(client, ws, sid)
            k1 = min(last, first + 5)
            r = client.post("/api/sessions/%s/export" % sid,
                            json={"k0": first, "k1": k1, "fps": 10,
                                  "width": 640, "height": 360})
            assert r.status_code == 202, r.text
            job = r.json()
            assert job["total"] == k1 - first + 1
            jid = job["job_id"]
            for _ in range(600):
                time.sleep(0.1)
                st = client.get("/api/exports/%s" % jid).json()
                if st["state"] in ("done", "error", "cancelled"):
                    break
            assert st["state"] == "done", st
            assert st["done"] == st["total"] and st["bytes"] > 0

            f = client.get("/api/exports/%s/file" % jid)
            assert f.status_code == 200
            assert f.headers["content-type"] == "video/mp4"
            assert len(f.content) == st["bytes"]

            path = videoexport.get(jid).path
            info_ = videoexport.probe_json(path)
            if info_ is not None:                 # ffprobe available
                v = info_["streams"][0]
                assert v["codec_name"] == "h264"
                assert int(v["nb_frames"]) == st["total"]

            assert client.delete("/api/exports/%s" % jid).json()["ok"]
            assert client.get("/api/exports/%s" % jid).status_code == 404
            assert not (tmp_path / "").exists() or not list(tmp_path.glob("*.mp4"))
        client.delete("/api/sessions/%s" % sid)


@needs_ffmpeg
def test_export_rejected_while_running_and_on_bad_range(tmp_path, monkeypatch):
    import json as _json
    import config as appconfig
    monkeypatch.setattr(appconfig, "EXPORT_DIR", str(tmp_path))
    with TestClient(app) as client:
        info = _mk(client, variants=["qn"])
        sid = info["session_id"]
        with client.websocket_connect(info["ws_url"]) as ws:
            ws.send_text(_json.dumps({"type": "play"}))
            time.sleep(0.3)
            r = client.post("/api/sessions/%s/export" % sid, json={})
            assert r.status_code == 409 and "pause" in r.text
            ws.send_text(_json.dumps({"type": "pause"}))
            time.sleep(0.3)
            # k1 < k0 is a schema error; a range past the frontier clamps
            assert client.post("/api/sessions/%s/export" % sid,
                               json={"k0": 5, "k1": 2}).status_code == 422
            assert client.post("/api/sessions/%s/export" % sid,
                               json={"variants": ["cr"]}).status_code == 422
            r = client.post("/api/sessions/%s/export" % sid,
                            json={"k0": 0, "k1": 10**6, "fps": 10,
                                  "width": 320, "height": 240})
            assert r.status_code == 202, r.text
            jid = r.json()["job_id"]
            # one export at a time per session
            assert client.post("/api/sessions/%s/export" % sid,
                               json={}).status_code in (409, 202)
            for _ in range(600):
                time.sleep(0.1)
                if client.get("/api/exports/%s" % jid).json()["state"] != "running":
                    break
            client.delete("/api/exports/%s" % jid)
        client.delete("/api/sessions/%s" % sid)
        # closing the session drops its jobs and their files
        assert not list(tmp_path.glob("*.mp4"))


def test_download_name_is_descriptive():
    """The browser must save something readable — variants, records, size,
    time — while the on-disk path keeps its collision-proof ids (two
    exports of the same range in one minute must not overwrite each other,
    least of all while one is being downloaded)."""
    class _Sess:      # duck-typed: ExportJob only needs .id and .cfg here
        id = "0123456789ab"
        cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                     variants=["qn", "cn"])
    spec = protocol.ExportSpec(fps=25, width=3840, height=2160)
    job = videoexport.ExportJob(_Sess(), spec, 0, 99, "/tmp")
    assert job.download_name.startswith("wignerf-QN-CN-100rec-3840x2160-")
    assert job.download_name.endswith(".mp4")
    assert "0123456789ab" not in job.download_name
    assert job.id in job.path            # on disk: still unique per job
    assert job.status()["filename"] == job.download_name
    # a strided export says so, and counts the frames it actually renders
    spec = protocol.ExportSpec(stride=4, width=1920, height=1080)
    job = videoexport.ExportJob(_Sess(), spec, 0, 99, "/tmp")
    assert "-25rec-every4-1920x1080-" in job.download_name


def test_layout_is_resolution_independent():
    """Fonts are in POINTS: a fixed dpi would render every label at half
    its relative size at 4K, so the figure keeps a constant inch size and
    the dpi carries the resolution."""
    cfg = protocol.SessionCreate(grid=GRID, potential="x^2/2", ic=IC,
                                 variants=["qn"])
    geom = protocol.RecordGeom(32, 32, -6.0, 6.0, -7.0, 7.0)
    stats = _stats()
    meta = meta_columns(cfg, geom, stats, ["qn"], 0, 2, 3, 30)
    sizes = {}
    for w, h in ((1920, 1080), (3840, 2160)):
        fig = FrameFigure(["qn"], stats, meta, width=w, height=h)
        try:
            sizes[w] = (fig.fig.get_size_inches().tolist(),
                        len(bytes(fig.update(0, 0.0, geom, [_vframe(1)], 0, 2))))
        finally:
            fig.close()
    assert sizes[1920][0] == pytest.approx(sizes[3840][0])   # same layout
    assert sizes[1920][1] == 1920*1080*4
    assert sizes[3840][1] == 3840*2160*4


def test_export_unknown_session_and_job():
    with TestClient(app) as client:
        assert client.post("/api/sessions/nope/export", json={}).status_code == 404
        assert client.get("/api/exports/nope").status_code == 404
        assert client.get("/api/exports/nope/file").status_code == 404
        assert client.delete("/api/exports/nope").status_code == 404

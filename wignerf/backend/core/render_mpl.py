"""
Video-frame renderer for the mp4 export: one matplotlib (Agg) figure that
reproduces the SPA's whole view — tiled W(x,p) panels per variant, the
rho(x)/phi(p) marginals, the E(t)/dX*dP(t)/gamma(t) series with a moving
time cursor — plus a metadata block (see core/describe.py) that makes an
exported frame self-contained.

The figure is built ONCE per export job and only its data is updated per
record (set_data / set_extent / set_text), which is what keeps a 1000-frame
export in the minutes range rather than the tens of minutes a fresh figure
per frame would cost.

Conventions taken from the live renderer (frontend/src/render/
WignerRenderer.ts) so the video and the screen read the same:
- W is dequantized as wmin + q*(wmax-wmin)/65535 and unshifted (records are
  fftshifted along both axes),
- the colour scale is the SYMMETRIC diverging one, W=0 at the centre of
  "bwr": vmin = -s, vmax = +s with s = max(Wmax, -Wmin) — here taken over
  the WHOLE exported range per variant, so the video does not flicker,
- variant colours and dash patterns mirror frontend/src/lib/variants.ts.
"""

import textwrap
from dataclasses import dataclass, field

import numpy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .protocol import VARIANTS

# mirrors frontend/src/lib/variants.ts (VARIANT_META)
VARIANT_STYLE = {
    "qn": ("Quantum, non-relativistic", "#38bdf8", (0, ())),
    "qr": ("Quantum, relativistic", "#a78bfa", (0, (12, 7))),
    "cn": ("Classical, non-relativistic", "#fbbf24", (0, (6, 6))),
    "cr": ("Classical, relativistic", "#34d399", (0, (2, 6))),
}

BG = "#0a0a0a"          # bg-neutral-950, as in the SPA
AXBG = "#171717"
FG = "#d4d4d4"
MUTED = "#a3a3a3"
GRIDC = "#3f3f46"       # uPlot grid stroke (SeriesPlot/MarginalsPlot)
# GridOverlay.vue's phase-space grid: rgba(120,120,120,.28), .55 at zero
PANEL_GRIDC = "#787878"
PANEL_GRID_ALPHA = 0.28
PANEL_ZERO_ALPHA = 0.55

AU_TIME_FS = 2.4188843265857e-2      # lib/units.ts
AU_ENERGY_EV = 27.211386245988


def key_of_vid(vid):
    return ("q" if vid & 1 else "c") + ("r" if vid & 2 else "n")


def dequantize_natural(vf):
    """Record W -> float32 (Np, Nx) in natural order, ready for imshow."""
    W = vf.wmin + vf.wq.astype(numpy.float32)*(
        numpy.float32((vf.wmax - vf.wmin)/65535.0))
    return numpy.fft.ifftshift(W, axes=(0, 1)).T


def axis_of(a1, a2, n):
    """Cell-centre-free natural axis, as MarginalsPlot.buildAxis does."""
    return a1 + numpy.arange(n)*((a2 - a1)/n)


@dataclass
class RangeStats:
    """What the scan pass over the exported records collects (see
    videoexport.ExportJob): the series to plot, the fixed colour scales and
    the axis limits that keep the video geometrically stable."""
    t: list = field(default_factory=list)
    E: dict = field(default_factory=dict)         # key -> list
    uncert: dict = field(default_factory=dict)
    purity: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)     # key -> symmetric W scale
    rho_max: float = 0.0
    phi_max: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


def _style_axes(ax, title=None, title_loc="center", grid=True):
    ax.set_facecolor(AXBG)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=7)
    # NB: grid(False, color=...) *enables* the grid (matplotlib treats the
    # line properties as a request), so the two cases must be separate
    if grid:
        ax.grid(True, color=GRIDC, linewidth=0.5, alpha=0.6)
    else:
        ax.grid(False)
    ax.yaxis.get_offset_text().set(color=MUTED, fontsize=6.5)
    if title:
        ax.set_title(title, color=FG, fontsize=8.5, pad=3, loc=title_loc)


def series_ylim(values):
    """The y-window uPlot uses for these plots in the SPA (SeriesPlot.vue's
    scales.y.range): pad = max(15% of the span, 1e-4 of the magnitude,
    1e-12). Reproduced EXACTLY here — matplotlib's own autoscale zooms onto
    the data span instead, which turned a purity drift the UI renders as a
    flat line at 1.000000 into a dramatic dive with a "×10⁻⁵+1" offset
    label. Same numbers, different picture; the video must read like the
    screen."""
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite:
        return (0.0, 1.0)
    mn, mx = min(finite), max(finite)
    pad = max((mx - mn)*0.15, abs(mx)*1e-4, 1e-12)
    return (mn - pad, mx + pad)


def _tick_decimals(splits):
    """SeriesPlot.vue's tick formatter: enough decimals to tell neighbouring
    ticks apart on a tightly-zoomed axis (default formatting prints '1')."""
    step = abs((splits[1] if len(splits) > 1 else 1) - splits[0]) or 1
    from math import ceil, log10
    return max(0, min(10, int(ceil(-log10(step))) + 1))


class FrameFigure:
    """Builds the figure once; `update()` returns the RGB bytes of a frame."""

    # The layout is defined at this width; every other resolution renders
    # the SAME figure at a different dpi (font sizes are in POINTS, so a
    # fixed dpi would shrink all text to half its relative size at 4K).
    REF_WIDTH = 1920.0

    def __init__(self, variants, stats, meta_lines, width=1920, height=1080,
                 show_grid=True):
        self.variants = list(variants)
        self.stats = stats
        # ONE grid setting for the whole frame, mirroring the SPA's "grid
        # lines on plots" checkbox: charts get uPlot's grid, the W panels
        # get GridOverlay.vue's (which is why they used to have none — the
        # heatmap is drawn over the axes grid)
        self.show_grid = bool(show_grid)
        self.width, self.height = int(width), int(height)
        dpi = 100.0*self.width/self.REF_WIDTH
        self.fig = Figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi,
                          facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        n = len(self.variants)
        rows, cols = (1, 1) if n <= 1 else (1, 2) if n == 2 else (2, 2)

        self.fig.text(0.012, 0.972, "wignerf — W(x, p, t)", color=FG,
                      fontsize=13, weight="bold", va="center")
        self.time_text = self.fig.text(0.30, 0.972, "", color="#38bdf8",
                                       fontsize=13, va="center")
        # right-anchored: the per-record geometry line is long and would run
        # off the canvas at 720p
        self.geom_text = self.fig.text(0.988, 0.972, "", color=MUTED,
                                       fontsize=9, va="center", ha="right")

        # ---- W panels (left block) + one colorbar each: the scales are
        # per-variant and fixed for the whole video, so a shared bar would
        # be wrong and a per-frame redraw unnecessary
        gs = self.fig.add_gridspec(rows, cols, left=0.045, right=0.60,
                                   bottom=0.235, top=0.935,
                                   wspace=0.30, hspace=0.34)
        self.images = []
        for i, key in enumerate(self.variants):
            ax = self.fig.add_subplot(gs[i//cols, i % cols])
            label, color, _ = VARIANT_STYLE[key]
            _style_axes(ax, grid=False)   # the panel grid is drawn on top
            ax.set_title(label, color=color, fontsize=9, pad=4)
            ax.set_xlabel("x (a₀)", color=MUTED, fontsize=8)
            ax.set_ylabel("p (a.u.)", color=MUTED, fontsize=8)
            s = stats.scale.get(key, 1.0)
            im = ax.imshow(numpy.zeros((2, 2), dtype=numpy.float32),
                           origin="lower", cmap="bwr", vmin=-s, vmax=s,
                           extent=(stats.x1, stats.x2, stats.p1, stats.p2),
                           aspect="auto", interpolation="antialiased")
            ax.set_xlim(stats.x1, stats.x2)
            ax.set_ylim(stats.p1, stats.p2)
            cb = self.fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.ax.tick_params(colors=MUTED, labelsize=6.5)
            cb.outline.set_edgecolor(GRIDC)
            self.images.append((ax, im))

        # ---- diagnostics column (right block), same order as PlotsColumn
        gsr = self.fig.add_gridspec(5, 1, left=0.675, right=0.965,
                                    bottom=0.235, top=0.935, hspace=0.75)
        self.rho_lines, self.phi_lines = {}, {}
        ax_rho = self.fig.add_subplot(gsr[0])
        _style_axes(ax_rho, "ρ(x) = ∫W dp", grid=self.show_grid)
        ax_phi = self.fig.add_subplot(gsr[1])
        _style_axes(ax_phi, "φ(p) = ∫W dx", grid=self.show_grid)
        for key in self.variants:
            _, color, dash = VARIANT_STYLE[key]
            self.rho_lines[key] = ax_rho.plot([], [], color=color,
                                              linestyle=dash, lw=1.2)[0]
            self.phi_lines[key] = ax_phi.plot([], [], color=color,
                                              linestyle=dash, lw=1.2)[0]
        ax_rho.set_xlim(stats.x1, stats.x2)
        ax_rho.set_ylim(min(0.0, -0.02*stats.rho_max), 1.08*stats.rho_max or 1)
        ax_phi.set_xlim(stats.p1, stats.p2)
        ax_phi.set_ylim(min(0.0, -0.02*stats.phi_max), 1.08*stats.phi_max or 1)

        # series: drawn ONCE for the whole exported range, a cursor moves
        self.cursors = []
        t = numpy.asarray(stats.t, dtype=float)
        # titles verbatim from SeriesPlot.vue's TITLES — the video must name
        # things exactly as the UI does, even where an equivalent form
        # (Tr ρ²) would be shorter
        for row, (title, data) in enumerate(
                (("E(t)", stats.E), ("ΔX·ΔP(t)", stats.uncert),
                 ("purity γ(t) = 2πℏ∬W²dxdp", stats.purity)), start=2):
            ax = self.fig.add_subplot(gsr[row])
            _style_axes(ax, title, title_loc="right", grid=self.show_grid)
            for key in self.variants:
                _, color, dash = VARIANT_STYLE[key]
                ax.plot(t, numpy.asarray(data[key], dtype=float), color=color,
                        linestyle=dash, lw=1.2)
            if t.size:
                ax.set_xlim(min(t[0], t[-1]), max(t[0], t[-1]))
            # same y-window and tick labels as the SPA (see series_ylim)
            ax.set_ylim(*series_ylim([v for key in self.variants
                                      for v in data[key]]))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _pos, ax=ax: "%.*f"
                % (_tick_decimals(list(ax.get_yticks())), v)))
            self.cursors.append(ax.axvline(t[0] if t.size else 0.0,
                                           color="#f472b6", lw=1.0, alpha=0.9))
        ax.set_xlabel("t (a.u.)", color=MUTED, fontsize=8)   # bottom plot only

        # ---- metadata block: everything needed to reproduce the run
        left, right = meta_lines
        self.fig.text(0.012, 0.185, "\n".join(left), color=MUTED, fontsize=8,
                      va="top", linespacing=1.6, family="DejaVu Sans")
        self.fig.text(0.335, 0.185, "\n".join(right), color=MUTED,
                      fontsize=8, va="top", linespacing=1.6,
                      family="DejaVu Sans")

        self._geom = None
        self._xax = self._pax = numpy.zeros(0)
        # W-panel grid: matplotlib draws the axes grid UNDER the image, so
        # these are explicit lines re-drawn after it every frame (see the
        # blitting note below). Positions come from the tick locator, which
        # is stable — the panel limits are the export-wide union.
        panel_grid = []
        if self.show_grid:
            for ax, _im in self.images:
                for v in ax.get_xticks():
                    if stats.x1 <= v <= stats.x2:
                        panel_grid.append(ax.axvline(
                            v, color=PANEL_GRIDC, lw=0.8,
                            alpha=(PANEL_ZERO_ALPHA if v == 0
                                   else PANEL_GRID_ALPHA)))
                for v in ax.get_yticks():
                    if stats.p1 <= v <= stats.p2:
                        panel_grid.append(ax.axhline(
                            v, color=PANEL_GRIDC, lw=0.8,
                            alpha=(PANEL_ZERO_ALPHA if v == 0
                                   else PANEL_GRID_ALPHA)))
        # Blitting: everything above is STATIC for the whole video (the
        # series are drawn once, the metadata never changes), so a full
        # redraw per frame would spend ~4/5 of its time on ticks, fonts and
        # curves nobody is animating. The dynamic artists are marked
        # animated, the static background is captured once, and each frame
        # restores it and re-draws only the images, marginals, cursors and
        # the two header texts.
        # order IS draw order: the panel grid comes after the images so it
        # stays visible over the heatmap, exactly as GridOverlay.vue sits
        # above the WebGL canvas in the SPA
        self._dynamic = ([im for _ax, im in self.images] + panel_grid
                         + list(self.rho_lines.values())
                         + list(self.phi_lines.values())
                         + self.cursors + [self.time_text, self.geom_text])
        for a in self._dynamic:
            a.set_animated(True)
        self.canvas.draw()      # static background, laid out once
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)

    # ------------------------------------------------------------------
    def update(self, k, t, geom, vframes, k0, k1):
        """Paint one record; returns its RGBA bytes (a view of the Agg
        buffer — write it to the encoder before the next update)."""
        self.time_text.set_text("t = %.4f a.u.  (%.4g fs)"
                                % (t, t*AU_TIME_FS))
        self.geom_text.set_text("record %d ∈ [%d, %d]   %d×%d   "
                                "x ∈ [%.4g, %.4g]  p ∈ [%.4g, %.4g]"
                                % (k, k0, k1, geom.Nx, geom.Np,
                                   geom.x1, geom.x2, geom.p1, geom.p2))
        new_geom = geom != self._geom
        if new_geom:
            # the domain is a PER-RECORD fact (auto-expand regrids); the axes
            # limits stay at the range union, only the images and the
            # marginal abscissae move
            self._geom = geom
            self._xax = axis_of(geom.x1, geom.x2, geom.Nx)
            self._pax = axis_of(geom.p1, geom.p2, geom.Np)
        for i, key in enumerate(self.variants):
            vf = vframes[i]
            _ax, im = self.images[i]
            im.set_data(dequantize_natural(vf))
            if new_geom:
                im.set_extent((geom.x1, geom.x2, geom.p1, geom.p2))
            self.rho_lines[key].set_data(self._xax, vf.rho)
            self.phi_lines[key].set_data(self._pax, vf.phi)
        for c in self.cursors:
            c.set_xdata([t, t])
        self.canvas.restore_region(self._bg)
        for a in self._dynamic:
            a.axes.draw_artist(a) if a.axes is not None \
                else self.fig.draw_artist(a)
        # RGBA straight out of the Agg buffer: ffmpeg is fed rgba rawvideo,
        # so no per-frame RGB repack (6 MiB/frame of pure copying at 1080p)
        return self.canvas.buffer_rgba()

    def close(self):
        self.fig.clf()


def meta_columns(cfg, geom, stats, variants, k0, k1, n_frames, fps,
                 param_log=()):
    """The two text columns of the metadata block (left: what this video is;
    right: the physics + IC expression, wrapped). `geom` is the geometry of
    the FIRST exported record; auto-expand may move it later, hence the
    per-record header line and the union window quoted here."""
    from . import describe
    left = _wrap([
        "variants: %s" % ", ".join(k.upper() for k in variants),
        "records %d … %d  →  %d frames @ %g fps  (%.1f s)"
        % (k0, k1, n_frames, fps, n_frames/float(fps)),
        "grid at record %d: %d×%d;  axes span x ∈ [%.6g, %.6g], "
        "p ∈ [%.6g, %.6g]"
        % (k0, geom.Nx, geom.Np, stats.x1, stats.x2, stats.p1, stats.p2),
        "units: Hartree atomic (ℏ = mₑ = e = 1);  1 a.u. of time = %g fs"
        % AU_TIME_FS,
    ], 62)
    right = _wrap(describe.param_lines(cfg, param_log, k0, k1)
                  + describe.ic_expression(cfg.ic, cfg.hbar_eff), 150)
    return left, right


def _wrap(lines, width):
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width, subsequent_indent="    ")
                   or [""])
    return out


def variant_keys(vframes):
    """Bundle order -> variant keys, validated against the known set."""
    keys = [key_of_vid(vf.vid) for vf in vframes]
    for k in keys:
        if k not in VARIANTS:
            raise ValueError("unknown variant id in record: %r" % k)
    return keys

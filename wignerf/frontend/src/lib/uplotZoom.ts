/**
 * Shared interactive zoom for the uPlot charts (series, marginals, U(x)):
 * drag-select zoom (x / y / box via cursor.drag with `uni`), wheel zoom
 * (plain = x anchored at the cursor, Shift = y), double-click = reset to
 * full auto-fit. One instance per chart component, created in setup so the
 * zoom window survives the grid-lines destroy+rebuild cycle.
 *
 * uPlot facts this relies on (verified against the installed 1.6.32):
 * - mouseUp fires the setSelect hook BEFORE applying the drag scales, so
 *   posToVal() still reads the pre-zoom scales there; under `uni` the
 *   non-engaged axis has its select expanded to the full plot dimension,
 *   which is how axis intent is detected below.
 * - setData(data, false) performs no rescale AND no redraw — the follow-up
 *   setScale() calls are what repaint.
 * - an explicit setScale('y', {min, max}) bypasses the scale's range()
 *   function, so a pinned y window doesn't fight adaptive-fit charts; any
 *   x setScale (incl. the built-in dblclick autoScaleX, which never skips
 *   as a no-op) re-ranges auto y scales through that range() function.
 * - the wheel/dblclick listeners live on u.over (the plotting area only),
 *   so preventDefault never blocks scrolling over titles/axes.
 */
import type uPlot from 'uplot'

export interface ZoomWindow { min: number; max: number }

export interface UplotZoom {
  plugin: uPlot.Plugin
  /** Zoom-preserving setData: full autoscale when idle, pinned window when
   *  zoomed (live appends must not move the user's window). */
  setData(u: uPlot, data: uPlot.AlignedData): void
  /** Re-assert the pinned window — setSeries visibility toggles internally
   *  auto-rescale y, which would silently drop a pinned y-zoom. */
  reapply(u: uPlot): void
  /** Forget the zoom (session restart); the caller's next setData autoscales. */
  reset(): void
  /** Current pinned x window (U(x) editor reads this as the sampling range). */
  readonly x: ZoomWindow | null
  readonly zoomed: boolean
}

export function createUplotZoom(opts: {
  /** Wheel zoom-in span multiplier per notch (zoom-out uses 1/factor). */
  wheelFactor?: number
  /** Clamp x-zoom to the data extent and auto-clear at full span — right
   *  for live charts; false = free range (U(x): zooming past the data is
   *  the point, it re-samples on the wider window). */
  clampX?: boolean
  onXChange?: (w: ZoomWindow | null) => void
} = {}): UplotZoom {
  const factor = opts.wheelFactor ?? 0.85
  const clampX = opts.clampX ?? true
  let zx: ZoomWindow | null = null
  let zy: ZoomWindow | null = null
  let overEl: HTMLElement | null = null
  let onWheelBound: ((e: WheelEvent) => void) | null = null
  let onDblBound: (() => void) | null = null

  function dataExtentX(u: uPlot): ZoomWindow | null {
    const xs = u.data[0]
    if (!xs?.length) return null
    const a = xs[0]!
    const b = xs[xs.length - 1]!
    // min/max, not [first, last]: a dt_sign flip makes t retrace itself
    return { min: Math.min(a, b), max: Math.max(a, b) }
  }

  function applyScales(u: uPlot) {
    u.batch(() => {
      const xw = zx ?? dataExtentX(u)
      if (xw) u.setScale('x', { min: xw.min, max: xw.max })
      if (zy) u.setScale('y', { min: zy.min, max: zy.max })
    })
  }

  function fireX() { opts.onXChange?.(zx) }

  function wheel(u: uPlot, e: WheelEvent) {
    const sx = u.scales.x!
    const sy = u.scales.y!
    if (sx.min == null || sx.max == null || sy.min == null || sy.max == null) return
    e.preventDefault()
    const rect = overEl!.getBoundingClientRect()
    const f = e.deltaY < 0 ? factor : 1 / factor
    if (e.shiftKey) {
      // y-zoom anchored at the cursor's y value (fy = 0 at the top)
      const fy = (e.clientY - rect.top) / rect.height
      const anchor = sy.max - fy * (sy.max - sy.min)
      const span = (sy.max - sy.min) * f
      zy = { min: anchor - (1 - fy) * span, max: anchor + fy * span }
    } else {
      // x-zoom anchored at the cursor's x value
      const fx = (e.clientX - rect.left) / rect.width
      const anchor = sx.min + fx * (sx.max - sx.min)
      const span = (sx.max - sx.min) * f
      let min = anchor - fx * span
      let next: ZoomWindow | null = { min, max: min + span }
      if (clampX) {
        const ext = dataExtentX(u)
        if (ext) {
          if (span >= ext.max - ext.min) {
            next = null // wheeled out to (past) the full extent: auto-fit resumes
          } else {
            min = Math.min(Math.max(min, ext.min), ext.max - span)
            next = { min, max: min + span }
          }
        }
      }
      zx = next
      fireX()
    }
    applyScales(u)
  }

  function dbl() {
    if (!zx && !zy) return
    // uPlot's own dblclick listener (registered first) already ran
    // autoScaleX(), which re-ranges auto y scales too — just clear mirrors
    zx = null
    zy = null
    fireX()
  }

  function onSetSelect(u: uPlot) {
    const sel = u.select
    if (sel.width <= 0 && sel.height <= 0) return
    const fullW = sel.width >= u.over.clientWidth - 0.5
    const fullH = sel.height >= u.over.clientHeight - 0.5
    if (fullW && fullH) return
    // mirror the scales uPlot's drag path applies right after this hook —
    // no setScale calls of our own here
    if (!fullH) {
      zy = { min: u.posToVal(sel.top + sel.height, 'y'), max: u.posToVal(sel.top, 'y') }
    }
    if (!fullW) {
      zx = { min: u.posToVal(sel.left, 'x'), max: u.posToVal(sel.left + sel.width, 'x') }
      // an x-only drag makes uPlot auto-rescale y over the new window
      if (fullH) zy = null
      fireX()
    }
  }

  const plugin: uPlot.Plugin = {
    opts: (_u, o) => {
      // cursor enabled (drag-select needs its mouse handlers) but visually
      // silent: no crosshair, no hover points — only the select box shows
      o.cursor = {
        x: false,
        y: false,
        points: { show: false },
        drag: { x: true, y: true, uni: 20, dist: 4 },
      }
    },
    hooks: {
      ready: (u: uPlot) => {
        overEl = u.over
        onWheelBound = (e: WheelEvent) => wheel(u, e)
        onDblBound = dbl
        overEl.addEventListener('wheel', onWheelBound, { passive: false })
        overEl.addEventListener('dblclick', onDblBound)
      },
      setSelect: onSetSelect,
      destroy: () => {
        if (overEl) {
          if (onWheelBound) overEl.removeEventListener('wheel', onWheelBound)
          if (onDblBound) overEl.removeEventListener('dblclick', onDblBound)
        }
        overEl = null
        onWheelBound = null
        onDblBound = null
      },
    },
  }

  return {
    plugin,
    setData(u, data) {
      if (!zx && !zy) {
        u.setData(data)
        return
      }
      u.setData(data, false)
      applyScales(u)
    },
    reapply(u) {
      if (zx || zy) applyScales(u)
    },
    reset() {
      zx = null
      zy = null
    },
    get x() { return zx },
    get zoomed() { return zx != null || zy != null },
  }
}

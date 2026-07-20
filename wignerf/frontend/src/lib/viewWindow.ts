/**
 * Pan/zoom view window for the WebGL W(x,p) heatmaps, in DOMAIN FRACTIONS:
 * x0..x1 horizontal (x axis), y0..y1 vertical (p axis, up); the full view
 * is (0,1,0,1). Reactive so panels and their axis overlays can watch it —
 * the main W panels share ONE window (same domain, they zoom together),
 * the IC preview owns its own. The window is clamped inside [0,1]: the
 * shader samples the torus periodically and showing the wrapped
 * continuation past the seam would be confusing as a default (could
 * become a toggle later).
 */
import { reactive } from 'vue'

export interface ViewWindow { x0: number; x1: number; y0: number; y1: number }

const MIN_SPAN = 1 / 256 // deepest zoom: 256×

export function createViewWindow(): ViewWindow {
  return reactive({ x0: 0, x1: 1, y0: 0, y1: 1 })
}

function place(lo: number, span: number): [number, number] {
  const l = Math.min(Math.max(lo, 0), 1 - span)
  return [l, l + span]
}

/** Zoom the spans by `factor`, keeping the domain point under the pointer
 *  fixed. fx/fy = pointer position as fractions of the element, fy p-up. */
export function zoomAt(v: ViewWindow, fx: number, fy: number, factor: number): void {
  const sx = Math.min(1, Math.max(MIN_SPAN, (v.x1 - v.x0) * factor))
  const sy = Math.min(1, Math.max(MIN_SPAN, (v.y1 - v.y0) * factor))
  ;[v.x0, v.x1] = place(v.x0 + fx * (v.x1 - v.x0) - fx * sx, sx)
  ;[v.y0, v.y1] = place(v.y0 + fy * (v.y1 - v.y0) - fy * sy, sy)
}

/** Pan by a pointer movement of (dfx, dfy) element fractions (dfy p-up):
 *  the content follows the pointer, so the window moves opposite. */
export function panBy(v: ViewWindow, dfx: number, dfy: number): void {
  const sx = v.x1 - v.x0
  const sy = v.y1 - v.y0
  ;[v.x0, v.x1] = place(v.x0 - dfx * sx, sx)
  ;[v.y0, v.y1] = place(v.y0 - dfy * sy, sy)
}

export function resetView(v: ViewWindow): void {
  v.x0 = 0
  v.x1 = 1
  v.y0 = 0
  v.y1 = 1
}

export function isZoomed(v: ViewWindow): boolean {
  return v.x0 > 0 || v.x1 < 1 || v.y0 > 0 || v.y1 < 1
}

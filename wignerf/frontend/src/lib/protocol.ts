/**
 * Binary frame-bundle decoder — the TypeScript mirror of
 * backend/core/protocol.py (see its docstring for the layout). Verified
 * against a python-generated golden fixture in protocol.test.ts.
 *
 * All sections are 4-byte aligned, so every TypedArray below is a
 * zero-copy view into the received ArrayBuffer.
 */

export const MAGIC = 0x57
export const VERSION = 3 // v3: per-record grid geometry (f64 x1,x2,p1,p2 in header)
export const MSG_FRAME = 1

export const FLAG_LIVE_PREVIEW = 1 << 0
export const FLAG_REPLAY = 1 << 1

export interface VariantFrame {
  vid: number // bit0 quantum, bit1 relativistic
  wmin: number
  wmax: number
  E: number
  xMean: number
  xStd: number
  pMean: number
  pStd: number
  purity: number // gamma = 2*pi*hbar*int W^2 dx dp = Tr rho^2
  dt: number
  wq: Uint16Array // (Nx, Np) row-major, fftshifted order
  rho: Float32Array // (Nx), natural order
  phi: Float32Array // (Np), natural order
}

export interface Frame {
  record: number
  t: number
  // grid geometry is a per-record fact: auto-expand may move/double the
  // domain mid-run, and a replayed record keeps the grid it was computed on
  Nx: number
  Np: number
  x1: number
  x2: number
  p1: number
  p2: number
  flags: number
  variants: VariantFrame[]
}

export function decodeFrame(buf: ArrayBuffer): Frame {
  const dv = new DataView(buf)
  if (dv.getUint8(0) !== MAGIC) throw new Error('bad magic')
  const version = dv.getUint8(1)
  if (version !== VERSION) throw new Error(`protocol version ${version} != ${VERSION}`)
  if (dv.getUint8(2) !== MSG_FRAME) throw new Error('unexpected msg_type')
  const nVariants = dv.getUint8(3)
  const record = dv.getUint32(4, true)
  const t = dv.getFloat64(8, true)
  const Nx = dv.getUint32(16, true)
  const Np = dv.getUint32(20, true)
  const flags = dv.getUint32(24, true)
  const x1 = dv.getFloat64(32, true)
  const x2 = dv.getFloat64(40, true)
  const p1 = dv.getFloat64(48, true)
  const p2 = dv.getFloat64(56, true)

  let off = 64
  const variants: VariantFrame[] = []
  for (let i = 0; i < nVariants; i++) {
    const vid = dv.getUint8(off)
    const f = (k: number) => dv.getFloat32(off + 4 + 4 * k, true)
    const [wmin, wmax, E, xMean, xStd, pMean, pStd, purity, dt] =
      [f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)]
    off += 40
    const wq = new Uint16Array(buf, off, Nx * Np)
    off += 2 * Nx * Np
    const rho = new Float32Array(buf, off, Nx)
    off += 4 * Nx
    const phi = new Float32Array(buf, off, Np)
    off += 4 * Np
    variants.push({ vid, wmin, wmax, E, xMean, xStd, pMean, pStd, purity, dt, wq, rho, phi })
  }
  if (off !== buf.byteLength) throw new Error(`trailing bytes: ${off} != ${buf.byteLength}`)
  return { record, t, Nx, Np, x1, x2, p1, p2, flags, variants }
}

export function variantName(vid: number): string {
  return (vid & 1 ? 'quantum' : 'classical') + ' ' +
    (vid & 2 ? 'relativistic' : 'non-relativistic')
}

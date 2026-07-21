import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { FLAG_REPLAY, decodeFrame } from './protocol'

const here = dirname(fileURLToPath(import.meta.url))
const bin = readFileSync(join(here, '__fixtures__', 'frame.bin'))
const meta = JSON.parse(
  readFileSync(join(here, '__fixtures__', 'frame.json'), 'utf8'),
)
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)

describe('decodeFrame', () => {
  it('matches the python-generated golden bundle', () => {
    const f = decodeFrame(buf as ArrayBuffer)
    expect(f.record).toBe(meta.record)
    expect(f.t).toBeCloseTo(meta.t, 12)
    expect(f.Nx).toBe(meta.Nx)
    expect(f.Np).toBe(meta.Np)
    expect(f.x1).toBe(meta.x1)
    expect(f.x2).toBe(meta.x2)
    expect(f.p1).toBe(meta.p1)
    expect(f.p2).toBe(meta.p2)
    expect(f.flags).toBe(FLAG_REPLAY)
    expect(f.variants.length).toBe(meta.variants.length)
    for (let i = 0; i < f.variants.length; i++) {
      const v = f.variants[i]!
      const m = meta.variants[i]
      expect(v.vid).toBe(m.vid)
      for (const k of ['wmin', 'wmax', 'E', 'purity', 'dt'] as const) {
        expect(v[k]).toBeCloseTo(m[k], 6)
      }
      expect(v.xMean).toBeCloseTo(m.x_mean, 6)
      expect(v.xStd).toBeCloseTo(m.x_std, 6)
      expect(v.pMean).toBeCloseTo(m.p_mean, 6)
      expect(v.pStd).toBeCloseTo(m.p_std, 6)
      expect(Array.from(v.wq)).toEqual(m.wq)
      for (let j = 0; j < v.rho.length; j++) {
        expect(v.rho[j]).toBeCloseTo(m.rho[j], 6)
      }
      for (let j = 0; j < v.phi.length; j++) {
        expect(v.phi[j]).toBeCloseTo(m.phi[j], 6)
      }
    }
  })

  it('rejects a wrong protocol version', () => {
    const bad = new Uint8Array(buf.slice(0) as ArrayBuffer)
    bad[1] = 99
    expect(() => decodeFrame(bad.buffer as ArrayBuffer)).toThrow(/version/)
  })
})

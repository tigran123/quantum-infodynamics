/**
 * Diverging colormap LUT for W(x,p): matplotlib's "bwr" (blue-white-red),
 * matching the convention of the old dynamics/solanim.py. 256 RGBA8
 * entries; the shader maps W=0 to the LUT center (MidpointNormalize).
 */

export function bwrLUT(): Uint8Array {
  const lut = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const u = i / 255 // 0 -> blue, 0.5 -> white, 1 -> red
    let r: number, g: number, b: number
    if (u < 0.5) {
      const s = u * 2
      r = s
      g = s
      b = 1
    } else {
      const s = (u - 0.5) * 2
      r = 1
      g = 1 - s
      b = 1 - s
    }
    lut[4 * i] = Math.round(255 * r)
    lut[4 * i + 1] = Math.round(255 * g)
    lut[4 * i + 2] = Math.round(255 * b)
    lut[4 * i + 3] = 255
  }
  return lut
}

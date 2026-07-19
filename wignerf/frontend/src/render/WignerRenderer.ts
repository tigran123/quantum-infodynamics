/**
 * WebGL2 heatmap renderer for W(x,p) — a plain class, deliberately OUTSIDE
 * Vue reactivity. Design (see plan):
 *
 * - R16UI texture + usampler2D + texelFetch: the received Uint16Array view
 *   uploads untouched (zero copies, exact 16-bit fidelity). Integer
 *   textures cannot LINEAR-filter, so bilinear interpolation is done
 *   manually in the shader — which simultaneously gives correct periodic
 *   (wrapped) sampling of the toroidal domain.
 * - The payload is in fftshifted order; the unshift is a free half-period
 *   texture-coordinate offset (fract(uv + 0.5); Nx, Np are even).
 * - Diverging colormap centered at W=0 with SYMMETRIC two-sided scaling:
 *   u = 0.5 + 0.5*W/max(Wmax, -Wmin), so color intensity is proportional
 *   to |W| with one shared scale. (The asymmetric MidpointNormalize of
 *   dynamics/midnorm.py would hand the whole blue half of the map to
 *   Wmin however tiny — rendering ~1e-6 numerical/quantization noise as
 *   saturated blue. Genuine negativity, e.g. cat-state fringes, is
 *   comparable to the peak and stays vividly blue under either scheme.)
 *   uQ = the frame's own (wmin, wmax), always used for dequantization;
 *   uC = the color scale — equal to uQ when autoscaling, or a locked pair.
 */

import type { VariantFrame } from '../lib/protocol'
import { bwrLUT } from '../lib/colormaps'
import { perfInfo, perfStage } from '../lib/perf'

const VS = `#version 300 es
in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

const FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uW;    // width = Np (p, fast axis), height = Nx (x)
uniform sampler2D uLUT;
uniform vec2 uQ;          // (wmin, wmax) of THIS frame - dequantization
uniform vec2 uC;          // color scale (min, max); == uQ when autoscaling
uniform vec2 uSize;       // (Np, Nx)
in vec2 vUV;
out vec4 fragColor;

float fetchW(ivec2 ij) {
  uint q = texelFetch(uW, ij, 0).r;
  return uQ.x + (uQ.y - uQ.x) * (float(q) / 65535.0);
}

float sampleW(vec2 st) {  // st: (s over Np, t over Nx) in [0,1], periodic
  vec2 xy = st * uSize - 0.5;
  vec2 f = fract(xy);
  ivec2 sz = ivec2(uSize);
  ivec2 a = (ivec2(floor(xy)) % sz + sz) % sz;
  ivec2 b = ivec2((a.x + 1) % sz.x, (a.y + 1) % sz.y);
  float w00 = fetchW(a);
  float w10 = fetchW(ivec2(b.x, a.y));
  float w01 = fetchW(ivec2(a.x, b.y));
  float w11 = fetchW(b);
  return mix(mix(w00, w10, f.x), mix(w01, w11, f.x), f.y);
}

void main() {
  // screen: x horizontal (texture row t), p vertical up (texture col s);
  // +0.5 = ifftshift as a half-period offset on the periodic domain
  vec2 st = fract(vec2(vUV.y, vUV.x) + 0.5);
  float w = sampleW(st);
  // symmetric diverging scale: W=0 -> LUT center (white), intensity
  // proportional to |W| on both sides
  float scale = max(max(uC.y, -uC.x), 1e-30);
  float u = 0.5 + 0.5 * w / scale;
  fragColor = texture(uLUT, vec2(clamp(u, 0.0, 1.0), 0.5));
}`

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!
  gl.shaderSource(sh, src)
  gl.compileShader(sh)
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(sh))
  }
  return sh
}

export class WignerRenderer {
  private gl: WebGL2RenderingContext | null = null
  private prog: WebGLProgram | null = null
  private texW: WebGLTexture | null = null
  private texLUT: WebGLTexture | null = null
  private uQ: WebGLUniformLocation | null = null
  private uC: WebGLUniformLocation | null = null
  private uSize: WebGLUniformLocation | null = null
  private nx = 0
  private np = 0
  private q: [number, number] = [0, 1]
  /** When set, the color scale is locked to this (min, max); otherwise
   *  every frame autoscales to its own range. */
  colorLock: [number, number] | null = null

  init(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false })
    if (!gl) throw new Error('WebGL2 is not available')
    this.gl = gl
    // expose the real renderer once — a "SwiftShader" here means software
    // rendering, which caps large-grid playback at a few fps
    if (!perfInfo.gl_renderer) {
      const dbg = gl.getExtension('WEBGL_debug_renderer_info')
      const name = String(gl.getParameter(
        dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER))
      perfInfo.gl_renderer = name
      console.info('wignerf WebGL renderer:', name)
    }
    const prog = gl.createProgram()!
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS))
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(prog))
    }
    this.prog = prog
    gl.useProgram(prog)

    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, 'aPos')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)

    this.uQ = gl.getUniformLocation(prog, 'uQ')
    this.uC = gl.getUniformLocation(prog, 'uC')
    this.uSize = gl.getUniformLocation(prog, 'uSize')

    // LUT: 256x1 RGBA8, LINEAR (it is a float texture, filtering is fine)
    this.texLUT = gl.createTexture()
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.texLUT)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA,
      gl.UNSIGNED_BYTE, bwrLUT())
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.uniform1i(gl.getUniformLocation(prog, 'uLUT'), 1)
    gl.uniform1i(gl.getUniformLocation(prog, 'uW'), 0)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 2)
  }

  private ensureTexture(Nx: number, Np: number) {
    const gl = this.gl!
    if (this.texW && this.nx === Nx && this.np === Np) return
    if (this.texW) gl.deleteTexture(this.texW)
    this.texW = gl.createTexture()
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.texW)
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16UI, Np, Nx)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    this.nx = Nx
    this.np = Np
  }

  /** Upload one variant's quantized W. wq is (Nx, Np) row-major, i.e. the
   *  texture is Np wide (p) and Nx tall (x). */
  upload(v: VariantFrame, Nx: number, Np: number) {
    const gl = this.gl
    if (!gl) return
    const t0 = performance.now()
    this.ensureTexture(Nx, Np)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.texW)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, Np, Nx,
      gl.RED_INTEGER, gl.UNSIGNED_SHORT, v.wq)
    this.q = [v.wmin, v.wmax]
    perfStage('upload', performance.now() - t0)
  }

  render() {
    const gl = this.gl
    if (!gl || !this.texW) return
    const t0 = performance.now()
    const canvas = gl.canvas as HTMLCanvasElement
    const dpr = window.devicePixelRatio || 1
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr))
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr))
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w
      canvas.height = h
    }
    gl.viewport(0, 0, w, h)
    gl.useProgram(this.prog)
    const c = this.colorLock ?? this.q
    gl.uniform2f(this.uQ, this.q[0], this.q[1])
    gl.uniform2f(this.uC, c[0], c[1])
    gl.uniform2f(this.uSize, this.np, this.nx)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    perfStage('draw', performance.now() - t0)
  }

  dispose() {
    const gl = this.gl
    if (!gl) return
    if (this.texW) gl.deleteTexture(this.texW)
    if (this.texLUT) gl.deleteTexture(this.texLUT)
    if (this.prog) gl.deleteProgram(this.prog)
    this.gl = null
  }
}

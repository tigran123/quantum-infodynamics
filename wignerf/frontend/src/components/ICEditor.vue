<script setup lang="ts">
/**
 * Initial-condition editor: a list of Gaussian components (mixture or cat)
 * plus a live phase-space preview (POST /api/preview/wigner returns the
 * same binary bundle as the stream, rendered by the same WebGL renderer).
 * Components are draggable directly on the preview canvas — pointer down
 * near a peak marker grabs it; x0/p0 follow the pointer.
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { api } from '../api'
import GridOverlay from './GridOverlay.vue'
import { decodeFrame } from '../lib/protocol'
import { defaultConfig, type GridCfg, type ICCfg, type ICComponentCfg } from '../lib/config'
import { WignerRenderer } from '../render/WignerRenderer'

const props = defineProps<{
  ic: ICCfg
  grid: GridCfg
  hbarEff: number
  showGrid?: boolean
}>()

const emit = defineEmits<{ (e: 'changed'): void }>()

const canvas = ref<HTMLCanvasElement | null>(null)
const overlay = ref<HTMLDivElement | null>(null)
const selected = ref(0)
const deficit = ref('')
const warnings = ref<string[]>([])
const renderer = new WignerRenderer()
let timer: ReturnType<typeof setTimeout> | null = null
let dragging = -1

const state = reactive({ previewOk: false, error: '' })

function derivedSigmaP(sx: number): number {
  return props.hbarEff / (2 * sx)
}

async function refresh() {
  try {
    const { data, headers } = await api.post('/preview/wigner', {
      type: props.ic.type,
      components: props.ic.components,
      grid: props.grid,
      hbar_eff: props.hbarEff,
    }, { responseType: 'arraybuffer' })
    const f = decodeFrame(data as ArrayBuffer)
    const v = f.variants[0]!
    renderer.upload(v, f.Nx, f.Np)
    renderer.render()
    deficit.value = String(headers['x-wignerf-norm-deficit'] ?? '')
    // percent-encoded server-side: HTTP headers are latin-1, the messages
    // carry Unicode (sigma, hbar, rho...)
    const w = decodeURIComponent(String(headers['x-wignerf-warnings'] ?? ''))
    warnings.value = w ? w.split(' | ') : []
    state.previewOk = true
    state.error = ''
  } catch (e: unknown) {
    const err = e as { response?: { data?: ArrayBuffer } }
    state.previewOk = false
    state.error = err.response?.data
      ? new TextDecoder().decode(err.response.data as ArrayBuffer)
      : String(e)
  }
}

function scheduleRefresh(notify = true) {
  if (notify) emit('changed')
  if (timer) clearTimeout(timer)
  timer = setTimeout(refresh, 150)
}

watch(() => [props.ic, props.grid, props.hbarEff], () => scheduleRefresh(false),
  { deep: true })

// -- drag-to-place ---------------------------------------------------------

function toData(ev: PointerEvent): { x: number; p: number } {
  const r = overlay.value!.getBoundingClientRect()
  const fx = (ev.clientX - r.left) / r.width
  const fy = (ev.clientY - r.top) / r.height
  return {
    x: props.grid.x1 + fx * (props.grid.x2 - props.grid.x1),
    p: props.grid.p2 - fy * (props.grid.p2 - props.grid.p1),
  }
}

function markerStyle(c: ICComponentCfg) {
  const fx = (c.x0 - props.grid.x1) / (props.grid.x2 - props.grid.x1)
  const fy = 1 - (c.p0 - props.grid.p1) / (props.grid.p2 - props.grid.p1)
  return { left: `${100 * fx}%`, top: `${100 * fy}%` }
}

function onDown(ev: PointerEvent) {
  const d = toData(ev)
  const sx = (props.grid.x2 - props.grid.x1) / 15
  const sp = (props.grid.p2 - props.grid.p1) / 15
  let best = -1
  let bestDist = 1
  props.ic.components.forEach((c, i) => {
    const dist = Math.hypot((c.x0 - d.x) / sx, (c.p0 - d.p) / sp)
    if (dist < bestDist) { best = i; bestDist = dist }
  })
  if (best >= 0) {
    dragging = best
    selected.value = best
    ;(ev.target as HTMLElement).setPointerCapture(ev.pointerId)
  }
}

function onMove(ev: PointerEvent) {
  if (dragging < 0) return
  const d = toData(ev)
  const c = props.ic.components[dragging]!
  c.x0 = Math.round(d.x * 1000) / 1000
  c.p0 = Math.round(d.p * 1000) / 1000
  scheduleRefresh()
}

function onUp() { dragging = -1 }

// -- component list ----------------------------------------------------------

function addComponent() {
  // new mixture components start as minimal packets (sigma_p = hbar/2sigma_x)
  // so adding one never makes W sub-Heisenberg by default
  props.ic.components.push({
    x0: 0, p0: 0, sigma_x: 0.5,
    sigma_p: props.ic.type === 'cat' ? null : derivedSigmaP(0.5),
    weight: 1, phase: 0,
  })
  selected.value = props.ic.components.length - 1
  scheduleRefresh()
}

function removeComponent(i: number) {
  if (props.ic.components.length <= 1) return
  props.ic.components.splice(i, 1)
  selected.value = Math.min(selected.value, props.ic.components.length - 1)
  scheduleRefresh()
}

function resetIC() {
  if (!confirm('Reset the initial condition to the default single Gaussian?')) return
  const d = defaultConfig().ic
  props.ic.type = d.type
  props.ic.components.splice(0, props.ic.components.length, ...d.components)
  selected.value = 0
  scheduleRefresh()
}

function setType(t: 'mixture' | 'cat') {
  props.ic.type = t
  for (const c of props.ic.components) {
    if (t === 'cat') c.sigma_p = null
    else if (c.sigma_p == null) c.sigma_p = derivedSigmaP(c.sigma_x)
  }
  scheduleRefresh()
}

const sel = computed(() => props.ic.components[selected.value])

onMounted(() => {
  renderer.init(canvas.value!)
  void refresh()
})

onBeforeUnmount(() => {
  if (timer) clearTimeout(timer)
  renderer.dispose()
})
</script>

<template>
  <section class="space-y-1.5">
    <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">
      Initial condition
    </h3>

    <div class="flex gap-1 text-xs">
      <button
        v-for="t in (['mixture', 'cat'] as const)" :key="t"
        class="flex-1 py-1 rounded border"
        :class="ic.type === t
          ? 'bg-sky-900/60 border-sky-600 text-sky-200'
          : 'bg-neutral-900 border-neutral-700 text-neutral-400 hover:bg-neutral-800'"
        :title="t === 'mixture'
          ? 'statistical mixture: W >= 0, independent σₓ, σₚ'
          : 'coherent superposition (cat): interference fringes, σₚ derived'"
        @click="setType(t)"
      >{{ t }}</button>
    </div>

    <!-- phase-space preview with draggable peaks -->
    <div class="relative aspect-square w-full border border-neutral-700 rounded overflow-hidden">
      <canvas ref="canvas" class="w-full h-full block bg-black"></canvas>
      <GridOverlay v-if="showGrid ?? true"
                   :x1="grid.x1" :x2="grid.x2" :p1="grid.p1" :p2="grid.p2" />
      <div ref="overlay" class="absolute inset-0 touch-none cursor-crosshair"
           @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp">
        <div v-for="(c, i) in ic.components" :key="i"
             class="absolute w-3 h-3 -ml-1.5 -mt-1.5 rounded-full border-2 pointer-events-none"
             :class="i === selected ? 'border-yellow-300' : 'border-neutral-400/70'"
             :style="markerStyle(c)"></div>
      </div>
    </div>
    <div v-if="state.error" class="text-[11px] text-red-400">{{ state.error }}</div>
    <div v-if="deficit" class="text-[11px] text-neutral-500">
      norm deficit on grid: {{ deficit }}
    </div>
    <div v-for="(w, i) in warnings" :key="i" class="text-[11px] text-amber-400">⚠ {{ w }}</div>

    <!-- component list -->
    <div class="flex items-center gap-1 text-xs">
      <button v-for="(c, i) in ic.components" :key="i"
              class="px-2 py-0.5 rounded border"
              :class="i === selected
                ? 'border-yellow-400 text-yellow-300'
                : 'border-neutral-700 text-neutral-400'"
              @click="selected = i">{{ i + 1 }}</button>
      <button class="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700"
              @click="addComponent">+</button>
      <button class="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40"
              :disabled="ic.components.length <= 1"
              @click="removeComponent(selected)">−</button>
      <button class="ml-auto px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700"
              title="reset the IC to the default single Gaussian (takes effect on restart)"
              @click="resetIC">↺ defaults</button>
    </div>

    <!-- step="any": with a discrete step= the browser rejects perfectly
         good values (0.60, drag-placed coordinates, even 1.0) -->
    <div v-if="sel" class="grid grid-cols-2 gap-x-2 gap-y-1 text-xs">
      <label class="flex items-center gap-1">
        <span class="w-8 text-neutral-500">x₀</span>
        <input v-model.number="sel.x0" type="number" step="any"
               class="wf-num" @change="scheduleRefresh()" />
      </label>
      <label class="flex items-center gap-1">
        <span class="w-8 text-neutral-500">p₀</span>
        <input v-model.number="sel.p0" type="number" step="any"
               class="wf-num" @change="scheduleRefresh()" />
      </label>
      <label class="flex items-center gap-1">
        <span class="w-8 text-neutral-500">σₓ</span>
        <input v-model.number="sel.sigma_x" type="number" step="any" min="0.01"
               class="wf-num" @change="scheduleRefresh()" />
      </label>
      <label class="flex items-center gap-1" :title="ic.type === 'cat'
               ? 'derived: ℏ/(2σₓ) — a Gaussian wavefunction is a minimal packet' : ''">
        <span class="w-8 text-neutral-500">σₚ</span>
        <input v-if="ic.type === 'mixture'" v-model.number="sel.sigma_p"
               type="number" step="any" min="0.01"
               class="wf-num" @change="scheduleRefresh()" />
        <input v-else :value="derivedSigmaP(sel.sigma_x).toFixed(4)" disabled
               class="wf-num opacity-50" />
      </label>
      <label class="flex items-center gap-1"
             :title="ic.type === 'mixture'
               ? 'relative weight: ensemble probability wⱼ/Σw in ρ = Σ wⱼ ρⱼ'
               : 'relative weight: |cⱼ|² in ψ ∝ Σ cⱼψⱼ, cⱼ = √wⱼ·exp(iφⱼ)'">
        <span class="w-8 text-neutral-500">w</span>
        <input v-model.number="sel.weight" type="number" step="any" min="0.01"
               class="wf-num" @change="scheduleRefresh()" />
      </label>
      <label class="flex items-center gap-1"
             :title="'phase of cⱼ = √wⱼ·exp(iφⱼ) — sets the interference fringe phase (e.g. even/odd cat); meaningless for a mixture (no coherence)'">
        <span class="w-8 text-neutral-500">φ</span>
        <input v-model.number="sel.phase" type="number" step="any"
               :disabled="ic.type === 'mixture'"
               class="wf-num" :class="{ 'opacity-50': ic.type === 'mixture' }"
               @change="scheduleRefresh()" />
      </label>
    </div>
  </section>
</template>

<style>
.wf-num {
  width: 100%;
  background: #171717;
  border: 1px solid #404040;
  border-radius: 4px;
  padding: 2px 6px;
  font-variant-numeric: tabular-nums;
}
</style>

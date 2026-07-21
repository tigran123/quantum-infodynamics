<script setup lang="ts">
/**
 * Overlaid marginal distributions of all active variants: rho(x) or phi(p),
 * updated per received frame. uPlot instance lives outside Vue reactivity;
 * data arrives via the frame-source callback.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { perfStage } from '../lib/perf'
import { loadHidden, saveHidden } from '../lib/plotPrefs'
import type { Frame } from '../lib/protocol'
import { createUplotZoom } from '../lib/uplotZoom'
import { VARIANT_META, type VariantKey } from '../lib/variants'

const props = defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  variants: VariantKey[]
  which: 'rho' | 'phi'
  // natural-order axis: a1 + i*(a2-a1)/N
  a1: number
  a2: number
  n: number
  showGrid?: boolean
}>()

const el = ref<HTMLDivElement | null>(null)
// per-plot display-only visibility (persisted)
const hidden = ref(loadHidden(props.which))
// created in setup so the zoom window survives the grid-lines rebuild
const zoom = createUplotZoom()
let chart: uPlot | null = null
let unsub: (() => void) | null = null
let lastData: uPlot.AlignedData | null = null

function toggle(v: VariantKey) {
  const h = hidden.value
  if (h.has(v)) h.delete(v)
  else h.add(v)
  saveHidden(props.which, h)
  chart?.setSeries(props.variants.indexOf(v) + 1, { show: !h.has(v) })
  // setSeries auto-rescales y, dropping a pinned y-zoom — re-assert it
  if (chart) zoom.reapply(chart)
}

// The axis is a per-FRAME fact: auto-expand regrids move/double the domain
// mid-run, and scrubbing across the regrid boundary flips it back and
// forth. Props seed the pre-first-frame axis; the frame handler rebuilds
// in place when the painted frame's geometry differs (no remount — that
// would drop the zoom and blank the plot until the next frame).
function buildAxis(a1: number, a2: number, n: number) {
  return Array.from({ length: n }, (_, i) => a1 + (i * (a2 - a1)) / n)
}
let axis = buildAxis(props.a1, props.a2, props.n)
let axisKey = `${props.a1}|${props.a2}|${props.n}`

function makeChart(width: number) {
  const series: uPlot.Series[] = [
    {},
    // distinct dashes: coincident curves (harmonic Q==C, c=137 rel~nonrel)
    // stay individually visible through the gaps of the ones on top
    ...props.variants.map((v) => ({
      label: v,
      stroke: VARIANT_META[v].color,
      dash: VARIANT_META[v].dash,
      width: 1.5,
      points: { show: false },
      // in the config (not just setSeries) so visibility survives the
      // grid-lines destroy+rebuild
      show: !hidden.value.has(v),
    })),
  ]
  const grid = { show: props.showGrid ?? true, stroke: '#3f3f46', width: 1 }
  return new uPlot(
    {
      width,
      height: 130,
      title: props.which === 'rho' ? 'ρ(x) = ∫W dp' : 'φ(p) = ∫W dx',
      legend: { show: false },
      plugins: [zoom.plugin], // owns the cursor config (drag/wheel/dblclick)
      scales: { x: { time: false } },
      axes: [
        { stroke: '#a3a3a3', grid, ticks: { stroke: '#525252' } },
        { stroke: '#a3a3a3', grid, ticks: { stroke: '#525252' } },
      ],
      series,
    },
    [axis, ...props.variants.map(() => new Array(axis.length).fill(null))],
    el.value!,
  )
}

onMounted(() => {
  chart = makeChart(el.value!.clientWidth || 360)
  const ro = new ResizeObserver(() => {
    if (chart && el.value) chart.setSize({ width: el.value.clientWidth, height: 130 })
  })
  ro.observe(el.value!)
  // the session fan-out is already rAF-timed; the Float32Array views go
  // into uPlot as-is (it only indexes them) — no Array.from boxing copies
  unsub = props.frameSource((f: Frame) => {
    if (!chart) return
    const a1 = props.which === 'rho' ? f.x1 : f.p1
    const a2 = props.which === 'rho' ? f.x2 : f.p2
    const n = props.which === 'rho' ? f.Nx : f.Np
    const key = `${a1}|${a2}|${n}`
    if (key !== axisKey) {
      axisKey = key
      axis = buildAxis(a1, a2, n)
    }
    lastData = [
      axis,
      ...f.variants.map((v) => props.which === 'rho' ? v.rho : v.phi),
    ] as unknown as uPlot.AlignedData
    const t0 = performance.now()
    // zoom-preserving per painted frame: same cost class as a plain
    // autoscaling setData (no new per-frame allocations)
    zoom.setData(chart, lastData)
    perfStage('plots', performance.now() - t0)
  })
  // grid-lines toggle: rebuild the chart in place, keeping the data — a
  // remount would blank the plot until the next frame arrives
  watch(() => props.showGrid, () => {
    chart?.destroy()
    chart = makeChart(el.value?.clientWidth || 360)
    if (lastData) zoom.setData(chart, lastData)
  })
  onBeforeUnmount(() => ro.disconnect())
})

onBeforeUnmount(() => {
  unsub?.()
  chart?.destroy()
})
</script>

<template>
  <div class="flex items-center gap-1">
    <div v-if="variants.length > 1" class="flex flex-col gap-0.5 shrink-0">
      <label v-for="v in variants" :key="v"
             class="flex items-center gap-0.5 cursor-pointer select-none"
             :title="`show/hide ${VARIANT_META[v].label} in this plot only`">
        <input type="checkbox" class="scale-75" :checked="!hidden.has(v)"
               @change="toggle(v)" />
        <span class="text-[9px] leading-none"
              :style="{ color: VARIANT_META[v].color }">{{ v.toUpperCase() }}</span>
      </label>
    </div>
    <div ref="el" class="wf-plot flex-1 min-w-0"></div>
  </div>
</template>

<style>
.wf-plot .u-title {
  color: #d4d4d4;
  font-size: 12px;
  font-weight: 500;
}
.wf-plot .u-values { color: #d4d4d4; }
/* drag-select zoom box (uPlot's default is invisible on the dark theme) */
.wf-plot .u-select { background: rgba(255, 255, 255, 0.12); }
</style>

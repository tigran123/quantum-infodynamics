<script setup lang="ts">
/**
 * Overlaid marginal distributions of all active variants: rho(x) or phi(p),
 * updated per received frame. uPlot instance lives outside Vue reactivity;
 * data arrives via the frame-source callback.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { Frame } from '../lib/protocol'
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
let chart: uPlot | null = null
let unsub: (() => void) | null = null
let pending = false
let lastData: uPlot.AlignedData | null = null

const axis = Array.from({ length: props.n },
  (_, i) => props.a1 + (i * (props.a2 - props.a1)) / props.n)

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
    })),
  ]
  const grid = { show: props.showGrid ?? true, stroke: '#3f3f46', width: 1 }
  return new uPlot(
    {
      width,
      height: 130,
      title: props.which === 'rho' ? 'ρ(x) = ∫W dp' : 'φ(p) = ∫W dx',
      legend: { show: false },
      cursor: { show: false },
      scales: { x: { time: false } },
      axes: [
        { stroke: '#a3a3a3', grid, ticks: { stroke: '#525252' } },
        { stroke: '#a3a3a3', grid, ticks: { stroke: '#525252' } },
      ],
      series,
    },
    [axis, ...props.variants.map(() => new Array(props.n).fill(null))],
    el.value!,
  )
}

onMounted(() => {
  chart = makeChart(el.value!.clientWidth || 360)
  const ro = new ResizeObserver(() => {
    if (chart && el.value) chart.setSize({ width: el.value.clientWidth, height: 130 })
  })
  ro.observe(el.value!)
  unsub = props.frameSource((f: Frame) => {
    if (!chart) return
    lastData = [
      axis,
      ...f.variants.map((v) =>
        Array.from(props.which === 'rho' ? v.rho : v.phi)),
    ]
    if (!pending) {
      pending = true
      requestAnimationFrame(() => {
        pending = false
        if (lastData) chart?.setData(lastData)
      })
    }
  })
  // grid-lines toggle: rebuild the chart in place, keeping the data — a
  // remount would blank the plot until the next frame arrives
  watch(() => props.showGrid, () => {
    chart?.destroy()
    chart = makeChart(el.value?.clientWidth || 360)
    if (lastData) chart.setData(lastData)
  })
  onBeforeUnmount(() => ro.disconnect())
})

onBeforeUnmount(() => {
  unsub?.()
  chart?.destroy()
})
</script>

<template>
  <div ref="el" class="wf-plot"></div>
</template>

<style>
.wf-plot .u-title {
  color: #d4d4d4;
  font-size: 12px;
  font-weight: 500;
}
.wf-plot .u-values { color: #d4d4d4; }
</style>

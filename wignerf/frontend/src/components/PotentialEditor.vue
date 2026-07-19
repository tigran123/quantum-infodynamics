<script setup lang="ts">
/**
 * Analytic U(x) editor with instant feedback: debounced compile+sample via
 * POST /api/preview/potential, a live plot of U over the grid range, and
 * per-variant-family validity badges (quantum needs U finite on the
 * extended Bopp range; classical needs a delta-free dU/dx).
 *
 * "Apply live" pushes the new U into the RUNNING session (set_params);
 * "Use at restart" only updates the config for the next restart.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '../api'

const props = defineProps<{
  modelValue: string
  grid: { x1: number; x2: number; Nx: number; p1: number; p2: number; Np: number }
  hbarEff: number
  live: boolean          // a session is running -> allow live apply
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'applyLive', expr: string): void
}>()

interface PreviewResult {
  ok: boolean
  error?: string
  validity?: { quantum: boolean; classical: boolean }
  reasons?: string[]
  warnings?: string[]
  samples?: { x: number[]; U: (number | null)[] }
}

const draft = ref(props.modelValue)
const result = ref<PreviewResult | null>(null)
const plotEl = ref<HTMLDivElement | null>(null)
let chart: uPlot | null = null
let timer: ReturnType<typeof setTimeout> | null = null

async function compile() {
  try {
    const { data } = await api.post<PreviewResult>('/preview/potential', {
      expr: draft.value,
      x1: props.grid.x1,
      x2: props.grid.x2,
      grid: props.grid,
      hbar_eff: props.hbarEff,
    })
    result.value = data
    if (data.ok && data.samples && chart) {
      chart.setData([data.samples.x, data.samples.U])
    }
  } catch (e) {
    result.value = { ok: false, error: String(e) }
  }
}

watch(draft, () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(compile, 300)
})

onMounted(() => {
  chart = new uPlot(
    {
      width: plotEl.value!.clientWidth || 300,
      height: 110,
      title: 'U(x)',
      legend: { show: false },
      cursor: { show: false },
      scales: { x: { time: false } },
      axes: [
        { stroke: '#a3a3a3', grid: { stroke: '#26262666' } },
        { stroke: '#a3a3a3', grid: { stroke: '#26262666' } },
      ],
      series: [{}, { stroke: '#f472b6', width: 1.5, points: { show: false } }],
    },
    [[], []],
    plotEl.value!,
  )
  const ro = new ResizeObserver(() => {
    if (chart && plotEl.value)
      chart.setSize({ width: plotEl.value.clientWidth, height: 110 })
  })
  ro.observe(plotEl.value!)
  onBeforeUnmount(() => ro.disconnect())
  void compile()
})

onBeforeUnmount(() => chart?.destroy())
</script>

<template>
  <section class="space-y-1.5">
    <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Potential U(x)</h3>
    <input
      v-model="draft"
      spellcheck="false"
      class="w-full bg-neutral-900 border border-neutral-700 rounded px-2 py-1 font-mono text-sm"
      placeholder="e.g. x^2/2, -1/sqrt(x^2+2), x^4/4 - x^2/2"
    />
    <div class="flex items-center gap-2 text-xs">
      <template v-if="result">
        <span v-if="!result.ok" class="text-red-400">{{ result.error }}</span>
        <template v-else>
          <span :class="result.validity?.quantum ? 'text-emerald-400' : 'text-red-400'"
                :title="(result.reasons ?? []).join('; ')">
            quantum {{ result.validity?.quantum ? '✓' : '✗' }}
          </span>
          <span :class="result.validity?.classical ? 'text-emerald-400' : 'text-red-400'"
                :title="(result.reasons ?? []).join('; ')">
            classical {{ result.validity?.classical ? '✓' : '✗' }}
          </span>
        </template>
      </template>
    </div>
    <div v-if="result?.warnings?.length" class="text-[11px] text-amber-400 space-y-0.5">
      <div v-for="(w, i) in result.warnings" :key="i">⚠ {{ w }}</div>
    </div>
    <div v-if="result?.reasons?.length && result.ok" class="text-[11px] text-red-400 space-y-0.5">
      <div v-for="(r, i) in result.reasons" :key="i">{{ r }}</div>
    </div>
    <div ref="plotEl" class="wf-plot"></div>
    <div class="flex gap-2">
      <button
        class="flex-1 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-xs disabled:opacity-40"
        :disabled="!result?.ok"
        @click="emit('update:modelValue', draft)"
      >Use at restart</button>
      <button
        class="flex-1 py-1 rounded bg-pink-800 hover:bg-pink-700 text-xs disabled:opacity-40"
        :disabled="!result?.ok || !live"
        title="push into the running session at the frontier"
        @click="emit('applyLive', draft)"
      >Apply live</button>
    </div>
  </section>
</template>

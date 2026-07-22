<script setup lang="ts">
/**
 * Analytic U(x) editor with instant feedback: debounced compile+sample via
 * POST /api/preview/potential, a live plot of U over the grid range, and
 * per-variant-family validity badges (quantum needs U finite on the
 * extended Bopp range; classical needs a delta-free dU/dx).
 *
 * "Apply live" pushes the new U into the RUNNING session (set_params);
 * "Use at restart" only updates the config for the next restart. Both are
 * disabled while the draft is invalid for the ACTIVE variant families,
 * and that validity is emitted upward — the transport Solve is gated on
 * it (an invalid form must never coexist with a running computation).
 * "Apply live" is also inert when the draft already IS the live U: the
 * backend drops such a no-op, and a button that looks like it did
 * something is how a run's change log filled up with U changes that never
 * happened.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { createUplotZoom } from '../lib/uplotZoom'
import type { VariantKey } from '../lib/variants'

const props = defineProps<{
  modelValue: string
  grid: { x1: number; x2: number; Nx: number; p1: number; p2: number; Np: number }
  hbarEff: number
  live: boolean          // a session is running -> allow live apply
  variants: VariantKey[] // active selection: decides which families must be valid
  liveExpr?: string | null   // the session's CURRENT U(x), for the no-op gate
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
  (e: 'applyLive', expr: string): void
  (e: 'validity', valid: boolean): void
  (e: 'gridDirty'): void
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
// external config changes ("Reset setup to defaults") must reach the
// editor — a stale draft would keep showing (and validating, and offering
// to apply) the pre-reset expression
watch(() => props.modelValue, (v) => { draft.value = v })
const result = ref<PreviewResult | null>(null)
const plotEl = ref<HTMLDivElement | null>(null)
let chart: uPlot | null = null
let timer: ReturnType<typeof setTimeout> | null = null

// zoomed x window of the plot, DELIBERATELY independent of the grid's
// x1/x2 (null = follow the grid): zooming is how the "interesting domain"
// of U(x) is discovered, and the button below copies it into the grid.
// The backend samples U on this window while the validity probe keeps
// using the SIMULATION range (req.grid) — no restriction to [x1, x2].
const viewX = ref<{ x1: number; x2: number } | null>(null)
// clampX: false — zooming out past the data is the point (it re-samples
// on the wider window via the debounced compile below)
const zoom = createUplotZoom({
  clampX: false,
  onXChange: (w) => { viewX.value = w && { x1: w.min, x2: w.max } },
})

async function compile() {
  try {
    const { data } = await api.post<PreviewResult>('/preview/potential', {
      expr: draft.value,
      x1: viewX.value?.x1 ?? props.grid.x1,
      x2: viewX.value?.x2 ?? props.grid.x2,
      grid: props.grid,
      hbar_eff: props.hbarEff,
    })
    result.value = data
    if (data.ok && data.samples && chart) {
      zoom.setData(chart, [data.samples.x, data.samples.U])
    }
  } catch (e) {
    result.value = { ok: false, error: String(e) }
  }
}

// grid/ℏ changes move the extended Bopp range, so they can flip validity
// (a pole may enter it) — recompile on those too, not just on typing.
// viewX: wheel/drag rescale the old samples instantly client-side; the
// debounced fetch then refines/extends them to exactly the new window.
watch([draft, () => props.grid, () => props.hbarEff, viewX], () => {
  if (timer) clearTimeout(timer)
  timer = setTimeout(compile, 300)
}, { deep: true })

/** Copy the zoomed-in "interesting domain" into the grid's x₁/x₂
 *  (applies at restart, like any grid edit). */
function setGridFromView() {
  const v = viewX.value
  if (!v) return
  props.grid.x1 = parseFloat(v.x1.toPrecision(6))
  props.grid.x2 = parseFloat(v.x2.toPrecision(6))
  emit('gridDirty')
}

/** Valid for the CURRENT variant selection: each active family must
 *  accept the draft (an inactive family's ✗ doesn't block). */
const draftValid = computed(() => {
  const r = result.value
  if (!r?.ok) return false
  const needQ = props.variants.some((v) => v.startsWith('q'))
  const needC = props.variants.some((v) => v.startsWith('c'))
  return (!needQ || !!r.validity?.quantum) && (!needC || !!r.validity?.classical)
})
// immediate: the gate starts pessimistic until the first compile lands
watch(draftValid, (v) => emit('validity', v), { immediate: true })

/** Nothing to push: the running session already evolves this expression. */
const isLiveExpr = computed(() =>
  props.live && props.liveExpr != null && draft.value === props.liveExpr)

onMounted(() => {
  chart = new uPlot(
    {
      width: plotEl.value!.clientWidth || 300,
      height: 110,
      title: 'U(x)',
      legend: { show: false },
      plugins: [zoom.plugin], // owns the cursor config (drag/wheel/dblclick)
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
    <!-- the header's `uppercase` styling must not mangle math notation:
         U(x) keeps its case -->
    <h3 class="text-xs font-semibold text-neutral-400 uppercase tracking-wider">Potential <span class="normal-case">U(x)</span></h3>
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
    <div v-if="viewX" class="flex justify-end">
      <button
        class="px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700 text-xs"
        title="copy the current zoom window into x₁,x₂ (applies at restart); double-click the plot to reset the zoom"
        @click="setGridFromView"
      >x₁,x₂ ← view</button>
    </div>
    <div class="flex gap-2">
      <button
        class="flex-1 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-xs disabled:opacity-40"
        :disabled="!draftValid"
        :title="draftValid ? '' : 'invalid for the active variants — see the reasons above'"
        @click="emit('update:modelValue', draft)"
      >Use at restart</button>
      <button
        class="flex-1 py-1 rounded bg-pink-800 hover:bg-pink-700 text-xs disabled:opacity-40"
        :disabled="!draftValid || !live || isLiveExpr"
        title="push into the running session at the frontier"
        @click="emit('applyLive', draft)"
      >Apply live</button>
    </div>
    <!-- spell out the gate instead of leaving a dead button: this one is
         only for U(x) — m, c, ℏ and tol apply from their own fields -->
    <p v-if="isLiveExpr" class="text-[11px] text-neutral-500">
      already the live U(x) — edit it to enable "Apply live"
      (m, c, ℏ, tol apply from the Physics fields).
    </p>
  </section>
</template>

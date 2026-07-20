<script setup lang="ts">
/**
 * Per-record scalar time series (E or ΔX·ΔP) for all active variants.
 * Self-sufficient: polls GET /sessions/{id}/series incrementally every 2 s,
 * so the plot stays gapless even when live streaming coalesced frames.
 * X axis is physical time t (the per-record t the backend returns), so "E(t)"
 * et al. read in a.u.; record index n is kept only as the ordering / dedup /
 * eviction-gap key. A mid-run dt_sign flip makes t non-monotone, so the curve
 * retraces itself — physically faithful, and the default run is forward-only.
 */
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from '../api'
import { VARIANT_META, keyOfVid, type VariantKey } from '../lib/variants'

const props = defineProps<{
  sessionId: string | null
  variants: VariantKey[]
  which: 'E' | 'uncertainty' | 'purity'
  showGrid?: boolean
}>()

interface SeriesRec {
  n: number
  t: number
  variants: { vid: number; E: number; x_std: number; p_std: number; purity: number }[]
}

const TITLES = {
  E: 'E(t)',
  uncertainty: 'ΔX·ΔP(t)',
  purity: 'purity γ(t) = 2πℏ∬W²dxdp',
}

const el = ref<HTMLDivElement | null>(null)
let chart: uPlot | null = null
let timer: ReturnType<typeof setInterval> | null = null
let merged = -1
let gone = false      // session 404'd: stop polling until it changes
let inFlight = false  // never overlap polls (slow responses under load)
let generation = 0    // bumped by reset(): stale responses are discarded
const ts: number[] = []   // x-values: physical time t of each record (a.u.)
const cols: Map<VariantKey, (number | null)[]> = new Map()

function value(v: { E: number; x_std: number; p_std: number; purity: number }): number {
  if (props.which === 'E') return v.E
  if (props.which === 'purity') return v.purity
  return v.x_std * v.p_std
}

function pushNulls() {
  for (const k of props.variants) {
    if (!cols.has(k)) cols.set(k, [])
    cols.get(k)!.push(null)
  }
}

async function poll() {
  if (!props.sessionId || !chart || gone || inFlight) return
  inFlight = true
  const sid = props.sessionId
  const gen = generation
  try {
    const { data } = await api.get(`/sessions/${sid}/series`,
      { params: { start: merged + 1 }, timeout: 8000 })
    // a response that raced a session restart/reset must not touch state
    if (gen !== generation || sid !== props.sessionId) return
    const recs: SeriesRec[] = data.records
    for (const r of recs) {
      if (r.n <= merged) continue          // duplicate/out-of-order guard
      if (merged >= 0 && r.n > merged + 1) {
        // records between polls were evicted: insert a null separator so
        // uPlot breaks the line instead of bridging the gap with a chord.
        // Its x is a t strictly between the last real point and this one
        // (ts.at(-1) is always the previous REAL time here) so x stays sorted.
        ts.push((ts[ts.length - 1] + r.t) / 2)
        pushNulls()
      }
      ts.push(r.t)
      const byKey = new Map(r.variants.map((v) => [keyOfVid(v.vid), value(v)]))
      for (const k of props.variants) {
        if (!cols.has(k)) cols.set(k, [])
        cols.get(k)!.push(byKey.get(k) ?? null)
      }
      merged = r.n
    }
    if (recs.length) setChartData()
  } catch (e: unknown) {
    const st = (e as { response?: { status?: number } }).response?.status
    // only a 404 for the CURRENT session means it is gone; reset() re-arms
    if (st === 404 && gen === generation && sid === props.sessionId) gone = true
    // other errors (timeouts, transient network) retry on the next tick
  } finally {
    inFlight = false
  }
}

function reset() {
  generation++
  merged = -1
  gone = false
  inFlight = false
  ts.length = 0
  cols.clear()
  chart?.setData([[], ...props.variants.map(() => [])])
}

function makeChart(width: number): uPlot {
  const grid = { show: props.showGrid ?? true, stroke: '#3f3f46', width: 1 }
  return new uPlot(
    {
      width,
      height: 130,
      title: TITLES[props.which],
      legend: { show: false },
      cursor: { show: false },
      scales: {
        x: { time: false },
        // Tight adaptive y-range. uPlot's default expands flat data to a
        // huge range (e.g. purity == 1.000 -> [0, 2]), parking the curve
        // exactly on the center gridline where it is invisible; instead,
        // zoom to the data so even 1e-4-level structure (purity drift,
        // E splitting oscillation) is readable.
        y: {
          range: (_u: uPlot, mn: number, mx: number): [number, number] => {
            if (!Number.isFinite(mn) || !Number.isFinite(mx)) return [0, 1]
            const pad = Math.max((mx - mn) * 0.15, Math.abs(mx) * 1e-4, 1e-12)
            return [mn - pad, mx + pad]
          },
        },
      },
      axes: [
        { stroke: '#a3a3a3', grid, ticks: { stroke: '#525252' } },
        {
          stroke: '#a3a3a3', grid, ticks: { stroke: '#525252' },
          size: 72,
          // enough decimals to distinguish ticks on a tightly-zoomed axis
          // (e.g. purity 0.9998..1.0002 — default formatting prints "1")
          values: (_u: uPlot, splits: number[]) => {
            const step = Math.abs((splits[1] ?? 1) - (splits[0] ?? 0)) || 1
            const d = Math.max(0, Math.min(10, Math.ceil(-Math.log10(step)) + 1))
            return splits.map((v) => v.toFixed(d))
          },
        },
      ],
      series: [
        {},
        ...props.variants.map((v) => ({
          label: v,
          stroke: VARIANT_META[v].color,
          dash: VARIANT_META[v].dash,
          width: 1.5,
          points: { show: false },
        })),
      ],
    },
    [[], ...props.variants.map(() => [])],
    el.value!,
  )
}

function setChartData() {
  chart?.setData([ts, ...props.variants.map((k) => cols.get(k) ?? [])])
}

onMounted(() => {
  chart = makeChart(el.value!.clientWidth || 360)
  const ro = new ResizeObserver(() => {
    if (chart && el.value) chart.setSize({ width: el.value.clientWidth, height: 130 })
  })
  ro.observe(el.value!)
  timer = setInterval(poll, 2000)
  watch(() => props.sessionId, reset)
  // grid-lines toggle: rebuild in place, keeping accumulated data
  watch(() => props.showGrid, () => {
    chart?.destroy()
    chart = makeChart(el.value?.clientWidth || 360)
    setChartData()
  })
  onBeforeUnmount(() => ro.disconnect())
  // debug surface: window.__wfSeries.<which>() -> poller state
  const dbg = ((window as unknown as Record<string, unknown>).__wfSeries ??= {}) as
    Record<string, () => unknown>
  dbg[props.which] = () => ({
    points: ts.length, merged, lastT: ts.at(-1) ?? null, gone, inFlight, generation,
    sessionId: props.sessionId,
    lastVals: props.variants.map((k) => cols.get(k)?.at(-1) ?? null),
  })
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  chart?.destroy()
})
</script>

<template>
  <div ref="el" class="wf-plot"></div>
</template>

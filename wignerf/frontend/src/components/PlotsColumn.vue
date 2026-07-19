<script setup lang="ts">
/** The diagnostics column: colorbar, marginals, E(t), ΔX·ΔP(t), purity γ(t). */
import Colorbar from './Colorbar.vue'
import MarginalsPlot from './MarginalsPlot.vue'
import SeriesPlot from './SeriesPlot.vue'
import type { Frame } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import type { VariantKey } from '../lib/variants'

defineProps<{
  frameSource: (h: (f: Frame) => void) => () => void
  sessionId: string | null
  variants: VariantKey[]
  grid: GridCfg
  lastFrame: Frame | null
  showGrid: boolean
  plotsKey: string
}>()
</script>

<template>
  <div class="flex flex-col gap-2">
    <Colorbar :last-frame="lastFrame" />
    <MarginalsPlot :key="'r' + plotsKey" :frame-source="frameSource"
                   :variants="variants" which="rho" :show-grid="showGrid"
                   :a1="grid.x1" :a2="grid.x2" :n="grid.Nx" />
    <MarginalsPlot :key="'p' + plotsKey" :frame-source="frameSource"
                   :variants="variants" which="phi" :show-grid="showGrid"
                   :a1="grid.p1" :a2="grid.p2" :n="grid.Np" />
    <SeriesPlot :key="'e' + plotsKey" :session-id="sessionId"
                :variants="variants" which="E" :show-grid="showGrid" />
    <SeriesPlot :key="'u' + plotsKey" :session-id="sessionId"
                :variants="variants" which="uncertainty" :show-grid="showGrid" />
    <SeriesPlot :key="'g' + plotsKey" :session-id="sessionId"
                :variants="variants" which="purity" :show-grid="showGrid" />
  </div>
</template>

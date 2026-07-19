<script setup lang="ts">
/**
 * Colorbar for the W panels, matching the shader's SYMMETRIC diverging
 * scaling exactly: white pinned at W = 0, color intensity proportional to
 * |W| with the shared scale max(Wmax, -Wmin). For a frame with tiny Wmin
 * the left end is therefore near-white, not saturated blue — same as the
 * panels.
 */
import { computed } from 'vue'
import type { Frame } from '../lib/protocol'

const props = defineProps<{ lastFrame: Frame | null }>()

const range = computed(() => {
  const v = props.lastFrame?.variants[0]
  return v ? { min: v.wmin, max: v.wmax } : null
})

function bwr(u: number): string {
  u = Math.min(1, Math.max(0, u))
  let r: number, g: number, b: number
  if (u < 0.5) {
    const s = u * 2
    r = s; g = s; b = 1
  } else {
    const s = (u - 0.5) * 2
    r = 1; g = 1 - s; b = 1 - s
  }
  return `rgb(${Math.round(255 * r)},${Math.round(255 * g)},${Math.round(255 * b)})`
}

const gradient = computed(() => {
  if (!range.value) return 'linear-gradient(to right, #00f, #fff, #f00)'
  const { min, max } = range.value
  const scale = Math.max(max, -min, 1e-300)
  const f0 = Math.min(100, Math.max(0, (100 * (0 - min)) / (max - min || 1)))
  return `linear-gradient(to right, ${bwr(0.5 + 0.5 * min / scale)} 0%, ` +
    `rgb(255,255,255) ${f0}%, ${bwr(0.5 + 0.5 * max / scale)} 100%)`
})
</script>

<template>
  <div class="text-[10px] text-neutral-400 tabular-nums">
    <div class="h-3 rounded border border-neutral-700"
         :style="{ background: gradient }"></div>
    <div class="flex justify-between mt-0.5" v-if="range">
      <span>{{ range.min.toExponential(2) }}</span>
      <span>W = 0</span>
      <span>{{ range.max.toExponential(2) }}</span>
    </div>
  </div>
</template>

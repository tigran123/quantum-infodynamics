<script setup lang="ts">
/**
 * Axis grid overlay for the phase-space canvases (W panels, IC preview):
 * an SVG of tick lines at "nice" data intervals with edge labels, drawn
 * above the WebGL canvas. pointer-events: none — dragging/panning below
 * is unaffected.
 */
import { computed } from 'vue'

const props = defineProps<{
  x1: number
  x2: number
  p1: number
  p2: number
}>()

function niceTicks(a: number, b: number, target = 8): number[] {
  const span = b - a
  if (!(span > 0)) return []
  const raw = span / target
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= target)!
  const ticks: number[] = []
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-12 * span; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  }
  return ticks
}

const xt = computed(() => niceTicks(props.x1, props.x2))
const pt = computed(() => niceTicks(props.p1, props.p2))

const fx = (v: number) => (100 * (v - props.x1)) / (props.x2 - props.x1)
const fp = (v: number) => 100 - (100 * (v - props.p1)) / (props.p2 - props.p1)

function fmt(v: number): string {
  return Math.abs(v) < 1e-12 ? '0' : String(parseFloat(v.toPrecision(6)))
}
</script>

<template>
  <svg class="absolute inset-0 w-full h-full pointer-events-none select-none"
       preserveAspectRatio="none">
    <line v-for="v in xt" :key="'x' + v"
          :x1="fx(v) + '%'" :x2="fx(v) + '%'" y1="0" y2="100%"
          :stroke="v === 0 ? 'rgba(120,120,120,0.55)' : 'rgba(120,120,120,0.28)'"
          stroke-width="1" />
    <line v-for="v in pt" :key="'p' + v"
          :y1="fp(v) + '%'" :y2="fp(v) + '%'" x1="0" x2="100%"
          :stroke="v === 0 ? 'rgba(120,120,120,0.55)' : 'rgba(120,120,120,0.28)'"
          stroke-width="1" />
    <text v-for="v in xt" :key="'xl' + v"
          :x="fx(v) + '%'" y="99%" dx="2"
          fill="#737373" font-size="9">{{ fmt(v) }}</text>
    <text v-for="v in pt" :key="'pl' + v"
          x="0" :y="fp(v) + '%'" dx="2" dy="-2"
          fill="#737373" font-size="9">{{ fmt(v) }}</text>
  </svg>
</template>

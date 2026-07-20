<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import GridOverlay from './GridOverlay.vue'
import type { Frame } from '../lib/protocol'
import { variantName } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import { isZoomed, panBy, resetView, zoomAt, type ViewWindow } from '../lib/viewWindow'
import { WignerRenderer } from '../render/WignerRenderer'

const props = defineProps<{
  /** Register a frame handler with the session; returns unsubscribe. */
  frameSource: (h: (f: Frame) => void) => () => void
  /** Which variant of the bundle this panel shows. */
  variantIndex: number
  label?: string
  domain?: GridCfg
  showGrid?: boolean
  /** Zoom/pan window driving this panel. PanelGrid swaps it between the
   *  panel's own and the shared one when "link zoom" toggles. */
  view?: ViewWindow
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
const title = ref(props.label ?? '')
const glError = ref('')
const panning = ref(false)
const renderer = new WignerRenderer()
let unsub: (() => void) | null = null
let lastX = 0
let lastY = 0

/** Domain extents of the current view window, for the axis overlay. */
const viewDomain = computed(() => {
  const d = props.domain
  if (!d) return null
  const v = props.view ?? { x0: 0, x1: 1, y0: 0, y1: 1 }
  return {
    x1: d.x1 + v.x0 * (d.x2 - d.x1),
    x2: d.x1 + v.x1 * (d.x2 - d.x1),
    p1: d.p1 + v.y0 * (d.p2 - d.p1),
    p2: d.p1 + v.y1 * (d.p2 - d.p1),
  }
})

function onWheel(e: WheelEvent) {
  if (!props.view) return
  const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
  zoomAt(props.view,
    (e.clientX - r.left) / r.width,
    1 - (e.clientY - r.top) / r.height,
    e.deltaY < 0 ? 0.85 : 1 / 0.85)
}

function onDown(e: PointerEvent) {
  if (!props.view) return
  panning.value = true
  lastX = e.clientX
  lastY = e.clientY
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}

function onMove(e: PointerEvent) {
  if (!panning.value || !props.view) return
  const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
  // screen y is down, the p fraction is up
  panBy(props.view, (e.clientX - lastX) / r.width, -(e.clientY - lastY) / r.height)
  lastX = e.clientX
  lastY = e.clientY
}

function onUp() { panning.value = false }

function onDblClick() {
  if (props.view) resetView(props.view)
}

onMounted(() => {
  try {
    renderer.init(canvas.value!)
  } catch (e) {
    glError.value = String(e)
    return
  }
  // the session fan-out is already rAF-timed (one frame per animation
  // frame), so upload + draw run directly, once per painted frame
  unsub = props.frameSource((f: Frame) => {
    const v = f.variants[props.variantIndex]
    if (!v) return
    if (!props.label) title.value = variantName(v.vid)
    renderer.upload(v, f.Nx, f.Np)
    renderer.render()
  })
  // zoom/pan repaint: this is what redraws while PAUSED; the getter tracks
  // both the window's fields AND its identity (own <-> shared on "link
  // zoom" toggles); render() no-ops until the first frame arrives
  watch(
    () => (props.view ? [props.view.x0, props.view.x1, props.view.y0, props.view.y1] : null),
    (v) => {
      if (!v) return
      renderer.setView(v[0], v[1], v[2], v[3])
      renderer.render()
    },
  )
})

onBeforeUnmount(() => {
  unsub?.()
  renderer.dispose()
})
</script>

<template>
  <div class="relative w-full h-full min-h-0 touch-none"
       :class="view && isZoomed(view) ? (panning ? 'cursor-grabbing' : 'cursor-grab') : ''"
       @wheel.prevent="onWheel"
       @pointerdown="onDown" @pointermove="onMove" @pointerup="onUp"
       @pointercancel="onUp" @dblclick="onDblClick">
    <canvas ref="canvas" class="w-full h-full block bg-black"></canvas>
    <GridOverlay v-if="viewDomain && showGrid"
                 :x1="viewDomain.x1" :x2="viewDomain.x2"
                 :p1="viewDomain.p1" :p2="viewDomain.p2" />
    <div class="absolute top-1 left-2 text-xs text-white bg-black/75 px-1.5 py-0.5 rounded">
      {{ title }}
    </div>
    <div v-if="glError" class="absolute inset-0 grid place-items-center text-red-400 text-sm p-4">
      {{ glError }}
    </div>
  </div>
</template>

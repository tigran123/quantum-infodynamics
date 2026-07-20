<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import GridOverlay from './GridOverlay.vue'
import type { Frame } from '../lib/protocol'
import { variantName } from '../lib/protocol'
import type { GridCfg } from '../lib/config'
import { WignerRenderer } from '../render/WignerRenderer'

const props = defineProps<{
  /** Register a frame handler with the session; returns unsubscribe. */
  frameSource: (h: (f: Frame) => void) => () => void
  /** Which variant of the bundle this panel shows. */
  variantIndex: number
  label?: string
  domain?: GridCfg
  showGrid?: boolean
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
const title = ref(props.label ?? '')
const glError = ref('')
const renderer = new WignerRenderer()
let unsub: (() => void) | null = null

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
})

onBeforeUnmount(() => {
  unsub?.()
  renderer.dispose()
})
</script>

<template>
  <div class="relative w-full h-full min-h-0">
    <canvas ref="canvas" class="w-full h-full block bg-black"></canvas>
    <GridOverlay v-if="domain && showGrid"
                 :x1="domain.x1" :x2="domain.x2" :p1="domain.p1" :p2="domain.p2" />
    <div class="absolute top-1 left-2 text-xs text-white bg-black/75 px-1.5 py-0.5 rounded">
      {{ title }}
    </div>
    <div v-if="glError" class="absolute inset-0 grid place-items-center text-red-400 text-sm p-4">
      {{ glError }}
    </div>
  </div>
</template>

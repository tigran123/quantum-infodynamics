/**
 * Session lifecycle: REST create + WebSocket stream + control channel.
 *
 * Binary frames are decoded here and fan out to registered handlers via
 * direct callbacks — frame data NEVER enters Vue reactivity (the WebGL
 * renderer consumes it straight from the decoded views). Only small JSON
 * state (status, errors) is reactive.
 */

import { ref, shallowRef } from 'vue'
import { api } from '../api'
import { decodeFrame, type Frame } from '../lib/protocol'

export interface VariantStatus {
  variant: string
  dt: number
  device: string
  steps_per_sec: number
  steps_total: number
}

export interface SessionStatus {
  type: 'status'
  session_id: string
  running: boolean
  mode: 'interactive' | 'runahead'
  t2: number | null
  delay: number
  sign: number
  record_dt: number
  record_extent: [number, number]
  t_extent: [number | null, number | null]
  cursor: number
  history_bytes: number
  devices: string[]
  per_variant: VariantStatus[]
}

export interface SessionInfo {
  session_id: string
  ws_url: string
  variants: string[]
  record_dt: number
  warnings: string[]
}

export type FrameHandler = (f: Frame) => void

export function useSession() {
  const status = ref<SessionStatus | null>(null)
  const info = shallowRef<SessionInfo | null>(null)
  const connected = ref(false)
  const errors = ref<string[]>([])
  const lastFrame = shallowRef<Frame | null>(null)

  let ws: WebSocket | null = null
  let closedByUs = false
  const handlers = new Set<FrameHandler>()
  const closeHandlers = new Set<() => void>()

  async function create(cfg: unknown): Promise<SessionInfo> {
    await destroy()
    const { data } = await api.post<SessionInfo>('/sessions', cfg)
    info.value = data
    open(data.ws_url)
    return data
  }

  function open(wsUrl: string) {
    closedByUs = false
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${proto}//${location.host}${wsUrl}`)
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => { connected.value = true }
    ws.onclose = () => {
      connected.value = false
      // an UNEXPECTED close (backend restart, network drop) is the
      // caller's cue to reattach or recreate the session
      if (!closedByUs) closeHandlers.forEach((h) => h())
    }
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        const f = decodeFrame(ev.data)
        lastFrame.value = f
        handlers.forEach((h) => h(f))
      } else {
        const d = JSON.parse(ev.data as string)
        if (d.type === 'status') status.value = d as SessionStatus
        else if (d.type === 'error') errors.value.push(d.message)
        else if (d.type === 'eviction' && status.value)
          status.value.record_extent = d.new_extent
      }
    }
  }

  function send(cmd: Record<string, unknown>) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify(cmd))
    // Optimistic transport echo: on a busy stream the authoritative status
    // can queue behind binary frames for a while; the button must flip NOW
    // or the user's next click computes the WRONG command (a "pause" click
    // on a stale not-running status sends play). The server's echo (sent
    // ahead of frame bursts) confirms or corrects shortly after.
    if (status.value && (cmd.type === 'play' || cmd.type === 'pause'))
      status.value = { ...status.value, running: cmd.type === 'play' }
  }

  /** Register a raw-frame handler; returns the unsubscribe function.
   *  The newest frame (if any) is replayed to the handler immediately, so
   *  panels remounted mid-session (layout/grid toggles) are never blank
   *  while paused. */
  function onFrame(h: FrameHandler): () => void {
    handlers.add(h)
    if (lastFrame.value) h(lastFrame.value)
    return () => handlers.delete(h)
  }

  /** Register a handler for UNEXPECTED WebSocket closes. */
  function onClose(h: () => void): () => void {
    closeHandlers.add(h)
    return () => closeHandlers.delete(h)
  }

  /** Reattach the WebSocket to the existing (still alive) session. */
  function reconnect() {
    if (info.value) open(info.value.ws_url)
  }

  async function destroy() {
    closedByUs = true
    ws?.close()
    ws = null
    connected.value = false
    lastFrame.value = null   // never replay a dead session's frame
    if (info.value) {
      const sid = info.value.session_id
      info.value = null
      status.value = null
      try { await api.delete(`/sessions/${sid}`) } catch { /* already gone */ }
    }
  }

  return { status, info, connected, errors, lastFrame, create, send,
           onFrame, onClose, reconnect, destroy }
}

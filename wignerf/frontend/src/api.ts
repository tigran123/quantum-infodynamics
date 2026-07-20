import axios from 'axios'

// Base URL for REST calls. Prefer an explicit VITE_API_URL; otherwise derive
// it from the SPA base (import.meta.env.BASE_URL — Vite fills this from `base`
// in vite.config: "/" in dev, "/wignerf/" behind the nginx prod prefix).
// BASE_URL always ends in "/", so `${BASE_URL}api` yields "/api" or
// "/wignerf/api" with no double slash. The WebSocket URL is prefixed
// server-side (ws_url carries the backend's --root-path), not here.
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || `${import.meta.env.BASE_URL}api`,
})

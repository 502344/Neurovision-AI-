// All calls go through /api (Vite proxy in dev). Override with VITE_API_URL for deployed builds.
const BASE = import.meta.env.VITE_API_URL || '/api'

export const abs = (path) => (path ? `${BASE}${path}` : null)

async function getJson(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Request failed (${res.status})`)
  return res.json()
}

export const api = {
  health: () => getJson('/health'),
  modelInfo: () => getJson('/model-info'),
  metrics: () => getJson('/metrics'),
  datasetInfo: () => getJson('/dataset-info'),
}

/** Upload with real byte-level progress (XMLHttpRequest). Resolves with the JSON result or rejects with Error(detail). */
export function predict(file, onUploadProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/predict`)
    xhr.upload.onprogress = (e) => e.lengthComputable && onUploadProgress?.(e.loaded / e.total)
    xhr.onload = () => {
      let body = {}
      try { body = JSON.parse(xhr.responseText) } catch { /* non-JSON error */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else {
        const err = new Error(typeof body.detail === 'string' ? body.detail : `Analysis failed (${xhr.status})`)
        err.status = xhr.status
        reject(err)
      }
    }
    xhr.onerror = () => reject(new Error('Cannot reach the NEUROVISION AI backend. Is it running on port 8000?'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

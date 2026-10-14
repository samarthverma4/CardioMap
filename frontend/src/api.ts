import type { Anatomy, Meta, Metrics, Prediction, Presets, Schema } from './types'

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path} returned ${r.status}`)
  return r.json() as Promise<T>
}

export const api = {
  schema: () => getJSON<Schema>('/api/schema'),
  metrics: () => getJSON<Metrics>('/api/metrics'),
  presets: () => getJSON<Presets>('/api/presets'),
  anatomy: () => getJSON<Anatomy>('/api/anatomy'),
  meta: () => getJSON<Meta>('/api/meta'),
  async predict(features: Record<string, number | string>, signal?: AbortSignal): Promise<Prediction> {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ features }),
      signal,
    })
    const body = await r.json()
    if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Prediction failed (${r.status})`)
    return body as Prediction
  },
}

export const pct = (p: number) => `${Math.round(p * 100)}%`
export const auc = (a: number) => a.toFixed(2)

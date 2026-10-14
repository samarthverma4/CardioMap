import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import Bullseye from './components/Bullseye'
import ModelCard from './components/ModelCard'
import PatientForm from './components/PatientForm'
import Results from './components/Results'
import type { Anatomy, Meta, Metrics, Prediction, Preset, Presets, Schema, Vessel } from './types'

const Heart = lazy(() => import('./components/Heart'))

export const DISCLAIMER =
  'Decision support / education only. Not a diagnosis and not a substitute for angiography or CT angiography. The 3D heart is a generic model, not this patient’s anatomy.'

type Values = Record<string, number | string>
interface Static { schema: Schema; metrics: Metrics; presets: Presets; anatomy: Anatomy; meta: Meta }

function Legend({ meta }: { meta: Meta }) {
  const grad = `linear-gradient(90deg, ${meta.color_scale.map((s) => `${s.color} ${s.p * 100}%`).join(', ')})`
  return (
    <div className="scene-legend" aria-label="Legend">
      <div>
        <span className="lg-title">Colour: vessel score</span>
        <span className="lg-ramp" style={{ background: grad }} />
        <span className="lg-ends"><span>0%</span><span>50%</span><span>100%</span></span>
      </div>
      <div>
        <span className="lg-title">Strength: evidence</span>
        <span className="lg-ev"><i style={{ opacity: 1 }} />moderate</span>
        <span className="lg-ev"><i style={{ opacity: 0.7 }} />limited</span>
        <span className="lg-ev"><i className="weak" style={{ opacity: 0.45 }} />weak, dashed</span>
      </div>
    </div>
  )
}

export default function App() {
  const [s, setS] = useState<Static | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [tab, setTab] = useState<'patient' | 'model'>('patient')
  const [values, setValues] = useState<Values>({})
  const [preset, setPreset] = useState<Preset | null>(null)
  const [pred, setPred] = useState<Prediction | null>(null)
  const [predError, setPredError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [selected, setSelected] = useState<Vessel | null>(null)
  const [territory, setTerritory] = useState(true)
  const ctrl = useRef<AbortController | null>(null)

  useEffect(() => {
    Promise.all([api.schema(), api.metrics(), api.presets(), api.anatomy(), api.meta()])
      .then(([schema, metrics, presets, anatomy, meta]) => setS({ schema, metrics, presets, anatomy, meta }))
      .catch((e: Error) => setLoadError(`Could not reach the CardioMap API (${e.message}). Start the backend on port 8000 and reload.`))
  }, [])

  useEffect(() => {
    if (!s) return
    const t = setTimeout(() => {
      ctrl.current?.abort()
      const c = new AbortController()
      ctrl.current = c
      setBusy(true)
      api.predict(values, c.signal)
        .then((p) => { setPred(p); setPredError(null) })
        .catch((e: Error) => { if (e.name !== 'AbortError') setPredError(e.message) })
        .finally(() => { if (ctrl.current === c) setBusy(false) })
    }, 180)
    return () => clearTimeout(t)
  }, [values, s])

  const shown = useMemo(() => ({ ...(pred?.inputs ?? {}), ...values }), [pred, values])

  const loadPreset = (p: Preset | null) => {
    setPreset(p)
    setSelected(null)
    if (!p) return setValues({})
    const { BMI: _bmi, ...rest } = p.features // BMI is derived from weight and height by the API
    void _bmi
    setValues(rest)
  }

  return (
    <div className="app">
      <div className="disclaimer" role="note">{DISCLAIMER}</div>
      <header className="topbar">
        <div className="brand">
          <span className="wordmark">CardioMap</span>
          <span className="tagline">Which coronary artery? An honest answer from bedside data.</span>
        </div>
        <nav className="tabs" aria-label="Views">
          <button type="button" aria-current={tab === 'patient' ? 'page' : undefined} onClick={() => setTab('patient')}>Patient</button>
          <button type="button" aria-current={tab === 'model' ? 'page' : undefined} onClick={() => setTab('model')}>Model card</button>
        </nav>
      </header>

      {loadError && <p className="error-box">{loadError}</p>}
      {!s && !loadError && <p className="loading">Loading models and schema…</p>}

      {s && tab === 'patient' && (
        <main className="workspace">
          <aside className="col-form">
            <PatientForm
              schema={s.schema}
              values={shown}
              onChange={(name, v) => setValues((old) => ({ ...old, [name]: v }))}
              presets={s.presets.presets}
              activePreset={preset}
              onLoadPreset={loadPreset}
            />
          </aside>
          <section className="col-scene" aria-label="3D heart">
            <Suspense fallback={<div className="scene-loading">Loading 3D view…</div>}>
              <Heart prediction={pred} selected={selected} onSelect={setSelected} showTerritory={territory} />
            </Suspense>
            <div className="scene-top">
              <span className="scene-hint">Drag to rotate · scroll to zoom · click an artery</span>
              <label className="toggle">
                <input type="checkbox" checked={territory} onChange={(e) => setTerritory(e.target.checked)} />
                Tint selected territory
              </label>
            </div>
            <Legend meta={s.meta} />
            <Bullseye anatomy={s.anatomy} prediction={pred} selected={selected} onSelect={setSelected} />
            <p className="model-credit">
              Heart: <a href="https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089" target="_blank" rel="noreferrer">“Realistic Human Heart” by neshallads</a>, CC BY 4.0.
              {' '}Coronary arteries: <a href="https://dbarchive.biosciencedbc.jp/en/bodyparts3d/" target="_blank" rel="noreferrer">BodyParts3D</a>, © The Database Center for Life Science, CC BY 4.0.
            </p>
          </section>
          <aside className="col-results">
            {predError && <p className="error-box" role="alert">{predError}</p>}
            {pred && (
              <Results prediction={pred} selected={selected} onSelect={setSelected} preset={preset} busy={busy} />
            )}
          </aside>
        </main>
      )}
      {s && tab === 'model' && <ModelCard m={s.metrics} schema={s.schema} />}
    </div>
  )
}

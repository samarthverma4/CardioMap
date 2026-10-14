import { useState } from 'react'
import { pct } from '../api'
import type { Evidence, Prediction, Preset, ShapItem, Vessel } from '../types'
import { VESSELS } from '../types'

const EVIDENCE_COPY: Record<Evidence, string> = {
  weak: 'Weak evidence',
  limited: 'Limited evidence',
  moderate: 'Moderate evidence',
}

export function EvidenceBadge({ e, auc }: { e: Evidence; auc?: number }) {
  return (
    <span className={`badge badge-${e}`} title={auc !== undefined ? `Within-CAD AUC ${auc.toFixed(2)}` : undefined}>
      <svg viewBox="0 0 12 12" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <rect key={i} x={1 + i * 3.6} y={8 - i * 3} width="2.6" height={3 + i * 3} rx="0.6"
            className={i <= ({ weak: 0, limited: 1, moderate: 2 } as const)[e] ? 'on' : 'off'} />
        ))}
      </svg>
      {EVIDENCE_COPY[e]}
    </span>
  )
}

/* Signature: the vessel score on the same 0-100% track as P(CAD), so a reader sees at once how much of a
   vessel score is just overall CAD risk. */
function Track({ p, band, cadP, color }: { p: number; band: [number, number]; cadP: number; color: string }) {
  const x = (v: number) => `${(v * 100).toFixed(2)}%`
  return (
    <div className="track" aria-hidden="true">
      <div className="track-band" style={{ left: x(band[0]), width: x(Math.max(band[1] - band[0], 0.004)) }} />
      <div className="track-cad" style={{ left: x(cadP) }} />
      <div className="track-dot" style={{ left: x(p), background: color }} />
    </div>
  )
}

/* animKey: whenever it changes (a vessel is touched or the selection changes) the bars regrow from the axis. */
export function ShapChart({ items, scale, animKey }: { items: ShapItem[]; scale: string; animKey?: string }) {
  const [tip, setTip] = useState<number | null>(null)
  const max = Math.max(...items.map((s) => Math.abs(s.contribution)), 1e-6)
  const row = 28, labelW = 196, w = 360, half = (w - labelW) / 2
  return (
    <figure className="shap">
      <svg key={animKey} viewBox={`0 0 ${w} ${items.length * row + 6}`} role="img"
        aria-label={`Top ${items.length} input contributions on the ${scale} scale`}>
        <line x1={labelW + half} x2={labelW + half} y1={0} y2={items.length * row + 6} className="axis" />
        {items.map((s, i) => {
          const len = (Math.abs(s.contribution) / max) * (half - 6)
          const up = s.contribution > 0
          const y = i * row + 5
          return (
            <g key={s.feature} onMouseEnter={() => setTip(i)} onMouseLeave={() => setTip(null)}
              onFocus={() => setTip(i)} onBlur={() => setTip(null)} tabIndex={0}>
              <rect x={0} y={y - 3} width={w} height={row} className="hit" />
              <text x={0} y={y + 12} className="shap-label">{s.label}</text>
              <text x={0} y={y + 23} className="shap-value">{s.display_value}</text>
              <rect x={up ? labelW + half + 1 : labelW + half - 1 - len} y={y + 4} width={Math.max(len, 1.5)} height={11}
                rx={2} className={`${up ? 'bar-up grow-right' : 'bar-down grow-left'}`}
                style={{ animationDelay: `${i * 55}ms` }} />
              {tip === i && (
                <text x={up ? labelW + half - 4 : labelW + half + 4} y={y + 13} textAnchor={up ? 'end' : 'start'} className="shap-num">
                  {s.contribution > 0 ? '+' : ''}{scale === 'probability' ? `${(s.contribution * 100).toFixed(1)} pts` : s.contribution.toFixed(2)}
                </text>
              )}
            </g>
          )
        })}
      </svg>
      <figcaption>
        <span className="key"><i className="bar-up" /> raises score</span>
        <span className="key"><i className="bar-down" /> lowers score</span>
        <span>Scale: {scale === 'probability' ? 'probability points (random forest)' : 'log-odds (logistic regression)'}, uncalibrated base model</span>
      </figcaption>
    </figure>
  )
}

interface Props {
  prediction: Prediction
  selected: Vessel | null
  onSelect: (v: Vessel | null) => void
  preset: Preset | null
  busy: boolean
}

export default function Results({ prediction: d, selected, onSelect, preset, busy }: Props) {
  const [revealed, setRevealed] = useState<string | null>(null)
  const target = selected ? d.vessels[selected] : d.cad
  const note = d.population_note.toLowerCase()
  return (
    <section className="results" aria-busy={busy} aria-live="polite">
      <div className="cad-block">
        <p className="eyebrow">Probability of coronary artery disease</p>
        <button type="button" className="cad-readout" aria-pressed={!selected} onClick={() => onSelect(null)}>
          <span className="cad-num">{pct(d.cad.p)}</span>
          <span className="cad-band">fold-model range {pct(d.cad.band[0])}–{pct(d.cad.band[1])}</span>
        </button>
        <p className="pop">{d.population_note}</p>
        <div className="esc-compare">
          <span>Guideline (ESC) inputs alone</span>
          <b>{pct(d.cad.esc_baseline_p)}</b>
          <span aria-hidden="true">→</span>
          <span>all {Object.keys(d.inputs).length} inputs</span>
          <b>{pct(d.cad.p)}</b>
        </div>
        {d.defaults_used.length > 0 && (
          <p className="fine">{d.defaults_used.length} inputs are at dataset defaults (median or most common value).</p>
        )}
      </div>

      <div className="vessels">
        <div className="vessel-head">
          <p className="eyebrow">Per-vessel score, {note}</p>
          <span className="legend-mini"><i className="lm-dot" /> vessel <i className="lm-cad" /> P(CAD) <i className="lm-band" /> range</span>
        </div>
        {VESSELS.map((v) => {
          const r = d.vessels[v]
          return (
            <button key={v} type="button" className="vessel-row" aria-pressed={selected === v} onClick={() => onSelect(selected === v ? null : v)}>
              <span className="v-name">{v}</span>
              <span className="v-p">{pct(r.p)}</span>
              <span className="v-cad">P(CAD) {pct(r.cad_p)}</span>
              <Track p={r.p} band={r.band} cadP={r.cad_p} color={d.scene_state[v].color} />
              <span className="v-meta">
                <EvidenceBadge e={r.evidence} auc={r.within_cad_auc} />
                <span className="v-range">range {pct(r.band[0])}–{pct(r.band[1])}</span>
              </span>
            </button>
          )
        })}
        <p className="fine">
          Evidence badge = how well the model ranks this vessel among patients who do have CAD (within-CAD AUC). It does not
          locate a lesion.
        </p>
      </div>

      <div className="explain">
        <p className="eyebrow">{selected ? `Why the ${selected} score` : 'Why the CAD probability'}</p>
        <p className="sentence">{target.explanation}</p>
        {/* On wide screens the chart is shown over the 3D view instead (App.tsx); here only on phones. */}
        <div className="shap-inline"><ShapChart items={target.shap} scale={target.shap_scale} animKey={selected ?? 'CAD'} /></div>
      </div>

      {preset && (
        <div className="reveal">
          {revealed === preset.id ? (
            <div>
              <p className="eyebrow">Angiography result, dataset row {preset.dataset_row}</p>
              <p><b>{preset.hidden_labels.description}.</b> CAD: {preset.hidden_labels.CAD ? 'yes' : 'no'} · LAD {preset.hidden_labels.LAD ? 'stenotic' : 'normal'} · LCX {preset.hidden_labels.LCX ? 'stenotic' : 'normal'} · RCA {preset.hidden_labels.RCA ? 'stenotic' : 'normal'}</p>
              <p className="fine">This patient was in the training data, so these predictions are optimistic.</p>
            </div>
          ) : (
            <button type="button" className="btn" onClick={() => setRevealed(preset.id)}>Reveal angiography result</button>
          )}
        </div>
      )}
    </section>
  )
}

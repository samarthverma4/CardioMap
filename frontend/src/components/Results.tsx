import { useState } from 'react'
import { pct } from '../api'
import type { Anatomy, Evidence, Prediction, Preset, ShapItem, Vessel } from '../types'
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

function ShapChart({ items, scale }: { items: ShapItem[]; scale: string }) {
  const [tip, setTip] = useState<number | null>(null)
  const max = Math.max(...items.map((s) => Math.abs(s.contribution)), 1e-6)
  const row = 26, labelW = 168, w = 360, half = (w - labelW) / 2
  return (
    <figure className="shap">
      <svg viewBox={`0 0 ${w} ${items.length * row + 6}`} role="img"
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
              <text x={0} y={y + 22} className="shap-value">{s.display_value}</text>
              <rect x={up ? labelW + half + 1 : labelW + half - 1 - len} y={y + 4} width={Math.max(len, 1.5)} height={11}
                rx={2} className={up ? 'bar-up' : 'bar-down'} />
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

function Bullseye({ anatomy, vessel, color }: { anatomy: Anatomy; vessel: Vessel | null; color: string | null }) {
  // AHA 17-segment: rings of 6 (basal), 6 (mid), 4 (apical), 1 (apex). Segment 1 = anterior at top.
  const owner = (seg: number) => VESSELS.find((v) => anatomy.aha_17_segments[v].includes(seg))
  const arc = (r0: number, r1: number, a0: number, a1: number) => {
    const p = (r: number, a: number) => [100 + r * Math.sin(a), 100 - r * Math.cos(a)].map((n) => n.toFixed(2)).join(' ')
    const large = a1 - a0 > Math.PI ? 1 : 0
    return `M${p(r1, a0)} A${r1} ${r1} 0 ${large} 1 ${p(r1, a1)} L${p(r0, a1)} A${r0} ${r0} 0 ${large} 0 ${p(r0, a0)}Z`
  }
  const segs: { n: number; d: string }[] = []
  const ring = (start: number, count: number, r0: number, r1: number, offset: number) => {
    for (let i = 0; i < count; i++) {
      // AHA numbering runs counter-clockwise on the standard bullseye
      const a0 = offset - ((i + 1) * 2 * Math.PI) / count
      segs.push({ n: start + i, d: arc(r0, r1, a0, a0 + (2 * Math.PI) / count) })
    }
  }
  ring(1, 6, 66, 94, Math.PI / 6)
  ring(7, 6, 40, 66, Math.PI / 6)
  ring(13, 4, 16, 40, Math.PI / 4)
  return (
    <figure className="bullseye">
      <svg viewBox="0 0 200 200" role="img" aria-label={`AHA 17-segment bullseye${vessel ? `, ${vessel} territory highlighted` : ''}`}>
        {segs.map((s) => (
          <path key={s.n} d={s.d} className="seg-path" style={owner(s.n) === vessel && color ? { fill: color, fillOpacity: 0.55 } : undefined}>
            <title>Segment {s.n}: {owner(s.n)}</title>
          </path>
        ))}
        <circle cx="100" cy="100" r="16" className="seg-path"
          style={owner(17) === vessel && color ? { fill: color, fillOpacity: 0.55 } : undefined}><title>Segment 17 (apex): LAD</title></circle>
        <text x="100" y="12" textAnchor="middle" className="bull-dir">anterior</text>
      </svg>
      <figcaption>{vessel ? `${vessel} territory, ` : 'Territories, '}{anatomy.dominance_assumption}</figcaption>
    </figure>
  )
}

interface Props {
  prediction: Prediction
  anatomy: Anatomy
  selected: Vessel | null
  onSelect: (v: Vessel | null) => void
  preset: Preset | null
  busy: boolean
}

export default function Results({ prediction: d, anatomy, selected, onSelect, preset, busy }: Props) {
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
        <ShapChart items={target.shap} scale={target.shap_scale} />
      </div>

      <Bullseye anatomy={anatomy} vessel={selected} color={selected ? d.scene_state[selected].color : null} />

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

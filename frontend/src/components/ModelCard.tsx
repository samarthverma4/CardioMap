import { useState } from 'react'
import { auc, pct } from '../api'
import type { AucCI, Metrics, Schema, Vessel } from '../types'
import { VESSELS } from '../types'
import { EvidenceBadge } from './Results'

type Family = 'LR' | 'RF'
const FAMILY_NAME: Record<Family, string> = { LR: 'Logistic regression', RF: 'Random forest' }
const ci = (a: AucCI) => `${auc(a.auc)} (${auc(a.ci95[0])}–${auc(a.ci95[1])})`
const sgn = (x: number) => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(3)

interface Tip { x: number; y: number; text: string }

function useTip() {
  const [tip, setTip] = useState<Tip | null>(null)
  const node = tip && (
    <div className="chart-tip" style={{ left: tip.x, top: tip.y }} role="status">{tip.text}</div>
  )
  return { node, show: (e: React.MouseEvent | React.FocusEvent, text: string) => {
    const host = (e.currentTarget as Element).closest('.chart')!.getBoundingClientRect()
    const r = (e.currentTarget as Element).getBoundingClientRect()
    setTip({ x: r.left - host.left + r.width / 2, y: r.top - host.top, text })
  }, hide: () => setTip(null) }
}

const SERIES = [
  { key: 'dedicated', label: 'Dedicated vessel model, all patients', cls: 's1' },
  { key: 'cadprob', label: 'CAD probability used as the vessel score', cls: 's2' },
  { key: 'within', label: 'Dedicated model, within CAD patients only', cls: 's3' },
] as const

function VesselAucChart({ m, fam }: { m: Metrics; fam: Family }) {
  const { node, show, hide } = useTip()
  const W = 620, H = 280, L = 40, Bt = 34, T = 12
  const y = (v: number) => T + (1 - (v - 0.4) / 0.6) * (H - T - Bt)
  const groupW = (W - L) / 3, barW = 30, gap = 6
  const val = (v: Vessel, k: (typeof SERIES)[number]['key']): AucCI => {
    const d = m[fam].vessels[v]
    return k === 'dedicated' ? d.dedicated_all_patients : k === 'cadprob' ? d.cad_probability_as_vessel_score : d.within_CAD.dedicated_model_scores
  }
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Vessel AUC by scoring approach, ${FAMILY_NAME[fam]}`}>
        {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => (
          <g key={t}>
            <line x1={L} x2={W} y1={y(t)} y2={y(t)} className={t === 0.5 ? 'axis' : 'grid'} strokeDasharray={t === 0.5 ? '4 3' : undefined} />
            <text x={L - 6} y={y(t) + 4} textAnchor="end" className="tick">{t.toFixed(1)}</text>
          </g>
        ))}
        <text x={L + 4} y={y(0.5) - 5} className="tick">chance</text>
        {VESSELS.map((v, gi) => {
          const x0 = L + gi * groupW + (groupW - (3 * barW + 2 * gap)) / 2
          return (
            <g key={v}>
              {SERIES.map((s, si) => {
                const a = val(v, s.key)
                const x = x0 + si * (barW + gap)
                const text = `${v} · ${s.label}: AUC ${ci(a)}`
                return (
                  <g key={s.key} tabIndex={0} onMouseEnter={(e) => show(e, text)} onMouseLeave={hide} onFocus={(e) => show(e, text)} onBlur={hide}>
                    <rect x={x - 3} y={T} width={barW + 6} height={H - T - Bt} className="hit" />
                    <path className={`bar ${s.cls}`}
                      d={`M${x},${y(0.5)} V${y(a.auc) + 4} q0,-4 4,-4 h${barW - 8} q4,0 4,4 V${y(0.5)} Z`} />
                    <line x1={x + barW / 2} x2={x + barW / 2} y1={y(a.ci95[0])} y2={y(a.ci95[1])} className="whisker" />
                    <text x={x + barW / 2} y={y(a.ci95[1]) - 5} textAnchor="middle" className="bar-val">{auc(a.auc)}</text>
                  </g>
                )
              })}
              <text x={x0 + (3 * barW + 2 * gap) / 2} y={H - 12} textAnchor="middle" className="cat">{v}</text>
            </g>
          )
        })}
      </svg>
      {node}
      <div className="legend">
        {SERIES.map((s) => <span key={s.key}><i className={`sw ${s.cls}`} />{s.label}</span>)}
        <span><i className="sw whisker-sw" />95% bootstrap CI</span>
      </div>
    </div>
  )
}

const STEPS = [
  { key: 'esc_features', label: 'ESC inputs (12)' },
  { key: 'esc+ecg', label: '+ ECG' },
  { key: 'esc+ecg+echo', label: '+ echo' },
  { key: 'all', label: 'All inputs' },
]

function AblationChart({ m }: { m: Metrics }) {
  const { node, show, hide } = useTip()
  const W = 620, H = 240, L = 46, R = 20, T = 14, Bt = 34
  const fams: Family[] = ['LR', 'RF']
  const all = fams.flatMap((f) => STEPS.flatMap((s) => m[f].ablation[s.key].CAD.ci95))
  const lo = Math.floor(Math.min(...all) * 50) / 50, hi = Math.ceil(Math.max(...all) * 50) / 50
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - Bt)
  const x = (i: number, f: number) => L + ((i + 0.5) * (W - L - R)) / STEPS.length + (f - 0.5) * 24
  const ticks: number[] = []
  for (let t = lo; t <= hi + 1e-9; t += 0.02) ticks.push(+t.toFixed(2))
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="CAD AUC as modalities are added">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
            <text x={L - 6} y={y(t) + 4} textAnchor="end" className="tick">{t.toFixed(2)}</text>
          </g>
        ))}
        {fams.map((f, fi) => (
          <g key={f} className={`line-${fi === 0 ? 's1' : 's2'}`}>
            <polyline fill="none" className="series-line"
              points={STEPS.map((s, i) => `${x(i, fi)},${y(m[f].ablation[s.key].CAD.auc)}`).join(' ')} />
            {STEPS.map((s, i) => {
              const a = m[f].ablation[s.key].CAD
              const text = `${FAMILY_NAME[f]} · ${s.label}: CAD AUC ${ci(a)}`
              return (
                <g key={s.key} tabIndex={0} onMouseEnter={(e) => show(e, text)} onMouseLeave={hide} onFocus={(e) => show(e, text)} onBlur={hide}>
                  <rect x={x(i, fi) - 9} y={T} width={18} height={H - T - Bt} className="hit" />
                  <line x1={x(i, fi)} x2={x(i, fi)} y1={y(a.ci95[0])} y2={y(a.ci95[1])} className="whisker" />
                  <circle cx={x(i, fi)} cy={y(a.auc)} r={4.5} className="dot" />
                </g>
              )
            })}
          </g>
        ))}
        {STEPS.map((s, i) => (
          <text key={s.key} x={x(i, 0.5)} y={H - 12} textAnchor="middle" className="cat">{s.label}</text>
        ))}
      </svg>
      {node}
      <div className="legend">
        <span><i className="sw s1" />{FAMILY_NAME.LR}</span>
        <span><i className="sw s2" />{FAMILY_NAME.RF}</span>
        <span><i className="sw whisker-sw" />95% bootstrap CI</span>
      </div>
    </div>
  )
}

export default function ModelCard({ m, schema }: { m: Metrics; schema: Schema }) {
  const [fam, setFam] = useState<Family>('LR')
  const d = m.derived
  const escDiff = (f: Family) => m[f].ablation_diffs.CAD['esc+ecg+echo minus esc_features']
  const within = VESSELS.flatMap((v) => (['LR', 'RF'] as Family[]).map((f) => m[f].vessels[v].within_CAD.dedicated_model_scores.auc))
  const flagged = m.counts.rows_vessel_stenotic_but_cath_normal
  return (
    <article className="model-card">
      <header className="mc-head">
        <h2>Model card</h2>
        <p>
          {m.n} patients referred for angiography ({m.counts.CAD} with CAD). {m.design.cv}, seed {m.design.seed},
          {' '}{m.design.bootstrap.toLocaleString()} bootstrap resamples. All numbers below are read from
          {' '}<code>artifacts/metrics.json</code>.
        </p>
      </header>

      <section>
        <h3>Do vessel models know more than overall CAD risk?</h3>
        <p className="lede">
          Using the CAD probability as the vessel score ranks patients about as well as a dedicated vessel model. Among
          patients who have CAD, vessel AUCs fall to {auc(Math.min(...within))}–{auc(Math.max(...within))}: the models
          mostly recognise who has disease, not where it is.
        </p>
        <div className="seg seg-inline" role="group" aria-label="Model family">
          {(['LR', 'RF'] as Family[]).map((f) => (
            <button key={f} type="button" aria-pressed={fam === f} onClick={() => setFam(f)}>{FAMILY_NAME[f]}</button>
          ))}
        </div>
        <VesselAucChart m={m} fam={fam} />
        <table>
          <caption>Dedicated vessel model minus CAD probability as the score (AUC difference, paired bootstrap)</caption>
          <thead><tr><th>Vessel</th><th>{FAMILY_NAME.LR}</th><th>{FAMILY_NAME.RF}</th><th>Hierarchical minus dedicated (LR / RF)</th><th>Evidence</th></tr></thead>
          <tbody>
            {VESSELS.map((v) => (
              <tr key={v}>
                <th scope="row">{v}</th>
                {(['LR', 'RF'] as Family[]).map((f) => {
                  const x = m[f].vessels[v].dedicated_minus_cadprob
                  return <td key={f}>{sgn(x.diff)} ({x.ci95[0].toFixed(3)} to {x.ci95[1].toFixed(3)}){x.ci_excludes_0 ? ' *' : ''}</td>
                })}
                <td>{sgn(m.LR.vessels[v].hierarchical_minus_dedicated.diff)} / {sgn(m.RF.vessels[v].hierarchical_minus_dedicated.diff)}</td>
                <td><EvidenceBadge e={d.evidence[v].badge} /> {auc(d.evidence[v].within_cad_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="fine">
          * 95% CI excludes zero. Where it does, the dedicated model is <em>worse</em> than simply using the CAD
          probability. Evidence badge: {d.evidence_rule}.
        </p>
      </section>

      <section>
        <h3>What each modality adds to CAD detection</h3>
        <p className="lede">
          ECG and echo on top of the 12 guideline inputs: {FAMILY_NAME.LR} {sgn(escDiff('LR').diff)} AUC
          (95% CI {escDiff('LR').ci95[0].toFixed(3)} to {escDiff('LR').ci95[1].toFixed(3)}), {FAMILY_NAME.RF}
          {' '}{sgn(escDiff('RF').diff)} ({escDiff('RF').ci95[0].toFixed(3)} to {escDiff('RF').ci95[1].toFixed(3)}).
        </p>
        <AblationChart m={m} />
      </section>

      <section>
        <h3>Discrimination by target (all inputs)</h3>
        <table>
          <thead><tr><th>Target</th><th>{FAMILY_NAME.LR} AUC (95% CI)</th><th>{FAMILY_NAME.RF} AUC (95% CI)</th><th>Served model</th><th>Selection CV AUC (LR / RF)</th></tr></thead>
          <tbody>
            {['CAD', ...VESSELS].map((t) => (
              <tr key={t}>
                <th scope="row">{t}</th>
                <td>{ci(m.LR.ablation.all[t])}</td>
                <td>{ci(m.RF.ablation.all[t])}</td>
                <td>{d.served_model_selection[t].chosen}</td>
                <td>{auc(d.served_model_selection[t].inner_cv_auc.LR)} / {auc(d.served_model_selection[t].inner_cv_auc.RF)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="fine">
          The served model per target was chosen by a separate 5-fold CV on the training data, never by the reported numbers.
          Served models are refit on all {m.n} rows and sigmoid-calibrated (5-fold). The shaded range on each score is the
          spread of 5 fold models and shows model variability, not a confidence interval for this patient.
        </p>
      </section>

      <section>
        <h3>Calibration of the CAD model</h3>
        <table>
          <thead><tr><th>Group</th><th>n</th><th>CAD prevalence</th><th>Mean predicted (LR / RF)</th><th>Calibration slope (LR / RF)</th><th>Brier (LR / RF)</th></tr></thead>
          <tbody>
            {Object.keys(m.LR.cad_calibration_by_sex).map((g) => {
              const a = m.LR.cad_calibration_by_sex[g], b = m.RF.cad_calibration_by_sex[g]
              return (
                <tr key={g}>
                  <th scope="row">{g}</th><td>{a.n}</td><td>{pct(a.prevalence)}</td>
                  <td>{pct(a.mean_predicted)} / {pct(b.mean_predicted)}</td>
                  <td>{a.calibration_slope.toFixed(2)} / {b.calibration_slope.toFixed(2)}</td>
                  <td>{a.brier.toFixed(3)} / {b.brier.toFixed(3)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="fine">
          From uncalibrated out-of-fold predictions. A slope above 1 means predictions are too cautious (too close to the
          average); the served models add sigmoid calibration. Percentages are {schemaPop}.
        </p>
      </section>

      <section className="two-col">
        <div>
          <h3>Data quality</h3>
          <ul>
            {flagged.map((r) => (
              <li key={r}>Row {r} (0-based) has a stenotic vessel but is labelled no CAD. It is kept as published.</li>
            ))}
            {Object.entries(schema.dropped).map(([k, why]) => (
              <li key={k}><code>{k}</code> is dropped: {why}.</li>
            ))}
            <li>Vessel labels are reported as stenotic or normal only; lesion severity and location within a vessel are not recorded.</li>
          </ul>
        </div>
        <div>
          <h3>Limitations</h3>
          <ul>
            <li>One centre, {m.n} patients, all already referred for angiography. Probabilities do not transfer to screening or emergency populations.</li>
            <li>No external validation. CIs come from resampling one dataset and ignore refit variance, so they are somewhat narrow.</li>
            <li>Vessel scores mostly re-encode overall CAD risk. They must not be read as lesion location.</li>
            <li>SHAP values explain the uncalibrated base model, so they are on a different scale from the displayed probability.</li>
            <li>The 3D heart and territory map are generic and assume right-dominant circulation.</li>
          </ul>
        </div>
      </section>
      <p className="fine">Served with Python {d.served_versions.python}, scikit-learn {d.served_versions.sklearn}. Analysis run: Python {m.python}, scikit-learn {m.sklearn}. Dataset SHA-256 {m.dataset_sha256.slice(0, 16)}…</p>
    </article>
  )
}

const schemaPop = 'among patients referred for angiography'

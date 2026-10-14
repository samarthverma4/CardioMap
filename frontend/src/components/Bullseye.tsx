import type { Anatomy, Prediction, Vessel } from '../types'
import { VESSELS } from '../types'

/* AHA 17-segment bullseye, drawn as an overlay on the 3D view. Rings of 6 (basal), 6 (mid), 4 (apical), 1 (apex);
   segment 1 = basal anterior at the top, numbering counter-clockwise, septum on the left.
   No selection: every segment in its artery's colour. Selection: that territory strong, the rest faint. */
interface Props {
  anatomy: Anatomy
  prediction: Prediction | null
  selected: Vessel | null
  onSelect: (v: Vessel | null) => void
}

const C = 100
function arc(r0: number, r1: number, a0: number, a1: number) {
  const p = (r: number, a: number) => [C + r * Math.sin(a), C - r * Math.cos(a)].map((n) => n.toFixed(2)).join(' ')
  const large = a1 - a0 > Math.PI ? 1 : 0
  return `M${p(r1, a0)} A${r1} ${r1} 0 ${large} 1 ${p(r1, a1)} L${p(r0, a1)} A${r0} ${r0} 0 ${large} 0 ${p(r0, a0)}Z`
}

const SEGMENTS: { n: number; d: string }[] = (() => {
  const out: { n: number; d: string }[] = []
  const ring = (start: number, count: number, r0: number, r1: number, offset: number) => {
    for (let i = 0; i < count; i++) {
      const a0 = offset - ((i + 1) * 2 * Math.PI) / count
      out.push({ n: start + i, d: arc(r0, r1, a0, a0 + (2 * Math.PI) / count) })
    }
  }
  ring(1, 6, 66, 94, Math.PI / 6)
  ring(7, 6, 40, 66, Math.PI / 6)
  ring(13, 4, 16, 40, Math.PI / 4)
  return out
})()

export default function Bullseye({ anatomy, prediction, selected, onSelect }: Props) {
  const owner = (seg: number) => VESSELS.find((v) => anatomy.aha_17_segments[v].includes(seg))!
  const fill = (seg: number) => {
    const v = owner(seg)
    const color = prediction?.scene_state[v].color
    if (!color) return undefined
    const opacity = selected ? (v === selected ? 0.9 : 0.12) : 0.5
    return { fill: color, fillOpacity: opacity }
  }
  const seg = (n: number, d: string | null) => {
    const v = owner(n)
    const common = {
      className: 'bull-seg',
      style: fill(n),
      onClick: () => onSelect(selected === v ? null : v),
    }
    const title = <title>{`Segment ${n}${n === 17 ? ' (apex)' : ''}: ${v} territory`}</title>
    return d
      ? <path key={n} d={d} {...common}>{title}</path>
      : <circle key={n} cx={C} cy={C} r={16} {...common}>{title}</circle>
  }
  return (
    <figure className="bullseye-overlay" aria-label="AHA 17-segment territories">
      <figcaption className="bull-title">AHA 17-segment map</figcaption>
      <svg viewBox="-22 -22 244 244" role="img"
        aria-label={`AHA 17-segment bullseye${selected ? `, ${selected} territory highlighted` : ''}`}>
        {SEGMENTS.map((s) => seg(s.n, s.d))}
        {seg(17, null)}
        <text x={C} y={-9} textAnchor="middle" className="bull-dir">anterior</text>
        <text x={C} y={216} textAnchor="middle" className="bull-dir">inferior</text>
        <text x={-14} y={C} textAnchor="middle" className="bull-dir" transform={`rotate(-90 -14 ${C})`}>septal</text>
        <text x={214} y={C} textAnchor="middle" className="bull-dir" transform={`rotate(90 214 ${C})`}>lateral</text>
      </svg>
      <p className="bull-note">{selected ? `${selected} territory, ` : ''}{anatomy.dominance_assumption}</p>
    </figure>
  )
}

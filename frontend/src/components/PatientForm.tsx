import type { Preset, Schema, SchemaInput } from '../types'

const GROUP_LABELS: Record<string, string> = {
  demographic: 'Demographics',
  history: 'History & risk factors',
  symptoms: 'Symptoms',
  exam: 'Examination',
  ecg: 'ECG',
  lab: 'Laboratory',
  echo: 'Echocardiography',
}

type Values = Record<string, number | string>

interface Props {
  schema: Schema
  values: Values
  onChange: (name: string, v: number | string) => void
  presets: Preset[]
  activePreset: Preset | null
  onLoadPreset: (p: Preset | null) => void
}

function Field({ it, value, onChange }: { it: SchemaInput; value: number | string; onChange: Props['onChange'] }) {
  const id = `f-${it.name.replace(/\W+/g, '-')}`
  const esc = it.in_esc_set ? <abbr className="esc-dot" title="One of the 12 guideline (ESC) inputs">ESC</abbr> : null

  if (it.type === 'binary') {
    return (
      <div className="field field-binary" role="group" aria-labelledby={id}>
        <span id={id} className="field-label">{it.label}{esc}</span>
        <div className="seg">
          {it.options!.map((o) => (
            <button
              key={String(o.value)}
              type="button"
              aria-pressed={value === o.value}
              onClick={() => onChange(it.name, o.value)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>
    )
  }
  if (it.type === 'ordinal' || it.type === 'categorical') {
    return (
      <label className="field" htmlFor={id}>
        <span className="field-label">{it.label}{esc}{it.unit && <em>{it.unit}</em>}</span>
        <select
          id={id}
          value={String(value)}
          onChange={(e) => onChange(it.name, it.type === 'ordinal' ? Number(e.target.value) : e.target.value)}
        >
          {it.options!.map((o) => (
            <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
          ))}
        </select>
      </label>
    )
  }
  const derived = !!it.derived_from
  const [lo, hi] = it.range!
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{it.label}{esc}{it.unit && <em>{it.unit}</em>}</span>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={lo}
        max={hi}
        step={it.step ?? 1}
        value={typeof value === 'number' ? Number(value.toFixed(derived ? 1 : 2)) : value}
        readOnly={derived}
        title={derived ? it.derived_from : `Training range ${lo}–${hi}`}
        onChange={(e) => e.target.value !== '' && onChange(it.name, Number(e.target.value))}
      />
    </label>
  )
}

export default function PatientForm({ schema, values, onChange, presets, activePreset, onLoadPreset }: Props) {
  return (
    <form className="patient-form" onSubmit={(e) => e.preventDefault()} aria-label="Patient inputs">
      <div className="preset-bar">
        <label htmlFor="preset" className="eyebrow">Load sample patient</label>
        <select
          id="preset"
          value={activePreset?.id ?? ''}
          onChange={(e) => onLoadPreset(presets.find((p) => p.id === e.target.value) ?? null)}
        >
          <option value="">Dataset defaults</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>{p.name} (dataset row {p.dataset_row})</option>
          ))}
        </select>
      </div>
      {schema.groups.map((g) => {
        const items = schema.inputs.filter((i) => i.group === g)
        return (
          <details key={g}>
            <summary>
              <span>{GROUP_LABELS[g] ?? g}</span>
              <span className="count">{items.length}</span>
            </summary>
            <div className="fields">
              {items.map((it) => (
                <Field key={it.name} it={it} value={values[it.name] ?? it.default} onChange={onChange} />
              ))}
            </div>
          </details>
        )
      })}
      <p className="fine">{schema.range_note}</p>
    </form>
  )
}

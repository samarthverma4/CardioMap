export type InputType = 'binary' | 'ordinal' | 'continuous' | 'categorical'
export type Evidence = 'weak' | 'limited' | 'moderate'
export type Vessel = 'LAD' | 'LCX' | 'RCA'
export const VESSELS: Vessel[] = ['LAD', 'LCX', 'RCA']

export interface SchemaInput {
  name: string
  label: string
  unit: string
  type: InputType
  group: string
  in_esc_set: boolean
  range?: [number, number]
  default: number | string
  step?: number
  options?: { value: number | string; label: string }[]
  derived_from?: string
}

export interface Schema {
  groups: string[]
  esc_features: string[]
  ecg_features: string[]
  echo_features: string[]
  label_fields_rejected: string[]
  dropped: Record<string, string>
  range_note: string
  inputs: SchemaInput[]
}

export interface ShapItem {
  feature: string
  label: string
  value: number | string
  display_value: string
  unit: string
  contribution: number
}

export interface TargetResult {
  p: number
  band: [number, number]
  model: string
  shap: ShapItem[]
  shap_scale: 'log-odds' | 'probability'
  explanation: string
}

export interface CadResult extends TargetResult {
  esc_baseline_p: number
  esc_baseline_band: [number, number]
}

export interface VesselResult extends TargetResult {
  cad_p: number
  evidence: Evidence
  within_cad_auc: number
}

export interface Prediction {
  cad: CadResult
  vessels: Record<Vessel, VesselResult>
  scene_state: Record<Vessel, { color: string; evidence: Evidence }>
  population_note: string
  defaults_used: string[]
  inputs: Record<string, number | string>
}

export interface Preset {
  id: string
  name: string
  dataset_row: number
  features: Record<string, number | string>
  hidden_labels: { description: string; CAD: number; LAD: number; LCX: number; RCA: number }
  in_training_data: boolean
}

export interface Presets {
  note: string
  presets: Preset[]
}

export interface Anatomy {
  dominance_assumption: string
  note: string
  aha_17_segments: Record<Vessel, number[]>
  vessels: Record<Vessel, { name: string; course: string; territory: string }>
}

export interface AucCI {
  auc: number
  ci95: [number, number]
}
export interface DiffCI {
  diff: number
  ci95: [number, number]
  ci_excludes_0: boolean
}

export interface FamilyMetrics {
  ablation: Record<string, Record<string, AucCI & { mean_of_repeats: number; sd_of_repeats: number }>>
  ablation_diffs: Record<string, Record<string, DiffCI>>
  vessels: Record<
    Vessel,
    {
      dedicated_all_patients: AucCI
      cad_probability_as_vessel_score: AucCI
      dedicated_minus_cadprob: DiffCI
      'hierarchical_P(CAD)xP(V|CAD)': AucCI
      hierarchical_minus_dedicated: DiffCI
      within_CAD: {
        dedicated_model_scores: AucCI
        conditional_model_trained_on_CAD_only: AucCI
        cad_probability: AucCI
        conditional_minus_dedicated: DiffCI
      }
    }
  >
  cad_calibration_by_sex: Record<
    string,
    { n: number; prevalence: number; mean_predicted: number; brier: number; calibration_slope: number; auc: number }
  >
}

export interface Metrics {
  dataset_sha256: string
  n: number
  python: string
  sklearn: string
  design: { cv: string; seed: number; bootstrap: number }
  counts: {
    CAD: number
    LAD: number
    LCX: number
    RCA: number
    n_vessels: Record<string, number>
    rows_vessel_stenotic_but_cath_normal: number[]
  }
  LR: FamilyMetrics
  RF: FamilyMetrics
  derived: {
    evidence: Record<Vessel, { within_cad_auc: number; by_model: Record<string, number>; badge: Evidence }>
    evidence_rule: string
    served_model_selection: Record<string, { chosen: string; inner_cv_auc: Record<string, number> }>
    served_versions: Record<string, string>
  }
}

export interface Meta {
  disclaimer: string
  population_note: string
  color_scale: { p: number; color: string }[]
}

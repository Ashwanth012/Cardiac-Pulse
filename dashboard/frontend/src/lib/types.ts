export type Provenance = "observed" | "historical_forecast" | "sensitivity" | "exploratory" | "evaluation";

export interface Kpi {
  id: string;
  label: string;
  value: number | [number, number];
  unit: string;
  detail: string;
  provenance: Provenance;
  series?: string;
  delta_vs_avg_pct?: number;
}

export interface Alert {
  id: string;
  severity: "info" | "warning" | "critical";
  title: string;
  detail: string;
}

export interface Insight {
  id: string;
  title: string;
  detail: string;
  provenance: Provenance;
}

export interface TrendPoint {
  date: string;
  admissions: number;
  trailing_28d_mean: number | null;
  low_count_flag: boolean;
  potentially_incomplete: boolean;
}

export interface Overview {
  kpis: Kpi[];
  alerts: Alert[];
  insights: Insight[];
  trend: { window_days: number; end: string; series: string; points: TrendPoint[] };
  forecast_fallback_used: boolean;
}

export interface ForecastRow {
  date: string;
  weekday: string;
  mean: number;
  lower_80: number;
  upper_80: number;
  holidays: string[];
}

export interface ObservedPoint {
  date: string;
  admissions: number;
  trailing_28d_mean: number;
  potentially_incomplete: boolean;
}

export interface HistoricalForecast {
  kind: "historical_forecast";
  label: string;
  cutoff: string;
  fallback_used: boolean;
  fallback_messages: string[];
  forecast: ForecastRow[];
  observed_before_cutoff: ObservedPoint[];
}

export interface Meta {
  data: {
    start: string;
    end: string;
    days: number;
    potentially_incomplete: { date: string; admissions: number; median_previous_28_days: number; status: string };
  };
  historical_forecast: { cutoff: string; horizon_start: string; horizon_end: string; label: string };
  model: {
    name: string;
    id: string;
    file: string;
    sha256: string;
    frozen: boolean;
    features: string[];
    not_used: string[];
    interval: { level: string; q_lo: number; q_hi: number; calibration_origins: number; meaning: string } & IntervalEvidence;
    validation: Record<string, number>;
    holdout: Record<string, number>;
  };
}

// ------------------------------------------------------------------ Phase 2
export interface ValuePoint {
  date: string;
  value: number;
}

export interface RetroForecast {
  kind: "retrospective_forecast";
  label: string;
  cutoff: string;
  allowed_range: { min: string; max: string };
  fallback_used: boolean;
  forecast: ForecastRow[];
  observed_before_cutoff: { date: string; admissions: number }[];
  actual_after_cutoff: { series: "raw"; label: string; points: ValuePoint[] };
  actual_after_cutoff_dedup: { series: "dedup_mrd"; label: string; points: ValuePoint[] };
  baseline: { name: string; points: ValuePoint[] };
  metrics: {
    MAE: number; RMSE: number; WAPE: number; bias: number; coverage80: number; days_inside: number;
    width80: number; MAE_baseline: number; MAE_ratio_vs_baseline: number | null; scored_against: string;
  };
}

export interface HoldoutMetricSet {
  MAE: number; RMSE: number; WAPE: number; bias: number; MAE_h1_7: number; MAE_h8_14: number;
  MAE_ratio_vs_B0: number; coverage80: number; coverage80_h1_7: number; coverage80_h8_14: number;
  width80: number; n_cutoffs: number; n_points: number;
}

export interface HoldoutEvaluation {
  kind: "holdout_evaluation";
  label: string;
  frozen_sha256_at_evaluation: string;
  sha256_matches_current: boolean;
  metrics: Record<"primary" | "daily" | "daily_excl_last_day", { model: HoldoutMetricSet; baseline: HoldoutMetricSet }>;
  ratio_ci: { primary: [number, number, number]; daily: [number, number, number] };
  by_horizon: { h: number; MAE: number; MAE_B0: number; coverage80: number }[];
  by_month: { cutoff_month: string; n_cutoffs: number; MAE: number; MAE_B0: number; ratio: number; coverage80: number; width80: number }[];
  per_cutoff: { cutoff: string; MAE: number; MAE_B0: number; coverage80: number; width80: number; ratio: number }[];
  stages: { stage: string; cutoffs: number; MAE: number; ratio_vs_B0: number; coverage80: number | null }[];
  diagnosis_by_month: {
    month: string; forecast: number; actual_raw: number; actual_dedup: number; gap_raw: number; gap_dedup: number;
    gap_from_duplicates: number; repeat_rows_pct: number; raw_same_month_2018: number; dedup_same_month_2018: number;
  }[];
  interval_evidence: IntervalEvidence;
  notes: string[];
}

export interface ModelExplain {
  cutoff: string;
  training: { start: string; end: string; days: number };
  ridge: number;
  fourier_harmonics: number;
  weekday: { day: string; multiplier: number }[];
  calendar: { term: string; multiplier: number }[];
  annual_curve: { day_of_year: number; label: string; multiplier: number }[];
  reproduces_forecast: boolean;
  note: string;
}

export interface IntervalEvidence {
  target_coverage: number;
  residual_quantiles: { lower: number; upper: number; central_range: number };
  calibration_origins: number;
  construction: string;
  selection: string;
  nominal_80_tuning_coverage_range: [number, number];
  coverage_history: { stage: "tuning" | "validation" | "holdout"; label: string; coverage: number; points: number | null }[];
  holdout_misses: { above_upper: number; below_lower: number; inside: number; points: number; source: string };
}

// ------------------------------------------------------------------ Phase 3A: data quality
export type LogAction = "corrected" | "set_missing" | "imputed" | "flagged";

export interface QualitySummary {
  period: { start: string; end: string; days: number };
  scored_target: "raw";
  records: {
    raw: number; dedup_exact: number; dedup_mrd: number; exact_extra: number; mrd_extra: number; near_extra: number;
    exact_extra_rate: number; mrd_extra_rate: number;
  };
  days: {
    zero_count: number;
    low_count: number;
    potentially_incomplete: { date: string; admissions: number; median_previous_28_days: number; status: string };
    low_count_days: { date: string; weekday: string; admissions: number; holidays: string[]; potentially_incomplete: boolean }[];
  };
  environment_cells: Record<LogAction, number>;
  cleaning: {
    total: number;
    by_source_action: { source: "admissions" | "pollution"; action: LogAction; entries: number }[];
    by_rule: { source: "admissions" | "pollution"; rule: string; description: string; entries: number;
      actions: Partial<Record<LogAction, number>>; fields?: string[] }[];
    actions: Record<LogAction, string>;
    admissions_note: string;
  };
}

export interface RepeatRates { rows: number; exact_extra: number; mrd_extra: number; exact_rate: number; mrd_rate: number }

export interface QualityDuplicates {
  provenance: Provenance;
  scored_series: "raw";
  definitions: { exact: string; mrd_date: string };
  note: string;
  monthly: (RepeatRates & { month: string; near_extra: number })[];
  weekday: (RepeatRates & { weekday: string })[];
  periods: (RepeatRates & { label: string; start: string; end: string })[];
  daily: { date: string; raw: number; dedup_exact: number; dedup_mrd: number; extra_exact: number; extra_mrd: number;
    low_count_flag: boolean; potentially_incomplete: boolean }[];
}

export interface QualityMissing {
  provenance: Provenance;
  period: { start: string; end: string; days: number };
  months: string[];
  admissions: { missing_days: number; zero_count_days: number };
  groups: { group: string; label: string; variables: { variable: string; label: string; unit: string; missing: number; imputed: number;
    months: { month: string; days: number; missing: number; imputed: number }[] }[] }[];
  notes: string[];
}

export interface PollutionLog {
  provenance: Provenance;
  source: string;
  entries: number;
  actions: Record<LogAction, string>;
  rows: { date: string; field: string; original_value: string | null; cleaned_value: string | null; action: LogAction; rule: string;
    rule_description: string; reason: string }[];
}

export interface CaseMix {
  provenance: Provenance;
  months: { month: string; records: number; emergency: number; opd: number; opd_share: number; anomaly: boolean }[];
  overall: { emergency_share: number; opd_share: number };
  other_development_months_opd_share_range: [number, number];
  anomaly_months: string[];
  note: string;
}

// ------------------------------------------------------------------ Stage 1: forecast run
export type HistoricalTier = "higher" | "typical" | "lower";

export interface RunDay {
  date: string; mean: number; lower_80: number; upper_80: number; weekday: string;
  holidays: { name: string; type: string }[];
  day_before_holiday: boolean; day_after_holiday: boolean;
  historical_comparison: { tier: HistoricalTier; label: string; p25: number; p75: number; n: number; includes_potentially_incomplete_day: boolean };
  window_rank: number; busiest_in_window: boolean; vs_window_average_pct: number;
}

export interface ForecastRun {
  mode: "demonstration";
  label: string;
  manifest: {
    as_of: string; forecast_start: string; forecast_end: string; data_through: string; data_through_note: string;
    calendar_source: string; calendar_covered_until: string; model: string; forecast_py_sha256: string; sha256_matches_frozen: boolean;
    fallback_used: boolean; model_inputs: string[]; not_model_inputs: string[]; period_note: string;
  };
  official: { date: string; mean: number; lower_80: number; upper_80: number }[];
  days: RunDay[];
  summary: {
    total_mean: number; average_mean: number; busiest: { date: string; mean: number }; quietest: { date: string; mean: number };
    recent_28_day_mean: number; historical_comparison_counts: Record<HistoricalTier, number>; busiest_days: string[];
  };
  historical_comparison_rule: { description: string; baseline_start: string; baseline_end: string; no_higher_days_note: string | null;
    all_lower_note: string | null; descriptive_only: string };
  window_ranking_rule: { description: string; descriptive_only: string };
  days_to_watch: { date: string; kind: "busiest" | "holiday"; title: string; detail: string }[];
  upcoming_holidays: { date: string; name: string; type: string; in_forecast_window: boolean }[];
  reliability: { source: string; mae: number; coverage80: number; bias: number; n_cutoffs: number; n_points: number;
    ratio_vs_baseline: number; ratio_ci: number[] };
}

// ------------------------------------------------------------------ Stage 2: live conditions (context only)
export type ProviderStatus = "ok" | "stale" | "unavailable";

interface ProviderBlock<Day> {
  status: ProviderStatus;
  error: string | null;
  cache_hit: boolean;
  retrieved_at: string | null;
  age_minutes: number | null;
  attribution: { text: string; licence: string; url: string; licence_url: string };
  available_dates: { first: string | null; last: string | null; count: number };
  units: Record<string, string>;
  labels: Record<string, string>;
  grid: { latitude?: number; longitude?: number; elevation?: number };
  provider_dates: { first: string; last: string } | null;
  issues: string[];
  days: Day[];
}

export interface WeatherDay { date: string; available: boolean; values: Partial<Record<
  "temperature_2m_max" | "temperature_2m_min" | "temperature_2m_mean" | "relative_humidity_2m_mean" | "precipitation_sum" |
  "precipitation_probability_max" | "wind_speed_10m_max" | "wind_gusts_10m_max", number>> }

export interface AirDay {
  date: string; available: boolean; status?: "full" | "partial" | "none"; hours_with_data?: number;
  values: Record<string, number>; us_aqi_category?: string;
}

export interface Conditions {
  kind: "live_environmental_forecast";
  label: string;
  separation_note: string;
  used_by_admissions_model: false;
  location: { name: string; latitude: number; longitude: number; timezone: string };
  today: string;
  window: { start: string; end: string; days: number };
  aqi_scale: { name: "US AQI"; note: string; categories: { upper: number | null; label: string }[] };
  cache_policy: { ttl_minutes: number; stale_max_hours: number; timeout_seconds: number };
  weather: ProviderBlock<WeatherDay> & { labels: Record<string, string> };
  air_quality: ProviderBlock<AirDay> & { aggregation?: { us_aqi: string; pollutants: string; min_hours: number } };
}

// ------------------------------------------------------------------ Stage A: multi-variable comparison (internal)
export interface MlRow {
  model: string; family: string; features: string; kind: "benchmark" | "baseline" | "candidate" | "oracle"; q: string;
  tune_MAE: number; tune_coverage80: number; val_MAE: number; val_RMSE: number; val_WAPE: number; val_bias: number;
  val_coverage80: number; val_width80: number; val_ratio_vs_M4: number; val_ratio_vs_B0: number; ci_vs_M4: [number, number, number] | null;
  sec_per_fit: number | null;
}
export interface MlComparison {
  provenance: "evaluation"; validation_period: string; tuning_period: string; selection_rule: string; rows: MlRow[];
  decision: { challenger: string; challenger_val_MAE: number; M4_val_MAE: number; ratio_vs_M4: number; ratio_ci_95: [number, number];
    challenger_val_coverage80: number; selected: string; rule_passed: boolean;
    environment_effect_val_MAE: Record<string, { H: number; HE: number; HX_oracle: number }> };
  jan_mar_2019_previously_viewed: unknown | null;
}

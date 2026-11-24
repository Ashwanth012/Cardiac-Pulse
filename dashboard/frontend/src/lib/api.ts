import { useQuery } from "@tanstack/react-query";
import type {
  CaseMix, Conditions, ForecastRun, MlComparison, HistoricalForecast, HoldoutEvaluation, Meta, ModelExplain, Overview, PollutionLog, QualityDuplicates, QualityMissing,
  QualitySummary, RetroForecast,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function getJson<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(0, "The analytics API is not reachable. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, String(detail));
  }
  return res.json() as Promise<T>;
}

const STATIC = { staleTime: Infinity, gcTime: Infinity, retry: 1 } as const;

export const useMeta = () => useQuery({ queryKey: ["meta"], queryFn: () => getJson<Meta>("/api/meta"), ...STATIC });

export const useOverview = (windowDays: number) =>
  useQuery({
    queryKey: ["overview", windowDays],
    queryFn: () => getJson<Overview>(`/api/overview?window_days=${windowDays}`),
    placeholderData: (prev) => prev,
    ...STATIC,
  });

export const useHistoricalForecast = () =>
  useQuery({
    queryKey: ["forecast", "historical"],
    queryFn: () => getJson<HistoricalForecast>("/api/forecast/historical?context_days=42"),
    ...STATIC,
  });

// ------------------------------------------------------------------ Phase 2
export const useRetroForecast = (cutoff: string | null) =>
  useQuery({
    queryKey: ["forecast", "retro", cutoff],
    queryFn: () => getJson<RetroForecast>(`/api/forecast/retrospective?cutoff=${encodeURIComponent(cutoff!)}&context_days=42`),
    enabled: !!cutoff,
    placeholderData: (prev) => prev,
    ...STATIC,
    retry: false,
  });

export const useHoldout = () =>
  useQuery({ queryKey: ["evaluation", "holdout"], queryFn: () => getJson<HoldoutEvaluation>("/api/evaluation/holdout"), ...STATIC });

export const useModelExplain = (cutoff = "2019-04-01") =>
  useQuery({ queryKey: ["model", "explain", cutoff], queryFn: () => getJson<ModelExplain>(`/api/model/explain?cutoff=${cutoff}`), ...STATIC });

// ------------------------------------------------------------------ Phase 3A: data quality
const quality = <T,>(part: string) => () => useQuery({ queryKey: ["quality", part], queryFn: () => getJson<T>(`/api/quality/${part}`), ...STATIC });
export const useQualitySummary = quality<QualitySummary>("summary");
export const useQualityDuplicates = quality<QualityDuplicates>("duplicates");
export const useQualityMissing = quality<QualityMissing>("missing");
export const usePollutionLog = quality<PollutionLog>("pollution-log");
export const useCaseMix = quality<CaseMix>("case-mix");

// ------------------------------------------------------------------ Stage 1: forecast run
export const useForecastRun = () =>
  useQuery({ queryKey: ["run", "2019-04-01"], queryFn: () => getJson<ForecastRun>("/api/run?as_of=2019-04-01"), ...STATIC });

// ------------------------------------------------------------------ Stage 2: live conditions (context only)
/** Backend caches the provider for 60 min; the browser re-asks every 15 min so a refreshed forecast shows up. */
export const useConditions = () =>
  useQuery({ queryKey: ["conditions"], queryFn: () => getJson<Conditions>("/api/conditions"), staleTime: 15 * 60_000,
    refetchInterval: 15 * 60_000, refetchOnWindowFocus: false, retry: 1 });

// ------------------------------------------------------------------ Stage A: multi-variable model comparison
export const useMlComparison = () =>
  useQuery({ queryKey: ["ml", "comparison"], queryFn: () => getJson<MlComparison>("/api/ml/comparison"), ...STATIC });

import { HistoricalForecastChart } from "../components/charts/HistoricalForecastChart";
import { Stat } from "../components/forecast/ForecastNotes";
import { ForecastStrip, IndicatorExplainer, NotModelInputs, OfficialTable, ReadingGuide, RunBanner } from "../components/forecast/RunParts";
import { Badge, ProvenanceBadge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { ErrorState, Loading } from "../components/ui/States";
import { useForecastRun, useHistoricalForecast } from "../lib/api";
import { fmt1, fmtLong } from "../lib/format";

export default function Forecast14() {
  const run = useForecastRun();
  const hist = useHistoricalForecast();
  if (run.isError) return <ErrorState title="The official forecast is unavailable" message={(run.error as Error).message} onRetry={() => run.refetch()} />;
  if (!run.data) return <Loading label="Running the official forecast.py" />;
  const r = run.data;
  const s = r.summary;
  return (
    <div>
      <RunBanner run={r} />

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Expected admissions, 14 days" value={Math.round(s.total_mean).toLocaleString("en-GB")} sub="sum of daily means (raw records)" />
        <Stat label="Average per day" value={fmt1(s.average_mean)} sub="expected records/day" />
        <Stat label="Busiest day in forecast" value={fmt1(s.busiest.mean)} sub={fmtLong(s.busiest.date)} />
        <Stat label="Quietest day in forecast" value={fmt1(s.quietest.mean)} sub={fmtLong(s.quietest.date)} />
        <Stat label="Recent observed level" value={fmt1(s.recent_28_day_mean)} tone="warn"
          sub="mean of the last 28 days before the cut-off (observed) – the forecast sits below it" />
      </div>

      <Panel className="mb-6" title="Daily expected admissions, 1–14 Apr 2019"
        subtitle="Observed records before the cut-off, then the official forecast with its 80% prediction interval. No actual admissions are shown for forecast dates."
        actions={<><ProvenanceBadge kind="historical_forecast" />{r.manifest.fallback_used && <Badge tone="amber">Fallback used</Badge>}</>}>
        {hist.data ? <HistoricalForecastChart data={hist.data} busiest={s.busiest_days} markHolidays height={340} />
          : hist.isError ? <ErrorState title="Chart unavailable" message={(hist.error as Error).message} onRetry={() => hist.refetch()} />
            : <Loading label="Loading chart" />}
      </Panel>

      <Panel className="mb-6" title="What this forecast is based on"
        subtitle="The forecasting model was compared with weather- and air-quality-aware alternatives; none forecast admissions more accurately, so the model below is used.">
        <NotModelInputs run={r} />
      </Panel>

      <Panel className="mb-6" title="Demand indicators" subtitle="Two separate, descriptive views of the same 14 forecast values. Neither is a capacity or staffing assessment.">
        <IndicatorExplainer run={r} />
      </Panel>

      <Panel className="mb-6" title="Forecast calendar" subtitle="Each day with its expected admissions, 80% range, holidays and both indicators"
        actions={<ProvenanceBadge kind="historical_forecast" />}>
        <ForecastStrip run={r} />
      </Panel>

      <Panel className="mb-6" title="Official forecast output" subtitle="The 14 rows returned by forecast(cutoff_date='2019-04-01'), with calendar context"
        actions={<ProvenanceBadge kind="historical_forecast" />}>
        <OfficialTable run={r} />
      </Panel>

      <Panel title="How to read this forecast">
        <ReadingGuide />
      </Panel>
    </div>
  );
}

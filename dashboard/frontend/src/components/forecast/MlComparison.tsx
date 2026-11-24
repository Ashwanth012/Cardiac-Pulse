import clsx from "clsx";
import { CircleCheck, FlaskConical } from "lucide-react";
import { useMlComparison } from "../../lib/api";
import { fmt1 } from "../../lib/format";
import type { MlRow } from "../../lib/types";
import { Badge, ProvenanceBadge } from "../ui/Badge";
import { Panel } from "../ui/Panel";
import { ErrorState, Loading } from "../ui/States";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const KIND: Record<MlRow["kind"], { tone: "forecast" | "neutral" | "teal" | "amber"; label: string }> = {
  benchmark: { tone: "forecast", label: "Benchmark" }, baseline: { tone: "neutral", label: "Baseline" },
  candidate: { tone: "teal", label: "Deployable" }, oracle: { tone: "amber", label: "Oracle – not deployable" },
};

/** Internal: Stage A multi-variable model comparison on the 2018 validation cut-offs (read from saved results). */
export function MlComparisonPanel() {
  const q = useMlComparison();
  if (q.isLoading) return <Panel title="Multi-variable model comparison"><Loading label="Loading model comparison" /></Panel>;
  if (q.isError || !q.data) return <ErrorState title="Model comparison unavailable" message={(q.error as Error)?.message ?? ""} onRetry={() => q.refetch()} />;
  const d = q.data;
  const dec = d.decision;
  const env = Object.entries(dec.environment_effect_val_MAE);
  return (
    <Panel className="mb-6" title="Multi-variable model comparison"
      subtitle={`Validation: ${d.validation_period}. Same cut-offs, truth and metric code as the M4 backtest.`}
      actions={<ProvenanceBadge kind="evaluation" />}>
      <div className={clsx("mb-4 flex gap-3 rounded-xl p-3.5 ring-1 ring-inset", dec.rule_passed ? "bg-teal/[0.07] ring-teal/30" : "bg-forecast/[0.07] ring-forecast/30")}>
        <CircleCheck aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-forecast-ink" />
        <div className="text-[12.5px] leading-5 text-text-2">
          <p className="font-semibold text-text">
            Selected: {dec.rule_passed ? dec.selected : "M4 (frozen forecast.py) – kept"}
          </p>
          <p>
            Best deployable challenger: <strong className="text-text">{dec.challenger}</strong>, validation MAE {fmt1(dec.challenger_val_MAE)} vs M4 {fmt1(dec.M4_val_MAE)}
            {" "}(ratio {dec.ratio_vs_M4.toFixed(3)}, 95% CI {dec.ratio_ci_95[0].toFixed(2)}–{dec.ratio_ci_95[1].toFixed(2)}).
            {dec.rule_passed ? " The pre-registered rule was met." : " The pre-registered rule (CI entirely below 1 and coverage ≥ 80%) was not met, so M4 stays."}
            {" "}Jan–Mar 2019 was not used to choose between candidates.
          </p>
          <p className="mt-1 text-text-3">{d.selection_rule}</p>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
        <table className="w-full min-w-[980px] text-[12.5px]">
          <caption className="sr-only">Validation metrics for every compared model</caption>
          <thead className="bg-surface-2"><tr className="text-left text-[11px] font-semibold uppercase tracking-wide text-text-3">
            {["Model", "Features", "Type", "MAE", "vs M4 (95% CI)", "RMSE", "WAPE", "Bias", "80% coverage", "Width"].map((h) =>
              <th key={h} scope="col" className="px-3 py-2">{h}</th>)}
          </tr></thead>
          <tbody>
            {d.rows.map((r) => (
              <tr key={r.model} className={clsx("border-t border-border/10", r.model === dec.selected && "bg-forecast/[0.06]")}>
                <td className="px-3 py-2 font-medium text-text">{r.family}<span className="block font-mono text-[10.5px] text-text-3">{r.model}</span></td>
                <td className="max-w-[260px] px-3 py-2 text-text-2">{r.features}</td>
                <td className="px-3 py-2"><Badge tone={KIND[r.kind].tone} icon={r.kind === "oracle" ? FlaskConical : undefined}>{KIND[r.kind].label}</Badge></td>
                <td className="tabular px-3 py-2 font-semibold text-text">{r.val_MAE.toFixed(2)}</td>
                <td className="tabular whitespace-nowrap px-3 py-2 text-text-2">
                  {r.ci_vs_M4 ? `${r.ci_vs_M4[0].toFixed(2)} (${r.ci_vs_M4[1].toFixed(2)}–${r.ci_vs_M4[2].toFixed(2)})` : "—"}</td>
                <td className="tabular px-3 py-2 text-text-2">{r.val_RMSE.toFixed(2)}</td>
                <td className="tabular px-3 py-2 text-text-2">{pct(r.val_WAPE)}</td>
                <td className="tabular px-3 py-2 text-text-2">{r.val_bias > 0 ? "+" : ""}{r.val_bias.toFixed(2)}</td>
                <td className="tabular px-3 py-2 text-text-2">{pct(r.val_coverage80)}</td>
                <td className="tabular px-3 py-2 text-text-2">{fmt1(r.val_width80)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="mt-5 text-[13px] font-semibold text-text">Does environmental data help? (validation MAE, lower is better)</h3>
      <div className="mt-2 overflow-x-auto rounded-xl ring-1 ring-inset ring-border/15">
        <table className="w-full min-w-[560px] text-[12.5px]">
          <caption className="sr-only">Validation MAE by environment feature set for each model family</caption>
          <thead className="bg-surface-2"><tr className="text-left text-[11px] font-semibold uppercase tracking-wide text-text-3">
            <th className="px-3 py-2">Model family</th><th className="px-3 py-2">History + calendar</th>
            <th className="px-3 py-2">+ environment known at cut-off</th><th className="px-3 py-2">+ actual future environment (oracle)</th>
          </tr></thead>
          <tbody>
            {env.map(([k, v]) => (
              <tr key={k} className="border-t border-border/10">
                <td className="px-3 py-2 font-medium text-text">{k}</td>
                <td className="tabular px-3 py-2 text-text-2">{v.H.toFixed(2)}</td>
                <td className="tabular px-3 py-2 text-text-2">{v.HE.toFixed(2)}</td>
                <td className="tabular px-3 py-2 text-text-2">{v.HX_oracle.toFixed(2)}</td>
              </tr>
            ))}
            <tr className="border-t border-border/10 bg-forecast/[0.05]"><td className="px-3 py-2 font-medium text-text">M4 (no environment)</td>
              <td colSpan={3} className="tabular px-3 py-2 text-text">{dec.M4_val_MAE.toFixed(2)}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[12px] leading-5 text-text-2">
        Environment known at the cut-off (what a forecast can actually use) did not improve any model. Even the actual future weather and air
        quality – information no forecast has – did not bring any model below M4. Archived weather/AQI forecasts for 2017–19 are not available,
        so a model driven by future environmental forecasts cannot be validated honestly with this data.
      </p>
    </Panel>
  );
}

import { Construction } from "lucide-react";
import { Badge } from "../components/ui/Badge";
import { hrefFor, type RouteDef } from "../lib/router";

export default function ComingSoon({ route }: { route: RouteDef }) {
  const Icon = route.icon;
  return (
    <section className="panel mx-auto mt-6 max-w-2xl p-8 text-center animate-fade-up" aria-labelledby="soon-title">
      <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-observed/25 to-forecast/20 text-observed-ink ring-1 ring-inset ring-observed/30">
        <Icon aria-hidden className="h-7 w-7" />
      </span>
      <div className="mt-4 flex justify-center"><Badge tone="neutral" icon={Construction}>Planned · Stage {route.stage}</Badge></div>
      <h2 id="soon-title" className="mt-3 text-xl font-semibold text-text">{route.label}</h2>
      <p className="mt-2 text-[14px] leading-6 text-text-2">{route.description}.</p>
      <p className="mt-1 text-[13px] text-text-3">Planned for redesign Stage {route.stage}; not built yet. Nothing on this page is live.</p>
      <a href={hrefFor("/")} className="mt-6 inline-flex rounded-xl bg-surface-2 px-4 py-2 text-[13px] font-semibold text-text ring-1 ring-inset ring-border/25 hover:bg-surface">
        Back to Command Centre
      </a>
    </section>
  );
}

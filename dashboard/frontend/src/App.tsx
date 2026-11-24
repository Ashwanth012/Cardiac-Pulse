import { Component, type ReactNode } from "react";
import { AppShell } from "./components/layout/AppShell";
import { ErrorState } from "./components/ui/States";
import { useRoute } from "./lib/router";
import CalendarContext from "./pages/CalendarContext";
import CommandCentre from "./pages/CommandCentre";
import Conditions from "./pages/Conditions";
import ComingSoon from "./pages/ComingSoon";
import DataQuality from "./pages/DataQuality";
import Forecast14 from "./pages/Forecast14";
import Insights from "./pages/Insights";
import ModelPerformance from "./pages/ModelPerformance";

class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error)
      return <ErrorState title="Something went wrong on this page" message={this.state.error.message}
        onRetry={() => this.setState({ error: null })} />;
    return this.props.children;
  }
}

const PAGES: Record<string, ReactNode> = {
  "/": <CommandCentre />,
  "/forecast": <Forecast14 />,
  "/conditions": <Conditions />,
  "/calendar": <CalendarContext />,
  "/insights": <Insights />,
  "/performance": <ModelPerformance />,
  "/quality": <DataQuality />,
};

export default function App() {
  const route = useRoute();
  return (
    <AppShell route={route}>
      <Boundary key={route.path}>{PAGES[route.path] ?? <ComingSoon route={route} />}</Boundary>
    </AppShell>
  );
}

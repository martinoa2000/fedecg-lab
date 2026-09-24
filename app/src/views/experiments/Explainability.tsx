import type { AppData } from "../../data";
import { ExplainResults } from "../PrivacyAndTrust";
import { NotRunYet, PageHead } from "./model";

const TITLE = "Where the model looks";
const LEDE =
  "Integrated Gradients and Grad-CAM on test ECGs the baseline detects correctly, with every beat split into QRS complex, ST segment and T wave. Each map is also computed for the same network with random weights: only what differs from that is the model's reasoning.";

export function Explainability({ data }: { data: AppData }) {
  const { explain } = data;
  if (!explain.segments.length) {
    return (
      <NotRunYet title={TITLE} lede={LEDE} commands={["uv run python scripts/explain_model.py"]} />
    );
  }
  return (
    <div className="page">
      <PageHead title={TITLE} status={`${explain.examples.length} classes explained`} lede={LEDE} />
      <section aria-label="Saliency by class">
        <ExplainResults explain={explain} />
      </section>
    </div>
  );
}

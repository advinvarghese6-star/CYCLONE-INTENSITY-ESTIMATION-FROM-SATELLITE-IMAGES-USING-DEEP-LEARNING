// =============================================================================
// TCNet Dashboard — Main Page
// Orchestrates all components: Header, ControlPanel, MetricsDisplay,
// VisualizerViewport, and Footer.
// =============================================================================

"use client";

import React, { useCallback, useState } from "react";
import { AlertCircle } from "lucide-react";
import Header from "@/components/header";
import ControlPanel from "@/components/control-panel";
import MetricsDisplay from "@/components/metrics-display";
import VisualizerViewport from "@/components/visualizer-viewport";
import { useBackendHealth } from "@/hooks/use-backend-health";
import { useTCNetInference } from "@/hooks/use-tcnet-inference";

/** Map preset IDs to their known ground-truth wind speeds (in knots). */
const PRESET_GROUND_TRUTH: Record<string, number> = {
  fani_2019: 135,
  amphan_2020: 140,
  tauktae_2021: 120,
  biparjoy_2023: 105,
  yaas_2021: 75,
  nisarga_2020: 55,
};

export default function HomePage() {
  const health = useBackendHealth();
  const { isLoading, result, error, runInference, runPresetInference, reset } =
    useTCNetInference();

  const [groundTruth, setGroundTruth] = useState<number | null>(null);

  const handleRunUpload = useCallback(
    (file: File, useRotationBlending: boolean) => {
      setGroundTruth(null);
      runInference(file, useRotationBlending);
    },
    [runInference]
  );

  const handleRunPreset = useCallback(
    (presetId: string, useRotationBlending: boolean) => {
      setGroundTruth(PRESET_GROUND_TRUTH[presetId] ?? null);
      runPresetInference(presetId, useRotationBlending);
    },
    [runPresetInference]
  );

  return (
    <>
      <Header health={health} />

      <main className="flex-1">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          {/* ── Offline Warning Banner ──────────────────────── */}
          {!health.isChecking && !health.isOnline && (
            <div className="mb-6 flex items-center gap-3 rounded-xl border border-red-500/20 bg-red-500/5 p-4 animate-in fade-in slide-in-from-top-2 duration-300">
              <AlertCircle className="h-5 w-5 shrink-0 text-red-400" />
              <div>
                <p className="text-sm font-medium text-red-300">
                  Backend Unreachable
                </p>
                <p className="text-xs text-zinc-400">
                  Cannot connect to the FastAPI server at{" "}
                  <code className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-zinc-300">
                    {process.env.NEXT_PUBLIC_API_URL ?? "localhost:8000"}
                  </code>
                  . Ensure the backend is running before submitting predictions.
                </p>
              </div>
            </div>
          )}

          {/* ── Error Banner ────────────────────────────────── */}
          {error && (
            <div className="mb-6 flex items-center gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 animate-in fade-in slide-in-from-top-2 duration-300">
              <AlertCircle className="h-5 w-5 shrink-0 text-amber-400" />
              <div>
                <p className="text-sm font-medium text-amber-300">
                  Estimation Failed
                </p>
                <p className="text-xs text-zinc-400">{error}</p>
              </div>
              <button
                onClick={reset}
                className="ml-auto shrink-0 rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 cursor-pointer"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* ── Main Grid ──────────────────────────────────── */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Left Column: Control Panel */}
            <div className="lg:col-span-4">
              <ControlPanel
                isLoading={isLoading}
                onRunUpload={handleRunUpload}
                onRunPreset={handleRunPreset}
              />
            </div>

            {/* Right Column: Results */}
            <div className="space-y-6 lg:col-span-8">
              {/* Metrics */}
              <MetricsDisplay
                result={result}
                groundTruthKnots={groundTruth}
              />

              {/* Visualizer */}
              <VisualizerViewport result={result} />
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

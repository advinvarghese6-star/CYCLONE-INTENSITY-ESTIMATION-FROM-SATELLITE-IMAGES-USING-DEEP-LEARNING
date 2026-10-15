// =============================================================================
// TCNet Dashboard — Visualizer Viewport Component
// Interactive Grad-CAM overlay with live opacity control, view modes
// (Overlay, Raw IR, Side-by-Side), and Phase 3 sensitivity explanation.
// =============================================================================

"use client";

import React, { useState } from "react";
import { Layers, ImageOff, Eye, Sliders, Columns } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { PredictionResponse } from "@/types/tcnet";

interface VisualizerViewportProps {
  result: PredictionResponse | null;
}

export default function VisualizerViewport({ result }: VisualizerViewportProps) {
  const [opacity, setOpacity] = useState<number>(85);
  const [viewMode, setViewMode] = useState<"overlay" | "raw" | "split">("overlay");

  const hasGradcam = Boolean(result?.gradcam_overlay_base64);
  const hasRaw = Boolean(result?.raw_image_base64);

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <div className="rounded-lg bg-orange-500/10 p-1.5">
              <Layers className="h-4 w-4 text-orange-400" />
            </div>
            Grad-CAM Sensitivity Analysis
          </CardTitle>

          {/* ── View Mode Selector ──────────────────────── */}
          {hasGradcam && (
            <div className="flex items-center rounded-lg border border-zinc-800 bg-zinc-900/80 p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setViewMode("overlay")}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 font-medium transition-colors ${
                  viewMode === "overlay"
                    ? "bg-orange-500/20 text-orange-300 border border-orange-500/40"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Layers className="h-3.5 w-3.5" />
                Overlay
              </button>
              <button
                type="button"
                onClick={() => setViewMode("raw")}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 font-medium transition-colors ${
                  viewMode === "raw"
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Eye className="h-3.5 w-3.5" />
                Raw IR
              </button>
              <button
                type="button"
                onClick={() => setViewMode("split")}
                className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 font-medium transition-colors ${
                  viewMode === "split"
                    ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Columns className="h-3.5 w-3.5" />
                Split View
              </button>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent>
        {!result || !hasGradcam ? (
          <div className="flex min-h-[320px] flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-800 bg-zinc-800/20 text-center">
            <div className="rounded-2xl bg-zinc-800/50 p-4">
              <ImageOff className="h-8 w-8 text-zinc-700" />
            </div>
            <p className="text-sm text-zinc-500">
              Grad-CAM heatmap will appear here after estimation
            </p>
            <p className="text-[11px] text-zinc-600">
              Visualizes neural gradient sensitivity across satellite cloud features
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* ── Viewport Display ────────────────────────── */}
            {viewMode === "split" ? (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {/* Raw IR Left */}
                <div className="relative flex flex-col items-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-center">
                  <span className="mb-2 block text-xs font-medium text-zinc-400">
                    Raw Satellite IR (Crop 64×64)
                  </span>
                  <div className="relative aspect-square w-full max-w-[420px] overflow-hidden rounded-lg">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.raw_image_base64 || result.gradcam_overlay_base64}`}
                      alt="Raw Satellite IR"
                      className="h-full w-full object-contain"
                    />
                  </div>
                </div>

                {/* Grad-CAM Right */}
                <div className="relative flex flex-col items-center overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-center">
                  <span className="mb-2 block text-xs font-medium text-orange-400">
                    Grad-CAM Sensitivity Overlay
                  </span>
                  <div className="relative aspect-square w-full max-w-[420px] overflow-hidden rounded-lg">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                      alt="Grad-CAM Overlay"
                      className="h-full w-full object-contain"
                    />
                  </div>
                </div>
              </div>
            ) : (
              /* Overlay or Raw Single Display — Full square container */
              <div
                className="group relative mx-auto flex aspect-square w-full max-w-[520px] items-center justify-center overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-950"
                style={{ boxShadow: "inset 0 0 60px rgba(249,115,22,0.06)" }}
              >
                {/* Underlayer: Raw IR image */}
                {hasRaw && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:image/png;base64,${result.raw_image_base64}`}
                    alt="Raw Satellite IR Crop"
                    className="absolute inset-0 h-full w-full object-contain"
                  />
                )}

                {/* Overlayer: Grad-CAM with interactive opacity */}
                {viewMode === "overlay" && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                    alt="Grad-CAM Sensitivity Heatmap"
                    style={{ opacity: opacity / 100 }}
                    className="relative z-10 h-full w-full object-contain transition-opacity duration-100"
                  />
                )}

                {/* Subtle corner decorations */}
                <div className="pointer-events-none absolute left-3 top-3 h-5 w-5 border-l-2 border-t-2 border-orange-500/40" />
                <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-orange-500/40" />
                <div className="pointer-events-none absolute right-3 top-3 h-5 w-5 border-r-2 border-t-2 border-orange-500/40" />
                <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-orange-500/40" />
              </div>
            )}

            {/* ── Interactive Sensitivity & Heatmap Opacity Slider ─────── */}
            <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/60 p-3 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 font-medium text-zinc-300">
                  <Sliders className="h-3.5 w-3.5 text-orange-400" />
                  Heatmap Opacity / Blend Level
                </span>
                <span className="font-mono text-xs font-semibold text-orange-400">
                  {opacity}%
                </span>
              </div>

              {/* Live interactive slider */}
              <div className="relative flex items-center">
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-zinc-800 accent-orange-500 transition-all hover:bg-zinc-700"
                />
              </div>

              {/* Color Gradient Scale Legend */}
              <div className="pt-1 flex items-center gap-3">
                <span className="text-[10px] uppercase tracking-wider text-zinc-500 shrink-0">
                  Color Scale
                </span>
                <div
                  className="h-2 flex-1 rounded-full"
                  style={{
                    background:
                      "linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)",
                  }}
                />
                <div className="flex gap-3 text-[10px] text-zinc-500 shrink-0">
                  <span>Low (Blue)</span>
                  <span>Moderate (Green)</span>
                  <span>High (Red)</span>
                </div>
              </div>
            </div>

            {/* ── Info Caption (Phase 3 Specification) ──────────────── */}
            <div className="rounded-lg border border-zinc-800/40 bg-zinc-900/30 p-2.5 text-center text-[11px] text-zinc-400">
              <p>
                <span className="font-semibold text-zinc-200">Phase 3 Regression Grad-CAM: </span>
                Warm regions (yellow/red) show where the CNN output is most sensitive to evidence for{" "}
                <span className="font-semibold text-orange-400">HIGHER predicted wind speed</span>.
                The 4×4 coarse feature map is bilinearly upsampled to 64×64 (sensitivity map, not an object detector).
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

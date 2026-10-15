// =============================================================================
// TCNet Dashboard — Preset Gallery Component
// Displays pre-loaded satellite samples for quick one-click inference.
// =============================================================================

"use client";

import React from "react";
import { Wind, Zap } from "lucide-react";
import { cn } from "@/lib/utils";
import { getIntensityColor } from "@/lib/imd-scale";
import type { PresetSample } from "@/types/tcnet";

/** Built-in preset samples — in production these come from the API. */
const PRESET_SAMPLES: PresetSample[] = [
  {
    id: "fani_2019",
    name: "Fani (2019)",
    storm_name: "Fani",
    actual_knots: 135,
    thumbnail_url: "",
  },
  {
    id: "amphan_2020",
    name: "Amphan (2020)",
    storm_name: "Amphan",
    actual_knots: 140,
    thumbnail_url: "",
  },
  {
    id: "tauktae_2021",
    name: "Tauktae (2021)",
    storm_name: "Tauktae",
    actual_knots: 120,
    thumbnail_url: "",
  },
  {
    id: "biparjoy_2023",
    name: "Biparjoy (2023)",
    storm_name: "Biparjoy",
    actual_knots: 105,
    thumbnail_url: "",
  },
  {
    id: "yaas_2021",
    name: "Yaas (2021)",
    storm_name: "Yaas",
    actual_knots: 75,
    thumbnail_url: "",
  },
  {
    id: "nisarga_2020",
    name: "Nisarga (2020)",
    storm_name: "Nisarga",
    actual_knots: 55,
    thumbnail_url: "",
  },
];

interface PresetGalleryProps {
  selectedPreset: string | null;
  onSelectPreset: (presetId: string) => void;
}

export default function PresetGallery({
  selectedPreset,
  onSelectPreset,
}: PresetGalleryProps) {
  return (
    <div className="space-y-3">
      <p className="text-xs text-zinc-500">
        Select a historic cyclone sample to run instant inference
      </p>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {PRESET_SAMPLES.map((sample) => {
          const isSelected = selectedPreset === sample.id;
          const color = getIntensityColor(sample.actual_knots);

          return (
            <button
              key={sample.id}
              onClick={() => onSelectPreset(sample.id)}
              className={cn(
                "group relative flex flex-col items-center gap-2 rounded-xl border p-3 transition-all duration-200 cursor-pointer",
                isSelected
                  ? "border-cyan-500/50 bg-cyan-500/10 shadow-lg shadow-cyan-500/10"
                  : "border-zinc-700/60 bg-zinc-800/40 hover:border-zinc-600 hover:bg-zinc-800/70"
              )}
            >
              {/* Cyclone visual placeholder */}
              <div
                className="relative flex h-14 w-14 items-center justify-center rounded-full"
                style={{
                  background: `radial-gradient(circle, ${color}30 0%, transparent 70%)`,
                }}
              >
                <div
                  className="absolute inset-1 rounded-full border-2 border-dashed opacity-40 animate-[spin_8s_linear_infinite]"
                  style={{ borderColor: color }}
                />
                <Wind
                  className="h-5 w-5 transition-transform group-hover:scale-110"
                  style={{ color }}
                />
              </div>

              {/* Label */}
              <div className="text-center">
                <p className="text-xs font-semibold text-zinc-200">
                  {sample.name}
                </p>
                <div className="mt-0.5 flex items-center justify-center gap-1">
                  <Zap className="h-3 w-3 text-zinc-500" />
                  <span className="text-[10px] font-mono text-zinc-500">
                    {sample.actual_knots} kts
                  </span>
                </div>
              </div>

              {/* Selection ring */}
              {isSelected && (
                <div className="absolute -inset-px rounded-xl ring-2 ring-cyan-500/50" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

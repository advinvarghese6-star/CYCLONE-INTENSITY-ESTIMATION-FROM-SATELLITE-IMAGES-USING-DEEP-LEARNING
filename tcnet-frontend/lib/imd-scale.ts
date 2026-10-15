// =============================================================================
// TCNet Dashboard — IMD Cyclone Classification Scale
// Maps predicted wind speeds (knots) to IMD categories with color coding.
// =============================================================================

import type { IMDCategory } from "@/types/tcnet";

/**
 * Full IMD Tropical Cyclone Classification Scale.
 * Each entry defines a wind-speed band with its label and badge color.
 * Sorted ascending — iterate in reverse for lookup.
 */
export const IMD_SCALE: IMDCategory[] = [
  { label: "Low Pressure Area", color: "#6B7280", minKnots: 0, maxKnots: 16.99 },
  { label: "Depression", color: "#3B82F6", minKnots: 17, maxKnots: 27.99 },
  { label: "Deep Depression", color: "#06B6D4", minKnots: 28, maxKnots: 33.99 },
  { label: "Cyclonic Storm", color: "#14B8A6", minKnots: 34, maxKnots: 47.99 },
  { label: "Severe Cyclonic Storm", color: "#EAB308", minKnots: 48, maxKnots: 63.99 },
  { label: "Very Severe Cyclonic Storm", color: "#F97316", minKnots: 64, maxKnots: 89.99 },
  { label: "Extremely Severe Cyclonic Storm", color: "#EF4444", minKnots: 90, maxKnots: 119.99 },
  { label: "Super Cyclonic Storm", color: "#A855F7", minKnots: 120, maxKnots: Infinity },
];

/**
 * Returns the IMD category object for a given wind speed in knots.
 * Uses half-open intervals matching Phase 3 IMD_THRESHOLDS: [17, 28, 34, 48, 64, 90, 120].
 */
export function classifyIntensity(knots: number): IMDCategory {
  if (knots >= 120.0) return IMD_SCALE[7]; // Super Cyclonic Storm
  if (knots >= 90.0) return IMD_SCALE[6];  // Extremely Severe Cyclonic Storm
  if (knots >= 64.0) return IMD_SCALE[5];  // Very Severe Cyclonic Storm
  if (knots >= 48.0) return IMD_SCALE[4];  // Severe Cyclonic Storm
  if (knots >= 34.0) return IMD_SCALE[3];  // Cyclonic Storm
  if (knots >= 28.0) return IMD_SCALE[2];  // Deep Depression
  if (knots >= 17.0) return IMD_SCALE[1];  // Depression
  return IMD_SCALE[0];                    // Low Pressure Area
}

/**
 * Returns the hex color associated with a given wind speed.
 */
export function getIntensityColor(knots: number): string {
  return classifyIntensity(knots).color;
}

/**
 * Returns the human-readable label for a given wind speed.
 */
export function getIntensityLabel(knots: number): string {
  return classifyIntensity(knots).label;
}

/**
 * Returns a Tailwind-safe text-color class for a given category color hex.
 * Falls back to inline style usage since Tailwind can't dynamically generate
 * arbitrary color classes at build time.
 */
export function getCategoryBadgeStyle(knots: number): React.CSSProperties {
  const color = getIntensityColor(knots);
  return {
    backgroundColor: `${color}20`,
    color: color,
    borderColor: `${color}40`,
  };
}

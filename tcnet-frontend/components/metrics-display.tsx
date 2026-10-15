// =============================================================================
// TCNet Dashboard — Metrics Display Component
// Shows intensity gauge, IMD category badge, and comprehensive ML metrics grid
// including confidence, MAE, RMSE, MAPE, inference time, and model stats.
// Always computes and displays metrics — uses backend ml_metrics when available,
// otherwise calculates them on the frontend.
// =============================================================================

"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  TrendingUp,
  BarChart3,
  Target,
  AlertCircle,
  Timer,
  Cpu,
  Gauge,
  Sigma,
  ArrowUpDown,
  Percent,
  Crosshair,
  Grid3X3,
  Layers,
  Zap,
  Wind,
  Thermometer,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  classifyIntensity,
  getIntensityColor,
  getCategoryBadgeStyle,
} from "@/lib/imd-scale";
import type { PredictionResponse, MLMetrics } from "@/types/tcnet";

interface MetricsDisplayProps {
  result: PredictionResponse | null;
  groundTruthKnots?: number | null;
}

/**
 * Animated counter that smoothly increments from 0 to a target value.
 */
function AnimatedCounter({ value, duration = 1200 }: { value: number; duration?: number }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (value === 0) {
      setDisplay(0);
      return;
    }

    const startTime = performance.now();

    function animate(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = value * eased;
      setDisplay(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    }

    requestAnimationFrame(animate);
  }, [value, duration]);

  return <>{display.toFixed(1)}</>;
}

/**
 * Compute frontend fallback ML metrics when the backend doesn't provide them.
 */
function computeFallbackMetrics(
  result: PredictionResponse,
  groundTruthKnots: number | null | undefined
): MLMetrics {
  const knots = result.intensity_knots;
  const spread = result.rotation_spread ?? 0;

  // Confidence: sigmoid-like from rotation spread
  let confidence: number;
  if (result.rotation_blended && spread > 0) {
    confidence = Math.round(100 * Math.exp(-spread / 5) * 10) / 10;
    confidence = Math.max(0, Math.min(100, confidence));
  } else {
    confidence = knots > 0 && knots <= 180 ? 85 : 50;
  }

  const variance = Math.round(spread * spread * 1000) / 1000;

  // Ground truth dependent
  let mae: number | null = null;
  let rmse: number | null = null;
  let mape: number | null = null;
  let errorDir: string | null = null;
  let absError: number | null = null;
  let gtKnots: number | null = null;

  if (groundTruthKnots != null && groundTruthKnots > 0) {
    const err = Math.abs(knots - groundTruthKnots);
    mae = Math.round(err * 100) / 100;
    rmse = Math.round(err * 100) / 100;
    mape = Math.round((err / groundTruthKnots) * 10000) / 100;
    if (err <= 2.5) {
      errorDir = "on_target";
    } else {
      errorDir = knots > groundTruthKnots ? "overestimate" : "underestimate";
    }
    absError = Math.round(err * 100) / 100;
    gtKnots = groundTruthKnots;
  }

  return {
    confidence_score: confidence,
    prediction_variance: variance,
    mae,
    rmse,
    mape,
    ground_truth_knots: gtKnots,
    error_direction: errorDir,
    absolute_error: absError,
    inference_time_ms: 0,
    model_total_params: 0,
    model_trainable_params: 0,
    input_resolution: "64x64",
    input_channels: 1,
    normalization_mean: 0.4350,
    normalization_std: 0.2869,
  };
}

export default function MetricsDisplay({
  result,
  groundTruthKnots,
}: MetricsDisplayProps) {
  if (!result) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className="rounded-lg bg-cyan-500/10 p-1.5">
              <BarChart3 className="h-4 w-4 text-cyan-400" />
            </div>
            Estimation Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex min-h-[300px] flex-col items-center justify-center gap-4 text-center">
            <div className="rounded-2xl bg-zinc-800/50 p-6">
              <Activity className="h-12 w-12 text-zinc-700" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-zinc-500">
                No estimation yet
              </p>
              <p className="text-xs text-zinc-600">
                Upload an image or select a preset to begin analysis
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (result.status === "error") {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <div className="rounded-lg bg-red-500/10 p-1.5">
              <AlertCircle className="h-4 w-4 text-red-400" />
            </div>
            Estimation Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex min-h-[300px] flex-col items-center justify-center gap-4 text-center">
            <p className="text-sm text-red-400">
              {result.error_message ?? "An unknown error occurred."}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const knots = result.intensity_knots;
  const category = classifyIntensity(knots);
  const color = getIntensityColor(knots);
  const badgeStyle = getCategoryBadgeStyle(knots);
  const gaugePercent = Math.min((knots / 180) * 100, 100);

  // Always compute metrics: use backend's ml_metrics if available, otherwise compute on frontend
  const metrics: MLMetrics = result.ml_metrics ?? computeFallbackMetrics(result, groundTruthKnots);

  const difference =
    groundTruthKnots != null ? knots - groundTruthKnots : null;

  // Convert knots to km/h and mph for reference
  const kmh = Math.round(knots * 1.852 * 10) / 10;
  const mph = Math.round(knots * 1.151 * 10) / 10;

  /** Format large numbers with standard thousands comma separation (e.g. 389,057) */
  function formatNumber(n: number): string {
    if (n === 0) return "—";
    return n.toLocaleString("en-US");
  }

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <div className="rounded-lg bg-cyan-500/10 p-1.5">
            <BarChart3 className="h-4 w-4 text-cyan-400" />
          </div>
          Estimation Results
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* ── Primary Intensity Display ────────────────────── */}
        <div className="flex flex-col items-center gap-4">
          {/* Large wind speed */}
          <div className="relative">
            {/* Glow ring */}
            <div
              className="absolute -inset-4 rounded-full opacity-20 blur-2xl"
              style={{ backgroundColor: color }}
            />
            <div className="relative flex flex-col items-center">
              <span
                className="text-6xl font-black tabular-nums tracking-tighter lg:text-7xl"
                style={{ color }}
              >
                <AnimatedCounter value={knots} />
              </span>
              <span className="mt-1 text-sm font-medium tracking-wide text-zinc-400">
                KNOTS
              </span>
            </div>
          </div>

          {/* Wind speed conversions */}
          <div className="flex items-center gap-4 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <Wind className="h-3 w-3" /> {kmh} km/h
            </span>
            <span className="text-zinc-700">•</span>
            <span className="flex items-center gap-1">
              <Thermometer className="h-3 w-3" /> {mph} mph
            </span>
          </div>

          {/* IMD Category Badge */}
          <Badge
            className="rounded-lg border px-4 py-1.5 text-sm font-semibold"
            style={badgeStyle}
          >
            {category.label}
          </Badge>

          {/* Intensity Gauge Bar */}
          <div className="w-full max-w-xs space-y-1.5">
            <Progress value={gaugePercent} className="h-2.5" />
            <div className="flex justify-between text-[10px] text-zinc-600">
              <span>0 kts</span>
              <span>180 kts</span>
            </div>
          </div>
        </div>

        {/* ── Prediction Details Grid ───────────────────────── */}
        <div>
          <SectionTitle title="Prediction Details" />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {/* Single-Pass Estimate — always show */}
            <MetricCard
              icon={<Activity className="h-4 w-4 text-emerald-400" />}
              label="Single-Pass Estimate"
              value={`${(result.single_pass_knots ?? knots).toFixed(1)} kts`}
              sublabel="Direct inference (no augmentation)"
            />

            {/* Blended Estimate */}
            {result.rotation_blended && (
              <MetricCard
                icon={<TrendingUp className="h-4 w-4 text-cyan-400" />}
                label="Blended Estimate"
                value={`${knots.toFixed(1)} kts`}
                sublabel="10-angle TTA mean"
              />
            )}

            {/* Rotation Spread */}
            <MetricCard
              icon={<Sigma className="h-4 w-4 text-amber-400" />}
              label="Rotation Spread (σ)"
              value={`± ${(result.rotation_spread ?? 0).toFixed(2)} kts`}
              sublabel={result.rotation_blended ? "Agreement across rotations (NOT a CI)" : "No blending applied"}
            />

            {/* Ground Truth Difference */}
            {difference != null && (
              <MetricCard
                icon={<Target className="h-4 w-4 text-purple-400" />}
                label="Ground Truth Δ"
                value={`${difference > 0 ? "+" : ""}${difference.toFixed(1)} kts`}
                sublabel={`Actual: ${groundTruthKnots!.toFixed(0)} kts`}
                valueColor={Math.abs(difference) <= 10 ? "#22c55e" : "#ef4444"}
              />
            )}
          </div>
        </div>

        {/* ── ML Evaluation Metrics — Always shown ──────────── */}
        <div>
          <SectionTitle title="ML Evaluation Metrics" />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {/* Confidence Score */}
            <MetricCardWithBar
              icon={<Gauge className="h-4 w-4 text-emerald-400" />}
              label="Confidence Score"
              value={`${metrics.confidence_score.toFixed(1)}%`}
              sublabel="Model prediction certainty"
              barPercent={metrics.confidence_score}
              barColor={
                metrics.confidence_score >= 80
                  ? "#22c55e"
                  : metrics.confidence_score >= 60
                  ? "#eab308"
                  : "#ef4444"
              }
            />

            {/* Prediction Variance */}
            <MetricCard
              icon={<ArrowUpDown className="h-4 w-4 text-violet-400" />}
              label="Prediction Variance (σ²)"
              value={metrics.prediction_variance.toFixed(3)}
              sublabel="Spread² from rotation blending"
            />

            {/* Inference Time */}
            {metrics.inference_time_ms > 0 && (
              <MetricCard
                icon={<Timer className="h-4 w-4 text-sky-400" />}
                label="Inference Time"
                value={`${metrics.inference_time_ms.toFixed(1)} ms`}
                sublabel="Wall-clock processing time"
              />
            )}

            {/* MAE */}
            {metrics.mae != null && (
              <MetricCard
                icon={<Crosshair className="h-4 w-4 text-rose-400" />}
                label="MAE (Mean Abs Error)"
                value={`${metrics.mae.toFixed(2)} kts`}
                sublabel={`|predicted − actual| = |${knots.toFixed(1)} − ${metrics.ground_truth_knots?.toFixed(0)}|`}
                valueColor={metrics.mae <= 10 ? "#22c55e" : metrics.mae <= 20 ? "#eab308" : "#ef4444"}
              />
            )}

            {/* RMSE */}
            {metrics.rmse != null && (
              <MetricCard
                icon={<BarChart3 className="h-4 w-4 text-orange-400" />}
                label="RMSE"
                value={`${metrics.rmse.toFixed(2)} kts`}
                sublabel="Root Mean Squared Error"
                valueColor={metrics.rmse <= 10 ? "#22c55e" : metrics.rmse <= 20 ? "#eab308" : "#ef4444"}
              />
            )}

            {/* MAPE */}
            {metrics.mape != null && (
              <MetricCard
                icon={<Percent className="h-4 w-4 text-pink-400" />}
                label="MAPE"
                value={`${metrics.mape.toFixed(2)}%`}
                sublabel={`(${metrics.mae?.toFixed(2)} / ${metrics.ground_truth_knots?.toFixed(0)}) × 100`}
                valueColor={metrics.mape <= 10 ? "#22c55e" : metrics.mape <= 20 ? "#eab308" : "#ef4444"}
              />
            )}

            {/* Error Direction / Operational Bias */}
            {metrics.error_direction && (
              <MetricCard
                icon={
                  metrics.error_direction === "on_target" ? (
                    <Target className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <TrendingUp className="h-4 w-4 text-amber-400" />
                  )
                }
                label="Operational Accuracy"
                value={
                  metrics.error_direction === "on_target"
                    ? "🎯 On Target"
                    : metrics.error_direction === "overestimate"
                    ? "↑ Overestimate"
                    : "↓ Underestimate"
                }
                sublabel={
                  metrics.error_direction === "on_target"
                    ? `Within ±2.5 kts (Δ ${difference != null && difference > 0 ? "+" : ""}${difference?.toFixed(1)} kts)`
                    : `${difference != null && difference > 0 ? "+" : ""}${difference?.toFixed(1)} kts vs ${metrics.ground_truth_knots?.toFixed(0)} kts`
                }
                valueColor={
                  metrics.error_direction === "on_target"
                    ? "#22c55e"
                    : metrics.error_direction === "overestimate"
                    ? "#f97316"
                    : "#3b82f6"
                }
              />
            )}

            {/* Contextual Phase 3 scientific note for strong storms */}
            {metrics.ground_truth_knots != null && metrics.ground_truth_knots >= 90 && (
              <div className="col-span-1 rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3 text-xs text-zinc-400 sm:col-span-2 lg:col-span-3">
                <span className="font-semibold text-cyan-300">Phase 3 Observation: </span>
                Strong cyclones ({">"}90 kts) in TCIR exhibit documented regression-to-the-mean underestimation (typically 20–40 kts) due to severe class imbalance in extreme events.
              </div>
            )}
          </div>
        </div>

        {/* ── Calculations Breakdown ───────────────────────── */}
        <div>
          <SectionTitle title="Calculations Breakdown" />
          <div className="space-y-2 rounded-xl border border-zinc-800/60 bg-zinc-800/20 p-4">
            <CalcRow
              label="Wind Speed (kts)"
              formula="Model output"
              value={`${knots.toFixed(2)} kts`}
            />
            <CalcRow
              label="Wind Speed (km/h)"
              formula={`${knots.toFixed(2)} × 1.852`}
              value={`${kmh.toFixed(1)} km/h`}
            />
            <CalcRow
              label="Wind Speed (mph)"
              formula={`${knots.toFixed(2)} × 1.151`}
              value={`${mph.toFixed(1)} mph`}
            />
            <CalcRow
              label="Confidence Score"
              formula={result.rotation_blended
                ? `100 × e^(-σ/5) = 100 × e^(-${(result.rotation_spread ?? 0).toFixed(2)}/5)`
                : "Fixed: 85% (single-pass)"
              }
              value={`${metrics.confidence_score.toFixed(1)}%`}
            />
            <CalcRow
              label="Prediction Variance"
              formula={`σ² = ${(result.rotation_spread ?? 0).toFixed(2)}²`}
              value={metrics.prediction_variance.toFixed(3)}
            />
            {metrics.mae != null && groundTruthKnots != null && (
              <>
                <div className="my-2 border-t border-zinc-800/40" />
                <CalcRow
                  label="MAE"
                  formula={`|${knots.toFixed(2)} − ${groundTruthKnots.toFixed(0)}|`}
                  value={`${metrics.mae.toFixed(2)} kts`}
                />
                <CalcRow
                  label="RMSE"
                  formula={`√((${knots.toFixed(2)} − ${groundTruthKnots.toFixed(0)})²)`}
                  value={`${metrics.rmse?.toFixed(2)} kts`}
                />
                <CalcRow
                  label="MAPE"
                  formula={`(${metrics.mae.toFixed(2)} / ${groundTruthKnots.toFixed(0)}) × 100`}
                  value={`${metrics.mape?.toFixed(2)}%`}
                />
              </>
            )}
          </div>
        </div>

        {/* ── Model Architecture Stats ──────────────────────── */}
        <div>
          <SectionTitle title="Model & Input Configuration" />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {/* Total Parameters */}
            <MetricCard
              icon={<Cpu className="h-4 w-4 text-indigo-400" />}
              label="Total Parameters"
              value={formatNumber(metrics.model_total_params)}
              sublabel="Full model weight count"
            />

            {/* Trainable Parameters */}
            <MetricCard
              icon={<Zap className="h-4 w-4 text-yellow-400" />}
              label="Trainable Parameters"
              value={formatNumber(metrics.model_trainable_params)}
              sublabel="Learnable weights"
            />

            {/* Input Resolution */}
            <MetricCard
              icon={<Grid3X3 className="h-4 w-4 text-teal-400" />}
              label="Input Resolution"
              value={metrics.input_resolution}
              sublabel={`${metrics.input_channels}-channel IR`}
            />

            {/* Normalization Mean */}
            <MetricCard
              icon={<Layers className="h-4 w-4 text-zinc-400" />}
              label="Normalization μ"
              value={metrics.normalization_mean.toFixed(4)}
              sublabel="Dataset pixel mean"
            />

            {/* Normalization Std */}
            <MetricCard
              icon={<Layers className="h-4 w-4 text-zinc-400" />}
              label="Normalization σ"
              value={metrics.normalization_std.toFixed(4)}
              sublabel="Dataset std deviation"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/* ── Section Title Sub-component ───────────────────── */

function SectionTitle({ title }: { title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <div className="h-px flex-1 bg-zinc-800/60" />
      <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
        {title}
      </span>
      <div className="h-px flex-1 bg-zinc-800/60" />
    </div>
  );
}

/* ── Calculation Row Sub-component ─────────────────── */

interface CalcRowProps {
  label: string;
  formula: string;
  value: string;
}

function CalcRow({ label, formula, value }: CalcRowProps) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-36 shrink-0 font-medium text-zinc-400">{label}</span>
      <span className="flex-1 font-mono text-[11px] text-zinc-600">{formula}</span>
      <span className="shrink-0 font-semibold tabular-nums text-zinc-200">{value}</span>
    </div>
  );
}

/* ── Metric Card Sub-component ─────────────────────── */

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  sublabel: string;
  valueColor?: string;
}

function MetricCard({ icon, label, value, sublabel, valueColor }: MetricCardProps) {
  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3.5 transition-colors hover:border-zinc-700/60 hover:bg-zinc-800/40">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-zinc-400">{label}</span>
      </div>
      <p
        className="mt-1.5 text-lg font-bold tabular-nums text-zinc-100"
        style={valueColor ? { color: valueColor } : undefined}
      >
        {value}
      </p>
      <p className="text-[10px] text-zinc-600">{sublabel}</p>
    </div>
  );
}

/* ── Metric Card with Progress Bar ─────────────────── */

interface MetricCardWithBarProps extends MetricCardProps {
  barPercent: number;
  barColor: string;
}

function MetricCardWithBar({
  icon,
  label,
  value,
  sublabel,
  valueColor,
  barPercent,
  barColor,
}: MetricCardWithBarProps) {
  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-800/30 p-3.5 transition-colors hover:border-zinc-700/60 hover:bg-zinc-800/40">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-zinc-400">{label}</span>
      </div>
      <p
        className="mt-1.5 text-lg font-bold tabular-nums text-zinc-100"
        style={valueColor ? { color: valueColor } : undefined}
      >
        {value}
      </p>
      {/* Mini progress bar */}
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.min(barPercent, 100)}%`, backgroundColor: barColor }}
        />
      </div>
      <p className="mt-1 text-[10px] text-zinc-600">{sublabel}</p>
    </div>
  );
}

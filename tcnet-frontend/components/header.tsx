// =============================================================================
// TCNet Dashboard — Header Component
// Displays branding, model info pill, and live backend status indicator.
// =============================================================================

"use client";

import React from "react";
import { Activity, Cpu, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { BackendHealthState } from "@/hooks/use-backend-health";

interface HeaderProps {
  health: BackendHealthState;
}

export default function Header({ health }: HeaderProps) {
  const { isOnline, isChecking, refresh } = health;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* ── Branding ─────────────────────────────────────── */}
        <div className="flex items-center gap-3">
          {/* Animated cyclone icon */}
          <div className="relative flex h-9 w-9 items-center justify-center">
            <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 opacity-20 blur-md animate-pulse" />
            <Activity className="relative h-5 w-5 text-cyan-400" />
          </div>
          <div className="flex flex-col">
            <h1 className="text-base font-bold tracking-tight text-zinc-100 sm:text-lg">
              TCNet
              <span className="ml-1.5 font-normal text-zinc-400">
                — Cyclone Intensity Estimator
              </span>
            </h1>
          </div>
        </div>

        {/* ── Right-side pills ─────────────────────────────── */}
        <div className="flex items-center gap-3">
          {/* Model Info Pill */}
          <Badge
            variant="secondary"
            className="hidden items-center gap-1.5 sm:inline-flex"
          >
            <Cpu className="h-3 w-3" />
            PyTorch CNN (389k params)
          </Badge>

          {/* Backend Status Indicator */}
          <button
            onClick={refresh}
            className="group flex items-center gap-2 rounded-full border border-zinc-700/60 bg-zinc-900/60 px-3 py-1.5 text-xs font-medium transition-all duration-200 hover:border-zinc-600 hover:bg-zinc-800/80 cursor-pointer"
            title="Click to refresh status"
          >
            {isChecking ? (
              <RefreshCw className="h-3 w-3 animate-spin text-zinc-400" />
            ) : (
              <span className="relative flex h-2 w-2">
                <span
                  className={cn(
                    "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
                    isOnline ? "bg-emerald-400" : "bg-red-400"
                  )}
                />
                <span
                  className={cn(
                    "relative inline-flex h-2 w-2 rounded-full",
                    isOnline ? "bg-emerald-500" : "bg-red-500"
                  )}
                />
              </span>
            )}
            <span className={cn(isOnline ? "text-emerald-400" : "text-red-400")}>
              {isChecking ? "Checking…" : isOnline ? "Online" : "Offline"}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
}

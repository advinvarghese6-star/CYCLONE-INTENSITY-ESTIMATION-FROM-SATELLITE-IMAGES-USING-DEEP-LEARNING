// =============================================================================
// TCNet Dashboard — Backend Health Hook
// Auto-pings GET /health every 10 seconds and exposes connection status.
// =============================================================================

"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { checkHealth } from "@/lib/api-client";
import type { HealthCheckResponse } from "@/types/tcnet";

export interface BackendHealthState {
  /** Whether the backend is reachable and model is loaded. */
  isOnline: boolean;
  /** Whether the initial health check has completed. */
  isChecking: boolean;
  /** Last successful health response (null if never connected). */
  lastResponse: HealthCheckResponse | null;
  /** Manually trigger a health check. */
  refresh: () => void;
}

/**
 * Custom hook that continuously monitors the FastAPI backend health.
 * Pings `/health` every `intervalMs` milliseconds (default 10s).
 */
export function useBackendHealth(intervalMs: number = 10_000): BackendHealthState {
  const [isOnline, setIsOnline] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [lastResponse, setLastResponse] = useState<HealthCheckResponse | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const ping = useCallback(async () => {
    try {
      const response = await checkHealth();
      if (response && response.status === "ok") {
        setIsOnline(true);
        setLastResponse(response);
      } else {
        setIsOnline(false);
      }
    } catch {
      setIsOnline(false);
    } finally {
      setIsChecking(false);
    }
  }, []);

  useEffect(() => {
    // Initial ping
    ping();

    // Set up interval
    intervalRef.current = setInterval(ping, intervalMs);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [ping, intervalMs]);

  const refresh = useCallback(() => {
    setIsChecking(true);
    ping();
  }, [ping]);

  return { isOnline, isChecking, lastResponse, refresh };
}

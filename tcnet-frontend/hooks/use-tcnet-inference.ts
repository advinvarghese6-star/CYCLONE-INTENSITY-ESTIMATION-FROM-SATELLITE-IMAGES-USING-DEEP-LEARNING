// =============================================================================
// TCNet Dashboard — Inference Hook
// Manages POST /predict upload flow including loading, result, and error state.
// =============================================================================

"use client";

import { useState, useCallback } from "react";
import { predictIntensity, predictPreset } from "@/lib/api-client";
import type { PredictionResponse, InferenceState } from "@/types/tcnet";

export interface UseTCNetInferenceReturn extends InferenceState {
  /** Run inference on an uploaded file. */
  runInference: (file: File, useRotationBlending: boolean) => Promise<void>;
  /** Run inference on a preset sample by ID. */
  runPresetInference: (presetId: string, useRotationBlending: boolean) => Promise<void>;
  /** Clear the current result and error state. */
  reset: () => void;
}

/**
 * Custom hook encapsulating the full prediction lifecycle:
 * upload → loading → result | error.
 */
export function useTCNetInference(): UseTCNetInferenceReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runInference = useCallback(async (file: File, useRotationBlending: boolean) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictIntensity(file, useRotationBlending);

      if (response.status === "error") {
        setError(response.error_message ?? "Prediction failed. Please try again.");
      } else {
        setResult(response);
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : "Network error — is the backend running?";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const runPresetInference = useCallback(
    async (presetId: string, useRotationBlending: boolean) => {
      setIsLoading(true);
      setError(null);
      setResult(null);

      try {
        const response = await predictPreset(presetId, useRotationBlending);

        if (response.status === "error") {
          setError(response.error_message ?? "Prediction failed. Please try again.");
        } else {
          setResult(response);
        }
      } catch (err: unknown) {
        const message =
          err instanceof Error
            ? err.message
            : "Network error — is the backend running?";
        setError(message);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return { isLoading, result, error, runInference, runPresetInference, reset };
}

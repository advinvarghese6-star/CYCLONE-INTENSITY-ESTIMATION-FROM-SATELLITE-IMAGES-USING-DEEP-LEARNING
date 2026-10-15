// =============================================================================
// TCNet Dashboard — API Client
// Axios-based HTTP client for communicating with the FastAPI backend.
// =============================================================================

import axios, { type AxiosInstance } from "axios";
import type { HealthCheckResponse, PredictionResponse } from "@/types/tcnet";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Pre-configured Axios instance with base URL and sensible defaults.
 */
const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: {
    Accept: "application/json",
  },
});

/**
 * Pings the FastAPI `/health` endpoint.
 * Returns the parsed response on success or `null` on any failure.
 */
export async function checkHealth(): Promise<HealthCheckResponse | null> {
  try {
    const { data } = await client.get<HealthCheckResponse>("/health", {
      timeout: 3000,
    });
    return data;
  } catch {
    return null;
  }
}

/**
 * Posts a satellite image (file upload) to the FastAPI `/predict` endpoint.
 *
 * @param file               - The image or .npy file to classify.
 * @param useRotationBlending - Whether to enable 10-angle TTA blending.
 * @returns Parsed PredictionResponse from the backend.
 */
export async function predictIntensity(
  file: File,
  useRotationBlending: boolean = false
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_rotation_blending", String(useRotationBlending));

  const { data } = await client.post<PredictionResponse>(
    "/predict",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000, // model inference can take a while
    }
  );

  return data;
}

/**
 * Posts a preset sample ID to the FastAPI `/predict` endpoint.
 *
 * @param presetId            - The ID of the preset sample to run.
 * @param useRotationBlending - Whether to enable 10-angle TTA blending.
 * @returns Parsed PredictionResponse from the backend.
 */
export async function predictPreset(
  presetId: string,
  useRotationBlending: boolean = false
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("preset_id", presetId);
  formData.append("use_rotation_blending", String(useRotationBlending));

  const { data } = await client.post<PredictionResponse>(
    "/predict",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 60000,
    }
  );

  return data;
}

export default client;

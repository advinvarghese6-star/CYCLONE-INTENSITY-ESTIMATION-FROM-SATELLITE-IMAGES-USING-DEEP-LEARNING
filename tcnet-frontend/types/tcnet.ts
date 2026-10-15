// =============================================================================
// TCNet Dashboard — TypeScript Definitions
// All interfaces for API payloads, domain models, and application state.
// =============================================================================

/** Payload sent to the FastAPI `/predict` endpoint. */
export interface PredictionRequest {
  file: File | null;
  preset_id?: string;
  use_rotation_blending: boolean;
}

/** ML evaluation metrics returned by the backend. */
export interface MLMetrics {
  confidence_score: number;
  prediction_variance: number;
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  ground_truth_knots: number | null;
  signed_error?: number | null;
  error_direction: string | null;
  absolute_error: number | null;
  bias_context?: string | null;
  inference_time_ms: number;
  model_total_params: number;
  model_trainable_params: number;
  input_resolution: string;
  input_channels: number;
  normalization_mean: number;
  normalization_std: number;
}

/** Response returned from the FastAPI `/predict` endpoint. */
export interface PredictionResponse {
  status: "success" | "error";
  intensity_knots: number;
  single_pass_knots?: number;
  rotation_spread?: number;
  rotation_blended: boolean;
  imd_category: string;
  raw_image_base64?: string;
  gradcam_overlay_base64?: string;
  ml_metrics?: MLMetrics;
  error_message?: string;
}

/** A pre-loaded satellite sample available in the Preset Gallery. */
export interface PresetSample {
  id: string;
  name: string;
  storm_name: string;
  actual_knots: number;
  thumbnail_url: string;
}

/** FastAPI backend health check response. */
export interface HealthCheckResponse {
  status: "ok" | "error";
  model_loaded: boolean;
  device: string;
}

/** IMD cyclone category descriptor. */
export interface IMDCategory {
  label: string;
  color: string;
  minKnots: number;
  maxKnots: number;
}

/** State shape for the inference hook. */
export interface InferenceState {
  isLoading: boolean;
  result: PredictionResponse | null;
  error: string | null;
}

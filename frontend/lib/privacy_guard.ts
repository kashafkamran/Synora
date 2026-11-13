/**
 * lib/privacy_guard.ts
 * Privacy Enforcement Layer - client side (SRS privacy requirement,
 * SDS 3.1.1.7).  Every outbound training payload passes through
 * assertWeightsOnly() BEFORE it is sent: only model-weight tensors,
 * the round number, and numeric metrics may leave the browser.
 * Raw text, sentences or labels can never be transmitted.
 */
export class PrivacyError extends Error {}

const ALLOWED_TOP = new Set(["round", "weights", "metrics", "train_seconds", "local_epochs", "backend_used"]);
const ALLOWED_METRICS = new Set(["accuracy", "loss"]);
const ALLOWED_TENSOR = new Set(["layer_name", "shape", "dtype", "data_b64"]);

export function assertWeightsOnly(payload: Record<string, unknown>): void {
  for (const k of Object.keys(payload)) {
    if (!ALLOWED_TOP.has(k)) throw new PrivacyError(`Blocked: field '${k}' is not allowed in an update`);
  }
  const m = (payload.metrics ?? {}) as Record<string, unknown>;
  for (const [k, v] of Object.entries(m)) {
    if (!ALLOWED_METRICS.has(k) || typeof v !== "number" || !Number.isFinite(v)) {
      throw new PrivacyError(`Blocked: metric '${k}' must be a finite number named accuracy/loss`);
    }
  }
  const w = payload.weights;
  if (!Array.isArray(w) || w.length === 0) throw new PrivacyError("Blocked: weights missing");
  for (const t of w as Record<string, unknown>[]) {
    for (const k of Object.keys(t)) {
      if (!ALLOWED_TENSOR.has(k)) throw new PrivacyError(`Blocked: tensor field '${k}' not allowed`);
    }
    if (typeof t.data_b64 !== "string" || !/^[A-Za-z0-9+/=]+$/.test(t.data_b64)) {
      throw new PrivacyError("Blocked: tensor data must be base64 float32");
    }
  }
}

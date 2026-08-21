/**
 * Thin HTTP client for the existing Python FastAPI inference service.
 * This file intentionally contains ZERO ML logic -- it only forwards
 * requests and shapes errors. All preprocessing/model/decoder/
 * personalization logic stays in Python, per project requirements.
 */
import axios from 'axios';
import { getPythonBaseUrl, PYTHON_TIMEOUT_MS } from '../config.js';

export class PythonServiceError extends Error {
  constructor(message, status, details) {
    super(message);
    this.name = 'PythonServiceError';
    this.status = status || 502;
    this.details = details;
  }
}

function client() {
  // Built per-call (not cached at module load) so PYTHON_BASE_URL can be
  // changed at runtime -- required for test isolation and for anyone
  // pointing this backend at a different inference host without a restart.
  return axios.create({ baseURL: getPythonBaseUrl(), timeout: PYTHON_TIMEOUT_MS });
}

async function call(fn) {
  try {
    return await fn();
  } catch (err) {
    if (err.response) {
      throw new PythonServiceError(
        `Python service error: ${err.response.status} ${JSON.stringify(err.response.data)}`,
        502,
        err.response.data
      );
    }
    throw new PythonServiceError(`Python service unreachable: ${err.message}`, 503);
  }
}

export function pythonHealth() {
  return call(async () => (await client().get('/health')).data);
}

export function pythonModelInfo() {
  return call(async () => (await client().get('/model/info')).data);
}

export function pythonPredict(sensor, topK = 5) {
  return call(async () => (await client().post('/predict', { sensor, top_k: topK })).data);
}

export function pythonStartSession(userId) {
  return call(async () => {
    const res = await client().post('/session/start', { user_id: userId ?? null });
    return res.data.session_id;
  });
}

export function pythonStroke(pythonSessionId, sensor, topK = 5) {
  return call(async () =>
    (await client().post(`/session/${pythonSessionId}/stroke`, { sensor, top_k: topK })).data
  );
}

export function pythonCommit(pythonSessionId) {
  return call(async () => (await client().post(`/session/${pythonSessionId}/commit`)).data);
}

export function pythonEndSession(pythonSessionId) {
  return call(async () => (await client().post(`/session/${pythonSessionId}/end`)).data);
}
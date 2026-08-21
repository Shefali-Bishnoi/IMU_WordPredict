/**
 * Central Node-backend configuration. The Python base URL is read
 * dynamically (via a getter, not a module-level constant) so tests can
 * point it at a stub server started at an arbitrary port without any
 * import-order gymnastics.
 */
export const PORT = Number(process.env.PORT || 4000);

export function getPythonBaseUrl() {
  return process.env.PYTHON_BASE_URL || 'http://localhost:8000';
}

export const PYTHON_TIMEOUT_MS = Number(process.env.PYTHON_TIMEOUT_MS || 15000);
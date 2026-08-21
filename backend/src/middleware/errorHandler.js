import logger from '../logger.js';
import { PythonServiceError } from '../services/pythonClient.js';

export default function errorHandler(err, req, res, next) { // eslint-disable-line no-unused-vars
  const status = err instanceof PythonServiceError ? err.status : err.status || 500;
  logger.error(`[ERROR] ${req.method} ${req.originalUrl} -> ${err.message}`);
  res.status(status).json({
    error: err.message || 'Internal server error',
    ...(err.details ? { details: err.details } : {}),
  });
}
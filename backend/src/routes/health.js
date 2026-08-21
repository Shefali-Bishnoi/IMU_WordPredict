import { Router } from 'express';
import { pythonHealth } from '../services/pythonClient.js';
import logger from '../logger.js';

const router = Router();

/**
 * @route GET /health
 * Combined Node + Python health. See swagger/openapiSpec.js for the
 * documented schema.
 */
router.get('/health', async (req, res) => {
  let python;
  try {
    const data = await pythonHealth();
    python = { reachable: true, ...data };
  } catch (err) {
    logger.warn(`[HEALTH] python service unreachable: ${err.message}`);
    python = { reachable: false, error: err.message };
  }
  res.json({
    status: 'ok',
    service: 'wordpredict-node-backend',
    timestamp: new Date().toISOString(),
    python,
  });
});

export default router;
import { Router } from 'express';
import { pythonModelInfo } from '../services/pythonClient.js';

const router = Router();

/** GET /api/model/info -- pass-through of the python service's /model/info. */
router.get('/model/info', async (req, res, next) => {
  try {
    const info = await pythonModelInfo();
    res.json(info);
  } catch (err) {
    next(err);
  }
});

export default router;
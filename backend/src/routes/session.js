import { Router } from 'express';
import { createSession, getSession, resetSession, stopSession } from '../session/sessionManager.js';

const router = Router();

router.post('/session/start', async (req, res, next) => {
  try {
    const session = await createSession();
    res.status(201).json(session.toStatus());
  } catch (err) {
    next(err);
  }
});

router.get('/session/status', (req, res) => {
  const { sessionId } = req.query;
  if (!sessionId) return res.status(400).json({ error: 'sessionId query param is required' });
  const session = getSession(sessionId);
  if (!session) return res.status(404).json({ error: `Unknown sessionId: ${sessionId}` });
  res.json(session.toStatus());
});

router.post('/session/stop', async (req, res, next) => {
  try {
    const { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });
    const session = await stopSession(sessionId);
    if (!session) return res.status(404).json({ error: `Unknown sessionId: ${sessionId}` });
    res.json(session.toStatus());
  } catch (err) {
    next(err);
  }
});

router.post('/session/reset', async (req, res, next) => {
  try {
    const { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });
    const session = await resetSession(sessionId);
    if (!session) return res.status(404).json({ error: `Unknown sessionId: ${sessionId}` });
    res.json(session.toStatus());
  } catch (err) {
    next(err);
  }
});

export default router;
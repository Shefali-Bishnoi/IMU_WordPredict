import { Router } from 'express';
import logger from '../logger.js';
import { pythonStroke } from '../services/pythonClient.js';
import { getSession, SessionState } from '../session/sessionManager.js';

const router = Router();

function isValidSensor(sensor) {
  return (
    Array.isArray(sensor) &&
    sensor.length > 0 &&
    sensor.every(
      (row) => Array.isArray(row) && row.length === 9 && row.every((v) => typeof v === 'number' && Number.isFinite(v))
    )
  );
}

/**
 * POST /api/predict/character
 *
 * "Predict Character" == the explicit commit-per-character boundary the
 * user described: everything accumulated since the previous predict call
 * (or since session start) is treated as ONE character instance, exactly
 * matching how the training dataset itself is structured (many IMU rows
 * per character file). There is no automatic character-boundary
 * detection here or anywhere else in this system.
 */
router.post('/predict/character', async (req, res, next) => {
  try {
    const { sessionId, sensor, topK } = req.body;
    if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

    const session = getSession(sessionId);
    if (!session) return res.status(404).json({ error: `Unknown sessionId: ${sessionId}` });
    if (session.state === SessionState.STOPPED) {
      return res.status(409).json({ error: 'Session is stopped' });
    }

    const strokeToUse = sensor !== undefined ? sensor : session.currentStroke;
    if (!isValidSensor(strokeToUse)) {
      return res.status(400).json({
        error: 'sensor must be a non-empty array of rows, each with exactly 9 numeric values',
      });
    }

    const prevState = session.state;
    session.state = SessionState.PREDICTING;

    const t0 = Date.now();
    const result = await pythonStroke(session.pythonSessionId, strokeToUse, topK ?? 5);
    const latencyMs = Date.now() - t0;

    session.committedCharacters.push(result.character);
    session.currentWordRaw = result.current_word_raw;
    session.currentStroke = [];
    session.predictionCount += 1;
    session.lastPrediction = result;
    session.debug.lastPredictLatencyMs = latencyMs;
    session.state = prevState === SessionState.STOPPED ? SessionState.STOPPED : SessionState.RUNNING;

    logger.info(
      `[INFERENCE] session=${sessionId} character prediction latency=${latencyMs}ms char=${result.character}`
    );

    res.json({
      sessionId,
      character: result.character,
      confidence: result.confidence,
      topK: result.top_k,
      currentWordRaw: session.currentWordRaw,
      committedCharacters: session.committedCharacters,
      pipeline: {
        stages: ['preprocessing', 'character_model'],
        note:
          'Beam search / dictionary / language-model correction happen at word-commit ' +
          'time (see /api/word/commit), not per character.',
      },
      latencyMs,
    });
  } catch (err) {
    next(err);
  }
});

export default router;
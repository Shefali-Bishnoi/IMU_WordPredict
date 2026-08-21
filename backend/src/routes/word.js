import { Router } from 'express';
import logger from '../logger.js';
import { pythonCommit } from '../services/pythonClient.js';
import { getSession, SessionState } from '../session/sessionManager.js';

const router = Router();

/**
 * POST /api/word/commit
 *
 * "Commit Word" == the explicit word-boundary the user's marker cannot
 * express with a stroke. Runs the EXISTING beam search + dictionary +
 * (if loaded) n-gram language-model correction on the Python side
 * (app/correction.py) -- Node does not reimplement any of it.
 */
router.post('/word/commit', async (req, res, next) => {
  try {
    const { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

    const session = getSession(sessionId);
    if (!session) return res.status(404).json({ error: `Unknown sessionId: ${sessionId}` });
    if (session.committedCharacters.length === 0) {
      return res.status(409).json({ error: 'No characters written since the last commit' });
    }

    const prevState = session.state;
    session.state = SessionState.WORD_COMMITTING;

    const t0 = Date.now();
    const result = await pythonCommit(session.pythonSessionId);
    const latencyMs = Date.now() - t0;

    session.lastWordResult = result;
    session.committedCharacters = [];
    session.currentWordRaw = '';
    session.debug.lastCommitLatencyMs = latencyMs;
    session.state = prevState === SessionState.STOPPED ? SessionState.STOPPED : SessionState.RUNNING;

    logger.info(`[WORD] session=${sessionId} committed="${result.corrected_word}" latency=${latencyMs}ms`);

    res.json({
      sessionId,
      rawWord: result.raw_word,
      correctedWord: result.corrected_word,
      confidence: result.confidence,
      isLowConfidence: result.is_low_confidence,
      textSoFar: result.text_so_far,
      pipeline: {
        stages: ['beam_search', 'dictionary_correction', 'language_model_rescoring'],
        beamScore: result.beam_score ?? null,
        editSimilarity: result.edit_similarity ?? null,
        wordFrequency: result.word_frequency ?? null,
        languageModelScore: result.lm_score ?? null,
        ...(result.beam_score === undefined
          ? { note: 'Granular beam/dictionary/LM scores not returned by this python service version.' }
          : {}),
      },
      latencyMs,
    });
  } catch (err) {
    next(err);
  }
});

export default router;
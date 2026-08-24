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
 *
 * NEW (additive, Level-3 contextual correction): the Python /commit
 * response now ALSO carries finalWord/languageModelUsed/reranked/
 * context/topCandidates -- the result of reranking the word decoder's
 * top candidates against the session's preceding committed text using
 * a pretrained causal language model (see app/main.py /
 * language/contextual_scorer.py). This is a RERANK of words the
 * existing pipeline already proposed, never a new generated word.
 * There is still NO "commit sentence" operation -- this remains a
 * per-word call, exactly as before.
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

    logger.info(
      `[WORD] session=${sessionId} committed="${result.corrected_word}" ` +
      `final="${result.final_word ?? result.corrected_word}" ` +
      `lm_used=${Boolean(result.language_model_used)} latency=${latencyMs}ms`
    );

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
      // NEW (additive): Level-3 contextual correction block. Present
      // (with sensible defaults) even when the language layer is
      // disabled/unavailable, so the frontend never has to special-case
      // a missing field -- languageModelUsed simply reads false.
      contextual: {
        finalWord: result.final_word ?? result.corrected_word,
        languageModelUsed: Boolean(result.language_model_used),
        reranked: Boolean(result.reranked),
        context: result.context ?? '',
        topCandidates: (result.top_candidates ?? []).map((c) => ({
          word: c.word,
          finalScore: c.final_score,
          lmLogProb: c.lm_log_prob ?? null,
          lmScore: c.lm_score ?? null,
          combinedScore: c.combined_score ?? null,
        })),
      },
      latencyMs,
    });
  } catch (err) {
    next(err);
  }
});

export default router;

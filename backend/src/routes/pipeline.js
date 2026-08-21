import { Router } from 'express';
import { pythonModelInfo } from '../services/pythonClient.js';

const router = Router();

/**
 * GET /api/pipeline/status
 * Reports which pipeline stages ACTUALLY exist on the currently
 * connected Python service (via GET /model/info), never invented.
 */
router.get('/pipeline/status', async (req, res, next) => {
  try {
    const info = await pythonModelInfo();
    res.json({
      stages: [
        { name: 'preprocessing', available: true },
        { name: 'character_model', available: true, architecture: info.architecture },
        { name: 'beam_search', available: Boolean(info.beam_search_available) },
        { name: 'dictionary_correction', available: Boolean(info.dictionary_correction_available) },
        {
          name: 'language_model',
          available: Boolean(info.ngram_language_model_available),
          detail: info.ngram_language_model_available ? `n-gram order=${info.ngram_order}` : 'not loaded',
        },
        {
          name: 'personalization',
          available: Boolean(info.personalization_available),
          active: false,
          detail:
            'Personalization endpoints exist on the python service ' +
            '(/session/{id}/correct-character, /session/{id}/personalized-predict) ' +
            'but are not wired into this realtime session/UI flow yet.',
        },
      ],
      decoderWeights: info.decoder_weights ?? null,
      tauWord: info.tau_word ?? null,
      searchLambdaLm: info.search_lambda_lm ?? null,
    });
  } catch (err) {
    next(err);
  }
});

export default router;
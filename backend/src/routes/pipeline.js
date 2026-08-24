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
        {
          // NEW (additive): Level-3 contextual correction (pretrained
          // causal LM reranking committed words against preceding
          // context). Distinct from the character-n-gram
          // 'language_model' stage above, which scores raw character
          // sequences during beam search, not committed words against
          // sentence-level context.
          name: 'contextual_correction',
          available: Boolean(info.language_model_available),
          enabledInConfig: Boolean(info.language_model_enabled_config),
          modelName: info.language_model_name ?? null,
          weight: info.language_model_weight ?? null,
          contextWords: info.language_context_words ?? null,
          detail: info.language_model_available
            ? `pretrained causal LM (${info.language_model_name}) reranking committed-word candidates`
            : (info.language_model_enabled_config
              ? 'enabled in config but failed to load -- falling back to existing pipeline'
              : 'disabled (LANGUAGE_MODEL_ENABLED=false)'),
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

/**
 * Hand-written OpenAPI 3.0 document (not jsdoc-derived, to guarantee
 * every field actually documented is accurate). Covers REST only --
 * the WebSocket protocol is documented in REALTIME_SYSTEM.md, since
 * OpenAPI has no native WebSocket support.
 */
const openapiSpec = {
  openapi: '3.0.3',
  info: {
    title: 'WordPredict Realtime Backend API',
    version: '1.0.0',
    description:
      'Node.js backend for the WordPredict IMU air-writing system. This backend ' +
      'manages realtime sessions (WebSocket + REST) and forwards ML inference/' +
      'decoding work to the existing Python FastAPI inference service. It does ' +
      'NOT implement preprocessing, model inference, beam search, dictionary ' +
      'correction, or personalization itself -- see the Python service for that. ' +
      'For the WebSocket protocol (not representable in OpenAPI), see REALTIME_SYSTEM.md.',
  },
  servers: [{ url: '/', description: 'This server' }],
  tags: [
    { name: 'Health' },
    { name: 'Session' },
    { name: 'Prediction' },
    { name: 'Word' },
    { name: 'Pipeline' },
    { name: 'Model' },
  ],
  paths: {
    '/health': {
      get: {
        tags: ['Health'],
        summary: 'Node backend + Python inference service health',
        responses: {
          200: {
            description: 'Health status',
            content: {
              'application/json': {
                example: {
                  status: 'ok',
                  service: 'wordpredict-node-backend',
                  timestamp: '2026-08-19T10:00:00.000Z',
                  python: { reachable: true, status: 'ok', model_loaded: true },
                },
              },
            },
          },
        },
      },
    },
    '/api/session/start': {
      post: {
        tags: ['Session'],
        summary: 'Start a new realtime session',
        description:
          'Creates a new Node-side session AND a corresponding session on the Python ' +
          'inference service. Returns the sessionId used by every other endpoint.',
        responses: {
          201: {
            description: 'Session created',
            content: {
              'application/json': {
                example: {
                  sessionId: '6e1b7c9a-...',
                  pythonSessionId: 'a1b2c3d4...',
                  state: 'RUNNING',
                  currentStrokeLength: 0,
                  committedCharacters: [],
                  currentWordRaw: '',
                  sampleCount: 0,
                  predictionCount: 0,
                  createdAt: '2026-08-19T10:00:00.000Z',
                  startedAt: '2026-08-19T10:00:00.000Z',
                  debug: { lastPredictLatencyMs: null, lastCommitLatencyMs: null },
                },
              },
            },
          },
          502: { description: 'Python inference service returned an error' },
          503: { description: 'Python inference service unreachable' },
        },
      },
    },
    '/api/session/status': {
      get: {
        tags: ['Session'],
        summary: 'Get current session status',
        parameters: [
          { name: 'sessionId', in: 'query', required: true, schema: { type: 'string' } },
        ],
        responses: {
          200: { description: 'Session status' },
          400: { description: 'Missing sessionId' },
          404: { description: 'Unknown sessionId' },
        },
      },
    },
    '/api/session/stop': {
      post: {
        tags: ['Session'],
        summary: 'Stop a session',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['sessionId'],
                properties: { sessionId: { type: 'string' } },
              },
            },
          },
        },
        responses: {
          200: { description: 'Session stopped' },
          400: { description: 'Missing sessionId' },
          404: { description: 'Unknown sessionId' },
        },
      },
    },
    '/api/session/reset': {
      post: {
        tags: ['Session'],
        summary: 'Reset a session (fresh word/stroke buffers, fresh Python session)',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['sessionId'],
                properties: { sessionId: { type: 'string' } },
              },
            },
          },
        },
        responses: {
          200: { description: 'Session reset' },
          400: { description: 'Missing sessionId' },
          404: { description: 'Unknown sessionId' },
        },
      },
    },
    '/api/predict/character': {
      post: {
        tags: ['Prediction'],
        summary: 'Predict the character for an accumulated IMU stroke',
        description:
          'Runs the existing Python preprocessing + character model on the given ' +
          'sensor rows -- one full character instance (many IMU rows, matching the ' +
          'training data format), explicitly bounded by this call. There is no ' +
          'automatic character-boundary detection anywhere in this system.',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['sessionId'],
                properties: {
                  sessionId: { type: 'string' },
                  sensor: {
                    type: 'array',
                    description:
                      'Optional. Rows of [ax,ay,az,gx,gy,gz,mx,my,mz]. If omitted, the ' +
                      'sensor rows buffered server-side via WebSocket "sensor" messages ' +
                      'for this session are used instead.',
                    items: { type: 'array', items: { type: 'number' }, minItems: 9, maxItems: 9 },
                  },
                  topK: { type: 'integer', default: 5 },
                },
              },
              example: {
                sessionId: '6e1b7c9a-...',
                sensor: [[0.1, -0.2, 0.9, 1.1, -0.3, 0.2, 40, -50, -100]],
                topK: 5,
              },
            },
          },
        },
        responses: {
          200: {
            description: 'Character prediction result',
            content: {
              'application/json': {
                example: {
                  sessionId: '6e1b7c9a-...',
                  character: 'a',
                  confidence: 0.91,
                  topK: [
                    { char: 'a', p: 0.91 },
                    { char: 'o', p: 0.04 },
                  ],
                  currentWordRaw: 'a',
                  committedCharacters: ['a'],
                  pipeline: { stages: ['preprocessing', 'character_model'] },
                  latencyMs: 42,
                },
              },
            },
          },
          400: { description: 'Missing/invalid sessionId or sensor data' },
          404: { description: 'Unknown sessionId' },
          409: { description: 'Session is stopped' },
          502: { description: 'Python inference service error' },
        },
      },
    },
    '/api/word/commit': {
      post: {
        tags: ['Word'],
        summary: 'Commit the currently accumulated character sequence as a word',
        description:
          'Runs the existing beam search + dictionary/language-model correction ' +
          'pipeline on the characters accumulated since the last commit. There is ' +
          'no automatic word-boundary detection -- this call IS the boundary.',
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                required: ['sessionId'],
                properties: { sessionId: { type: 'string' } },
              },
            },
          },
        },
        responses: {
          200: {
            description: 'Committed word result',
            content: {
              'application/json': {
                example: {
                  sessionId: '6e1b7c9a-...',
                  rawWord: 'aplle',
                  correctedWord: 'apple',
                  confidence: 0.87,
                  isLowConfidence: false,
                  textSoFar: 'apple',
                  pipeline: {
                    stages: ['beam_search', 'dictionary_correction', 'language_model_rescoring'],
                    beamScore: 0.8,
                    editSimilarity: 0.9,
                    wordFrequency: 0.7,
                    languageModelScore: 0.6,
                  },
                  latencyMs: 55,
                },
              },
            },
          },
          400: { description: 'Missing sessionId' },
          404: { description: 'Unknown sessionId' },
          409: { description: 'No characters written since the last commit' },
          502: { description: 'Python inference service error' },
        },
      },
    },
    '/api/pipeline/status': {
      get: {
        tags: ['Pipeline'],
        summary: 'Which pipeline stages are actually available on the connected Python service',
        responses: {
          200: {
            description: 'Pipeline stage availability, derived from GET /model/info -- never invented',
          },
        },
      },
    },
    '/api/model/info': {
      get: {
        tags: ['Model'],
        summary: 'Model / decoder metadata (architecture, decoder weights, LM availability)',
        responses: { 200: { description: 'Model info, pass-through of the Python service' } },
      },
    },
  },
};

export default openapiSpec;
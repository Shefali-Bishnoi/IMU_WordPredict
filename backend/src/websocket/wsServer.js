/**
 * WebSocket protocol implementation. See REALTIME_SYSTEM.md for the
 * full documented protocol (message schemas + example sequences).
 *
 * One WebSocket connection <-> at most one active Node session, tracked
 * via ws.sessionId. This mirrors the REST session model exactly (same
 * sessionManager, same Python session underneath) -- WS is just a more
 * convenient transport for continuous sensor streaming.
 */
import { WebSocketServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';
import logger from '../logger.js';
import { pythonCommit, pythonStroke } from '../services/pythonClient.js';
import {
  createSession, getSession, resetSession, SessionState, stopSession,
} from '../session/sessionManager.js';

function envelope(type, sessionId, data, requestId, error) {
  return JSON.stringify({
    type,
    timestamp: new Date().toISOString(),
    session_id: sessionId ?? null,
    request_id: requestId ?? null,
    data: data ?? null,
    error: error ?? null,
  });
}

function isValidSensorRow(row) {
  return (
    Array.isArray(row) &&
    row.length === 9 &&
    row.every((v) => typeof v === 'number' && Number.isFinite(v))
  );
}

export function attachWebSocketServer(server) {
  const wss = new WebSocketServer({ server, path: '/ws' });

  wss.on('connection', (ws) => {
    const connectionId = uuidv4();
    ws.sessionId = null;
    logger.info(`[WS] client connected connection=${connectionId}`);
    ws.send(envelope('connected', null, { connectionId }, null));

    ws.on('message', async (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw.toString());
      } catch (e) {
        ws.send(envelope('error', ws.sessionId, null, null, 'Malformed JSON message'));
        return;
      }

      const { type, request_id: requestId } = msg;

      try {
        switch (type) {
          case 'start_session': {
            const existing = getSession(ws.sessionId);
            if (existing && existing.state !== SessionState.STOPPED) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'A session is already active on this connection'));
              return;
            }
            const session = await createSession();
            ws.sessionId = session.id;
            logger.info(`[WS] session_started session=${session.id}`);
            ws.send(envelope('session_started', session.id, session.toStatus(), requestId));
            break;
          }

          case 'sensor': {
            const session = getSession(ws.sessionId);
            if (!session || session.state === SessionState.STOPPED) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'No active session'));
              return;
            }
            if (!isValidSensorRow(msg.data)) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'sensor data must be an array of 9 numbers'));
              return;
            }
            session.currentStroke.push(msg.data);
            session.sampleCount += 1;
            ws.send(
              envelope('sensor_ack', ws.sessionId, {
                strokeLength: session.currentStroke.length,
                totalSamples: session.sampleCount,
              }, requestId)
            );
            break;
          }

          case 'predict_character': {
            const session = getSession(ws.sessionId);
            if (!session || session.state === SessionState.STOPPED) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'No active session'));
              return;
            }
            if (session.currentStroke.length === 0) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'No current stroke to predict from'));
              return;
            }

            const prevState = session.state;
            session.state = SessionState.PREDICTING;
            const t0 = Date.now();
            const result = await pythonStroke(session.pythonSessionId, session.currentStroke, 5);
            const latencyMs = Date.now() - t0;

            session.committedCharacters.push(result.character);
            session.currentWordRaw = result.current_word_raw;
            session.currentStroke = [];
            session.predictionCount += 1;
            session.lastPrediction = result;
            session.debug.lastPredictLatencyMs = latencyMs;
            session.state = prevState === SessionState.STOPPED ? SessionState.STOPPED : SessionState.RUNNING;

            logger.info(
              `[INFERENCE] session=${session.id} character prediction latency=${latencyMs}ms char=${result.character}`
            );

            ws.send(
              envelope('prediction_update', session.id, {
                character: {
                  predicted: result.character,
                  confidence: result.confidence,
                  top_k: result.top_k,
                },
                currentWordRaw: session.currentWordRaw,
                committedCharacters: session.committedCharacters,
                pipeline: {
                  stages: ['preprocessing', 'character_model'],
                  note:
                    'Beam search / dictionary / language-model correction happen at ' +
                    'word-commit time, not per character.',
                },
                latencyMs,
              }, requestId)
            );
            break;
          }

          case 'commit_word': {
            const session = getSession(ws.sessionId);
            if (!session || session.state === SessionState.STOPPED) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'No active session'));
              return;
            }
            if (session.committedCharacters.length === 0) {
              ws.send(envelope('error', ws.sessionId, null, requestId, 'No characters written since the last commit'));
              return;
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

            logger.info(`[WORD] session=${session.id} committed="${result.corrected_word}" latency=${latencyMs}ms`);

            ws.send(
              envelope('word_committed', session.id, {
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
                },
                latencyMs,
              }, requestId)
            );
            break;
          }

          case 'reset_session': {
            if (!ws.sessionId) {
              ws.send(envelope('error', null, null, requestId, 'No active session to reset'));
              return;
            }
            const session = await resetSession(ws.sessionId);
            ws.send(envelope('session_reset', ws.sessionId, session.toStatus(), requestId));
            break;
          }

          case 'stop_session': {
            if (!ws.sessionId) {
              ws.send(envelope('error', null, null, requestId, 'No active session to stop'));
              return;
            }
            const session = await stopSession(ws.sessionId);
            ws.send(envelope('session_stopped', ws.sessionId, session.toStatus(), requestId));
            break;
          }

          default:
            ws.send(envelope('error', ws.sessionId, null, requestId, `Unknown message type: ${type}`));
        }
      } catch (err) {
        logger.error(`[WS] handler error type=${type}: ${err.message}`);
        ws.send(envelope('error', ws.sessionId, null, requestId, err.message || 'Internal error'));
      }
    });

    ws.on('close', () => {
      logger.info(`[WS] client disconnected connection=${connectionId} session=${ws.sessionId}`);
    });
  });

  return wss;
}
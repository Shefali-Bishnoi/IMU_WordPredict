/**
 * Node-side session state machine, layered on top of the Python
 * service's own session (app/session.py). One Node session <-> exactly
 * one Python session.
 *
 * State machine:
 *   IDLE (never persisted -- a session only exists once RUNNING)
 *     -> RUNNING
 *     -> PREDICTING (transient, during a predict_character call)
 *     -> RUNNING
 *     -> WORD_COMMITTING (transient, during a commit_word call)
 *     -> RUNNING
 *     -> STOPPED
 *
 * In-memory only, per project's "don't over-engineer" instruction --
 * matches the Python side's own in-memory SessionStore (app/session.py).
 * Swap for Redis/a DB before any multi-process/multi-worker deployment.
 */
import { v4 as uuidv4 } from 'uuid';
import logger from '../logger.js';
import { pythonEndSession, pythonStartSession } from '../services/pythonClient.js';

export const SessionState = Object.freeze({
  RUNNING: 'RUNNING',
  PREDICTING: 'PREDICTING',
  WORD_COMMITTING: 'WORD_COMMITTING',
  STOPPED: 'STOPPED',
});

class Session {
  constructor(id, pythonSessionId) {
    this.id = id;
    this.pythonSessionId = pythonSessionId;
    this.state = SessionState.RUNNING;

    // Buffers -- see ActionPlan.md's real-time inference path (Sec.15)
    // and the commit-button word-boundary design (app/session.py).
    this.currentStroke = [];        // raw sensor rows for the IN-PROGRESS character
    this.committedCharacters = [];  // characters predicted since the last word commit
    this.currentWordRaw = '';       // authoritative value from Python's /stroke response

    this.lastPrediction = null;
    this.lastWordResult = null;

    this.createdAt = new Date().toISOString();
    this.startedAt = this.createdAt;
    this.sampleCount = 0;
    this.predictionCount = 0;

    this.debug = {
      lastPredictLatencyMs: null,
      lastCommitLatencyMs: null,
    };
  }

  toStatus() {
    return {
      sessionId: this.id,
      pythonSessionId: this.pythonSessionId,
      state: this.state,
      currentStrokeLength: this.currentStroke.length,
      committedCharacters: this.committedCharacters,
      currentWordRaw: this.currentWordRaw,
      sampleCount: this.sampleCount,
      predictionCount: this.predictionCount,
      createdAt: this.createdAt,
      startedAt: this.startedAt,
      debug: this.debug,
    };
  }
}

const sessions = new Map();

export async function createSession() {
  const pythonSessionId = await pythonStartSession();
  const id = uuidv4();
  const session = new Session(id, pythonSessionId);
  sessions.set(id, session);
  logger.info(`[SESSION] started session=${id} python_session=${pythonSessionId}`);
  return session;
}

export function getSession(id) {
  return id ? sessions.get(id) || null : null;
}

export async function stopSession(id) {
  const session = getSession(id);
  if (!session) return null;
  try {
    await pythonEndSession(session.pythonSessionId);
  } catch (err) {
    logger.warn(`[SESSION] error ending python session ${session.pythonSessionId}: ${err.message}`);
  }
  session.state = SessionState.STOPPED;
  logger.info(`[SESSION] stopped session=${id}`);
  return session;
}

export async function resetSession(id) {
  const existing = getSession(id);
  if (!existing) return null;
  try {
    await pythonEndSession(existing.pythonSessionId);
  } catch (err) {
    logger.warn(`[SESSION] error ending python session during reset: ${err.message}`);
  }
  const pythonSessionId = await pythonStartSession();

  existing.pythonSessionId = pythonSessionId;
  existing.state = SessionState.RUNNING;
  existing.currentStroke = [];
  existing.committedCharacters = [];
  existing.currentWordRaw = '';
  existing.lastPrediction = null;
  existing.lastWordResult = null;
  existing.sampleCount = 0;
  existing.predictionCount = 0;
  existing.startedAt = new Date().toISOString();

  logger.info(`[SESSION] reset session=${id} new_python_session=${pythonSessionId}`);
  return existing;
}

export function deleteSession(id) {
  sessions.delete(id);
}

export function _debugAllSessions() {
  return sessions;
}
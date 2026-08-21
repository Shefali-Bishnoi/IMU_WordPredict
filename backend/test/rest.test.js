/**
 * REST integration tests. Runs against a stub Python HTTP server
 * (no real ML/model required) so this suite is fast and self-contained.
 *
 *   node --test test/
 */
import assert from 'node:assert/strict';
import http from 'node:http';
import { test } from 'node:test';

async function startStubPython() {
  const sessions = new Map();
  let counter = 0;

  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      const url = req.url;

      if (req.method === 'GET' && url === '/health') {
        res.end(JSON.stringify({ status: 'ok', model_loaded: true }));
        return;
      }
      if (req.method === 'GET' && url === '/model/info') {
        res.end(JSON.stringify({
          architecture: 'tcn',
          seq_len: 80,
          num_classes: 52,
          beam_search_available: true,
          dictionary_correction_available: true,
          ngram_language_model_available: true,
          ngram_order: 4,
          personalization_available: true,
          tau_word: 0.569,
          search_lambda_lm: 0.3,
          decoder_weights: { alpha: 0.05, beta: 0.85, gamma: 0.05, delta: 0.05 },
        }));
        return;
      }
      if (req.method === 'POST' && url === '/session/start') {
        counter += 1;
        const id = `py-session-${counter}`;
        sessions.set(id, { characters: [] });
        res.end(JSON.stringify({ session_id: id }));
        return;
      }
      if (req.method === 'POST' && /^\/session\/[^/]+\/stroke$/.test(url)) {
        const sid = url.split('/')[2];
        const session = sessions.get(sid);
        if (session) session.characters.push('a');
        res.end(JSON.stringify({
          character: 'a',
          confidence: 0.9,
          top_k: [{ char: 'a', p: 0.9 }, { char: 'o', p: 0.05 }],
          current_word_raw: session ? session.characters.join('') : 'a',
        }));
        return;
      }
      if (req.method === 'POST' && /^\/session\/[^/]+\/commit$/.test(url)) {
        const sid = url.split('/')[2];
        const session = sessions.get(sid);
        const raw = session ? session.characters.join('') : 'a';
        if (session) session.characters = [];
        res.end(JSON.stringify({
          raw_word: raw, corrected_word: raw, confidence: 0.95,
          is_low_confidence: false, text_so_far: raw,
        }));
        return;
      }
      if (req.method === 'POST' && /^\/session\/[^/]+\/end$/.test(url)) {
        res.end(JSON.stringify({ text: '' }));
        return;
      }
      res.statusCode = 404;
      res.end(JSON.stringify({ error: 'not found' }));
    });
  });

  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();
  return { server, baseUrl: `http://localhost:${port}` };
}

let stub;
let appModule;

test('setup', async () => {
  stub = await startStubPython();
  process.env.PYTHON_BASE_URL = stub.baseUrl;
  appModule = await import('../src/server.js');
});

test('GET /health reports python reachable', async () => {
  const app = appModule.createApp();
  const server = app.listen(0);
  const { port } = server.address();
  const res = await fetch(`http://localhost:${port}/health`);
  const body = await res.json();
  assert.equal(res.status, 200);
  assert.equal(body.python.reachable, true);
  server.close();
});

test('full session lifecycle: start -> predict -> commit -> stop', async () => {
  const app = appModule.createApp();
  const server = app.listen(0);
  const base = `http://localhost:${server.address().port}`;

  const startRes = await fetch(`${base}/api/session/start`, { method: 'POST' });
  const startBody = await startRes.json();
  assert.equal(startRes.status, 201);
  assert.ok(startBody.sessionId);

  const predictRes = await fetch(`${base}/api/predict/character`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: startBody.sessionId, sensor: [new Array(9).fill(0.1)] }),
  });
  const predictBody = await predictRes.json();
  assert.equal(predictRes.status, 200);
  assert.equal(predictBody.character, 'a');

  const commitRes = await fetch(`${base}/api/word/commit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: startBody.sessionId }),
  });
  const commitBody = await commitRes.json();
  assert.equal(commitRes.status, 200);
  assert.equal(commitBody.correctedWord, 'a');

  const stopRes = await fetch(`${base}/api/session/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: startBody.sessionId }),
  });
  assert.equal(stopRes.status, 200);

  server.close();
});

test('predict/character without sessionId returns 400', async () => {
  const app = appModule.createApp();
  const server = app.listen(0);
  const res = await fetch(`http://localhost:${server.address().port}/api/predict/character`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: [[0, 0, 0, 0, 0, 0, 0, 0, 0]] }),
  });
  assert.equal(res.status, 400);
  server.close();
});

test('word/commit with no characters returns 409', async () => {
  const app = appModule.createApp();
  const server = app.listen(0);
  const base = `http://localhost:${server.address().port}`;
  const startRes = await fetch(`${base}/api/session/start`, { method: 'POST' });
  const startBody = await startRes.json();
  const res = await fetch(`${base}/api/word/commit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId: startBody.sessionId }),
  });
  assert.equal(res.status, 409);
  server.close();
});

test('session/status with unknown sessionId returns 404', async () => {
  const app = appModule.createApp();
  const server = app.listen(0);
  const res = await fetch(`http://localhost:${server.address().port}/api/session/status?sessionId=nope`);
  assert.equal(res.status, 404);
  server.close();
});

test('teardown', () => {
  stub.server.close();
});
/**
 * WebSocket integration tests, against a stub Python HTTP server.
 *   node --test test/
 */
import assert from 'node:assert/strict';
import http from 'node:http';
import { test } from 'node:test';
import { WebSocket } from 'ws';

async function startStubPython() {
  const sessions = new Map();
  let counter = 0;
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      const url = req.url;
      if (req.method === 'GET' && url === '/health') { res.end(JSON.stringify({ status: 'ok' })); return; }
      if (req.method === 'POST' && url === '/session/start') {
        counter += 1;
        const id = `py-${counter}`;
        sessions.set(id, { chars: [] });
        res.end(JSON.stringify({ session_id: id }));
        return;
      }
      if (req.method === 'POST' && /^\/session\/[^/]+\/stroke$/.test(url)) {
        const sid = url.split('/')[2];
        const s = sessions.get(sid);
        if (s) s.chars.push('h');
        res.end(JSON.stringify({
          character: 'h',
          confidence: 0.88,
          top_k: [{ char: 'h', p: 0.88 }],
          current_word_raw: s ? s.chars.join('') : 'h',
        }));
        return;
      }
      if (req.method === 'POST' && /^\/session\/[^/]+\/commit$/.test(url)) {
        const sid = url.split('/')[2];
        const s = sessions.get(sid);
        const raw = s ? s.chars.join('') : 'h';
        if (s) s.chars = [];
        res.end(JSON.stringify({
          raw_word: raw, corrected_word: raw, confidence: 0.9,
          is_low_confidence: false, text_so_far: raw,
        }));
        return;
      }
      if (req.method === 'POST' && /^\/session\/[^/]+\/end$/.test(url)) {
        res.end(JSON.stringify({ text: '' }));
        return;
      }
      res.statusCode = 404;
      res.end('{}');
    });
  });
  await new Promise((r) => server.listen(0, r));
  return { server, baseUrl: `http://localhost:${server.address().port}` };
}

function waitForMessage(ws) {
  return new Promise((resolve) => {
    ws.once('message', (data) => resolve(JSON.parse(data.toString())));
  });
}

async function startNodeServer() {
  const { createApp } = await import('../src/server.js');
  const { attachWebSocketServer } = await import('../src/websocket/wsServer.js');
  const http2 = await import('node:http');
  const app = createApp();
  const server = http2.createServer(app);
  attachWebSocketServer(server);
  await new Promise((r) => server.listen(0, r));
  return server;
}

test('websocket full flow: start -> sensor -> predict -> commit -> reset -> stop', async () => {
  const stub = await startStubPython();
  process.env.PYTHON_BASE_URL = stub.baseUrl;
  const server = await startNodeServer();
  const port = server.address().port;

  const ws = new WebSocket(`ws://localhost:${port}/ws`);
  await new Promise((resolve) => ws.once('open', resolve));

  const connectedMsg = await waitForMessage(ws);
  assert.equal(connectedMsg.type, 'connected');

  ws.send(JSON.stringify({ type: 'start_session', request_id: '1' }));
  const started = await waitForMessage(ws);
  assert.equal(started.type, 'session_started');

  ws.send(JSON.stringify({ type: 'sensor', data: new Array(9).fill(0.2) }));
  const ack = await waitForMessage(ws);
  assert.equal(ack.type, 'sensor_ack');
  assert.equal(ack.data.strokeLength, 1);

  ws.send(JSON.stringify({ type: 'predict_character', request_id: '2' }));
  const pred = await waitForMessage(ws);
  assert.equal(pred.type, 'prediction_update');
  assert.equal(pred.data.character.predicted, 'h');

  ws.send(JSON.stringify({ type: 'commit_word', request_id: '3' }));
  const committed = await waitForMessage(ws);
  assert.equal(committed.type, 'word_committed');
  assert.equal(committed.data.correctedWord, 'h');

  ws.send(JSON.stringify({ type: 'reset_session', request_id: '4' }));
  const reset = await waitForMessage(ws);
  assert.equal(reset.type, 'session_reset');

  ws.send(JSON.stringify({ type: 'stop_session', request_id: '5' }));
  const stopped = await waitForMessage(ws);
  assert.equal(stopped.type, 'session_stopped');

  ws.close();
  server.close();
  stub.server.close();
});

test('websocket rejects predict_character with no active session', async () => {
  const stub = await startStubPython();
  process.env.PYTHON_BASE_URL = stub.baseUrl;
  const server = await startNodeServer();
  const port = server.address().port;

  const ws = new WebSocket(`ws://localhost:${port}/ws`);
  await new Promise((resolve) => ws.once('open', resolve));
  await waitForMessage(ws); // connected

  ws.send(JSON.stringify({ type: 'predict_character' }));
  const err = await waitForMessage(ws);
  assert.equal(err.type, 'error');

  ws.close();
  server.close();
  stub.server.close();
});

test('websocket rejects malformed JSON gracefully', async () => {
  const stub = await startStubPython();
  process.env.PYTHON_BASE_URL = stub.baseUrl;
  const server = await startNodeServer();
  const port = server.address().port;

  const ws = new WebSocket(`ws://localhost:${port}/ws`);
  await new Promise((resolve) => ws.once('open', resolve));
  await waitForMessage(ws); // connected

  ws.send('{not valid json');
  const err = await waitForMessage(ws);
  assert.equal(err.type, 'error');

  ws.close();
  server.close();
  stub.server.close();
});
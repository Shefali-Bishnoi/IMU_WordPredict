/**
 * WordPredict realtime console.
 *
 * ARCHITECTURE: three interchangeable sensor sources (Demo, Training
 * Sample paste, Marker bridge) all funnel into ONE captureRow() function,
 * which is the ONLY place that (a) updates the local UI buffer/table/
 * sparklines and (b) sends the existing 'sensor' WebSocket message to
 * the Node backend. There is exactly one prediction path
 * ('predict_character' / 'commit_word', both pre-existing, unchanged) --
 * the source selector only changes where rows come from, never how a
 * character or word gets predicted. This mirrors the SensorSource
 * interface requested in the task spec without over-engineering it into
 * real JS classes, since all three sources reduce to "call captureRow()
 * with a 9-number array."
 */

const MAX_TABLE_ROWS = 300;      // cap on rendered <tr> rows (perf)
const MAX_SPARKLINE_POINTS = 150; // cap on points drawn per sparkline

const state = {
  ws: null,
  connected: false,
  sessionId: null,

  source: 'demo',              // 'demo' | 'training' | 'marker'
  buffer: [],                  // local mirror of the current character capture (rows of 9 numbers)
  sampleTimestamps: [],        // for rate estimation
  lastFlag: null,              // optional writing-flag metadata from marker rows

  pendingGroundTruth: null,    // set when a training sample with a label is loaded

  markerWs: null,
  minRawLines: null,
  maxRawLines: null,
};

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------
// Logging / status chips
// ---------------------------------------------------------------------
function log(msg) {
  const box = el('log');
  box.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n${box.textContent}`.slice(0, 8000);
}

function setChip(id, level, text) {
  const chip = el(id);
  chip.className = `status-chip ${level}`;
  const dotClass = { on: 'dot-green', warn: 'dot-amber', off: 'dot-red' }[level] || 'dot-gray';
  chip.innerHTML = `<span class="dot ${dotClass}"></span>${text}`;
}

function setBackendStatus(connected) {
  setChip('chip-backend', connected ? 'on' : 'off', connected ? 'Backend: connected' : 'Backend: disconnected');
}

function setMarkerStatus(connected) {
  setChip('chip-marker', connected ? 'on' : 'off', connected ? 'Marker: connected' : 'Marker: disconnected');
}

function setSessionStatus(text, level = 'off') {
  setChip('chip-session', level, `Session: ${text}`);
}

// ---------------------------------------------------------------------
// Main WebSocket (Node backend) -- unchanged protocol
// ---------------------------------------------------------------------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => { state.connected = true; setBackendStatus(true); log('Backend WebSocket connected'); };
  ws.onclose = () => {
    state.connected = false;
    setBackendStatus(false);
    log('Backend WebSocket disconnected — retrying in 2s');
    setTimeout(connect, 2000);
  };
  ws.onerror = () => log('Backend WebSocket error');
  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch (e) { log('Received malformed message from server'); return; }
    handleServerMessage(msg);
  };
}

function send(type, data) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) { log('Cannot send: backend not connected'); return; }
  state.ws.send(JSON.stringify({ type, data, request_id: `${Date.now()}` }));
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case 'connected':
      log(`Connection established (connection ${msg.data.connectionId})`);
      break;

    case 'session_started':
      state.sessionId = msg.session_id;
      setSessionStatus('active', 'on');
      el('meta-session').textContent = msg.session_id;
      el('meta-state').textContent = msg.data.state;
      resetCharacterBuffer();
      resetWord();
      log(`Session started: ${msg.session_id}`);
      renderDebug(msg.data);
      break;

    case 'sensor_ack':
      // Server-side confirmation the row was appended to its own
      // buffer -- local UI already reflects this optimistically via
      // captureRow(), this is just a consistency signal in the log.
      break;

    case 'prediction_update': {
      const c = msg.data.character;
      renderPrediction(c);
      renderPipeline(msg.data.pipeline, c);
      renderDebug(msg.data);
      el('current-word').textContent = msg.data.currentWordRaw || '—';
      renderCharHistory(msg.data.committedCharacters || [], c);
      checkGroundTruth(c.predicted);
      resetCharacterBuffer(); // server buffer was cleared after predicting -- mirror it
      // First character of a fresh word (server resets committedCharacters
      // to [] on every commit -- see wsServer.js's commit_word handler) --
      // clear the PREVIOUS word's result now that a new one has started.
      if ((msg.data.committedCharacters || []).length === 1) {
        el('final-word-section').style.display = 'none';
      }
      break;
    }

    case 'word_committed':
      log(`Word committed: raw="${msg.data.rawWord}" -> corrected="${msg.data.correctedWord}"`);
      el('current-word').textContent = '—';
      el('meta-text-so-far').textContent = msg.data.textSoFar || '—';
      renderPipeline(msg.data.pipeline, null);
      renderFinalWord(msg.data);
      renderDebug(msg.data);
      // NOTE: deliberately NOT calling resetCharHistory() here -- it used
      // to run immediately after renderFinalWord() and set the result
      // panel straight back to display:none in the same tick, so the
      // word result never actually appeared (only the log line above
      // did). The character chips + final word both stay visible so you
      // can see how the word was built, and clear automatically once you
      // start the next word (see the committedCharacters.length === 1
      // check in the prediction_update handler above).
      break;

    case 'session_reset':
      log('Session reset');
      setSessionStatus('active', 'on');
      el('meta-session').textContent = msg.session_id;
      el('meta-state').textContent = msg.data.state;
      el('meta-text-so-far').textContent = '—';
      resetCharacterBuffer();
      resetWord();
      clearPrediction();
      renderDebug(msg.data);
      break;

    case 'session_stopped':
      log('Session stopped');
      setSessionStatus('stopped', 'warn');
      el('meta-state').textContent = msg.data.state;
      renderDebug(msg.data);
      break;

    case 'error':
      log(`ERROR: ${msg.error}`);
      break;

    default:
      log(`Unknown message type: ${msg.type}`);
  }
}

// ---------------------------------------------------------------------
// THE single ingestion point for all three sensor sources.
// ---------------------------------------------------------------------
function captureRow(row9, meta = {}) {
  if (!Array.isArray(row9) || row9.length !== 9 || row9.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
    log(`Rejected malformed row (expected 9 finite numbers): ${JSON.stringify(row9)}`);
    return;
  }
  state.buffer.push(row9);
  state.sampleTimestamps.push(performance.now());
  if (meta.flag !== undefined && meta.flag !== null) state.lastFlag = meta.flag;

  send('sensor', row9);

  renderSampleCount();
  renderRateEstimate();
  appendTableRow(row9);
  renderSparklines();
  if (meta.flag !== undefined) renderWritingFlag(meta.flag);
}

function resetCharacterBuffer() {
  state.buffer = [];
  state.sampleTimestamps = [];
  el('sensor-table-body').innerHTML = '';
  renderSampleCount();
  renderRateEstimate();
  renderSparklines();
}

function renderSampleCount() {
  el('meta-samples').textContent = String(state.buffer.length);
}

function renderRateEstimate() {
  const ts = state.sampleTimestamps;
  if (ts.length < 2) { el('meta-rate').textContent = '—'; return; }
  const window = ts.slice(-20);
  const dt = (window[window.length - 1] - window[0]) / (window.length - 1);
  if (dt <= 0) { el('meta-rate').textContent = '—'; return; }
  el('meta-rate').textContent = `${(1000 / dt).toFixed(0)} Hz`;
}

function renderWritingFlag(flag) {
  const label = flag === 1 ? 'HELD' : flag === 0 ? 'released' : '—';
  el('meta-writing-flag').textContent = label;
}

function appendTableRow(row9) {
  const tbody = el('sensor-table-body');
  const tr = document.createElement('tr');
  const idx = state.buffer.length;
  tr.innerHTML = `<td>${idx}</td>` + row9.map((v) => `<td>${v.toFixed(2)}</td>`).join('');
  tbody.appendChild(tr);
  while (tbody.children.length > MAX_TABLE_ROWS) tbody.removeChild(tbody.firstChild);
  tbody.parentElement.scrollTop = tbody.parentElement.scrollHeight;
}

// ---------------------------------------------------------------------
// Sparklines -- three mini canvases (accel/gyro/mag), 3 lines each.
// ---------------------------------------------------------------------
function drawSparkline(canvasId, colStart) {
  const canvas = el(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 400;
  const cssH = canvas.clientHeight || 56;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const rows = state.buffer.slice(-MAX_SPARKLINE_POINTS);
  if (rows.length < 2) return;

  const colors = ['#4263eb', '#2f9e44', '#e8830c'];
  for (let c = 0; c < 3; c++) {
    const series = rows.map((r) => r[colStart + c]);
    const min = Math.min(...series);
    const max = Math.max(...series);
    const span = (max - min) || 1;
    ctx.beginPath();
    ctx.strokeStyle = colors[c];
    ctx.lineWidth = 1.4;
    series.forEach((v, i) => {
      const x = (i / (series.length - 1)) * cssW;
      const y = cssH - ((v - min) / span) * (cssH - 6) - 3;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function renderSparklines() {
  drawSparkline('spark-accel', 0);
  drawSparkline('spark-gyro', 3);
  drawSparkline('spark-mag', 6);
}

window.addEventListener('resize', renderSparklines);

// ---------------------------------------------------------------------
// Prediction / word-flow rendering
// ---------------------------------------------------------------------
function clearPrediction() {
  el('pred-char').textContent = '—';
  el('pred-conf').textContent = '—';
  el('topk-list').innerHTML = '';
  el('ground-truth-panel').style.display = 'none';
}

function renderPrediction(character) {
  el('pred-char').textContent = character.predicted;
  el('pred-conf').textContent = `${(character.confidence * 100).toFixed(1)}%`;
  const list = el('topk-list');
  list.innerHTML = '';
  (character.top_k || []).forEach((entry) => {
    const row = document.createElement('div');
    row.className = 'top-k-row';
    const pct = Math.round(entry.p * 100);
    row.innerHTML = `
      <span class="top-k-char">${entry.char}</span>
      <span class="top-k-bar-track"><span class="top-k-bar-fill" style="width:${pct}%"></span></span>
      <span class="top-k-pct">${pct}%</span>`;
    list.appendChild(row);
  });
}

function renderCharHistory(committedCharacters, latest) {
  const box = el('char-history');
  box.innerHTML = '';
  committedCharacters.forEach((ch, i) => {
    const chip = document.createElement('div');
    chip.className = 'char-chip';
    const conf = (i === committedCharacters.length - 1 && latest) ? latest.confidence : null;
    chip.innerHTML = `${ch}${conf !== null ? `<span class="conf">${(conf * 100).toFixed(0)}%</span>` : ''}`;
    box.appendChild(chip);
  });
}

function resetCharHistory() {
  el('char-history').innerHTML = '';
  el('final-word-section').style.display = 'none';
}

function resetWord() {
  el('current-word').textContent = '—';
  resetCharHistory();
  clearPrediction();
}

function renderFinalWord(data) {
  const section = el('final-word-section');
  section.style.display = 'block';
  el('final-word').textContent = data.correctedWord || '—';
  el('final-word-raw').textContent = `raw: "${data.rawWord}"  ·  confidence: ${(data.confidence * 100).toFixed(1)}%${data.isLowConfidence ? '  ·  LOW CONFIDENCE' : ''}`;
  const bar = el('final-confidence-bar');
  bar.style.width = `${Math.round(data.confidence * 100)}%`;
  bar.className = `confidence-bar-fill${data.isLowConfidence ? ' low' : ''}`;
}

function renderPipeline(pipeline, character) {
  if (!pipeline) { el('pipeline-diagram').textContent = 'Not available'; return; }
  const lines = ['IMU Stroke', '   ↓', 'Preprocessing', '   ↓', 'Character Model'];
  if (character) {
    lines.push(`   Prediction: ${character.predicted}`);
    lines.push(`   Confidence: ${(character.confidence * 100).toFixed(1)}%`);
  }
  if (pipeline.stages && pipeline.stages.includes('beam_search')) {
    lines.push('   ↓', 'Beam Search');
    lines.push(`   Score: ${pipeline.beamScore ?? 'Not available'}`);
    lines.push('   ↓', 'Dictionary Correction');
    lines.push(`   Edit similarity: ${pipeline.editSimilarity ?? 'Not available'}`);
    lines.push(`   Word frequency: ${pipeline.wordFrequency ?? 'Not available'}`);
    lines.push('   ↓', 'Language Model');
    lines.push(`   Score: ${pipeline.languageModelScore ?? 'Not available'}`);
  }
  lines.push('   ↓', 'Personalization: INACTIVE (not wired into this UI flow)');
  lines.push('   ↓', 'Final Output');
  if (pipeline.note) lines.push('', `Note: ${pipeline.note}`);
  el('pipeline-diagram').textContent = lines.join('\n');
}

function renderDebug(data) {
  el('debug-json').textContent = JSON.stringify(data, null, 2);
}

// ---------------------------------------------------------------------
// Ground truth (Training Sample source only)
// ---------------------------------------------------------------------
function checkGroundTruth(predicted) {
  const panel = el('ground-truth-panel');
  if (!state.pendingGroundTruth) { panel.style.display = 'none'; return; }
  const truth = state.pendingGroundTruth;
  const match = truth.toLowerCase() === String(predicted).toLowerCase();
  panel.style.display = 'flex';
  panel.className = `ground-truth-row ${match ? 'match' : 'mismatch'}`;
  el('ground-truth-text').textContent = `Ground truth: ${truth}  ·  Predicted: ${predicted}  ·  ${match ? 'Correct' : 'Mismatch'}`;
  state.pendingGroundTruth = null;
}

// ---------------------------------------------------------------------
// Sensor source: DEMO
// ---------------------------------------------------------------------
function randomSensorRow() {
  return Array.from({ length: 9 }, () => (Math.random() - 0.5) * 4);
}

// ---------------------------------------------------------------------
// Sensor source: TRAINING SAMPLE (paste raw .txt, same schema as
// config.py: 12 cols, no header -- label(0), timestamp(1), 9 sensor
// values(2-10), writing flag(11)). Extracts ONLY the 9 sensor columns;
// a consistent label across rows is shown as ground truth, never sent
// to the model.
// ---------------------------------------------------------------------
function parseTrainingSample(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  const rows = [];
  const labels = new Set();
  let rejected = 0;

  for (const line of lines) {
    const parts = line.split(',').map((p) => p.trim());
    // Tolerate the documented 13-column quirk (empty field right after
    // the timestamp) the same way preprocessing/io.py's _normalize_row
    // does, without re-deriving every edge case here -- this covers the
    // common real-world case (extra empty slot at index 2).
    let cols = parts;
    if (cols.length === 13 && cols[2] === '') cols = cols.slice(0, 2).concat(cols.slice(3));
    if (cols.length === 13 && cols[cols.length - 1] === '') cols = cols.slice(0, -1);

    if (cols.length !== 12) { rejected++; continue; }

    const sensorVals = cols.slice(2, 11).map((v) => {
      const s = String(v).trim().toLowerCase();
      if (s === '' || s === 'ovf' || s === 'nan') return NaN;
      const f = parseFloat(v);
      return Number.isFinite(f) ? f : NaN;
    });

    if (sensorVals.some((v) => Number.isNaN(v))) {
      // Match the model's own tolerance: ovf/nan values are handled by
      // server-side clean_sensor_matrix (ffill/bfill), not silently
      // dropped here -- but since ffill/bfill needs the whole sequence,
      // do a simple local carry-forward for DISPLAY purposes only, same
      // documented trade-off as hardware/marker_bridge.py's RowParser.
      for (let i = 0; i < sensorVals.length; i++) {
        if (Number.isNaN(sensorVals[i])) {
          sensorVals[i] = rows.length ? rows[rows.length - 1][i] : 0;
        }
      }
    }

    rows.push(sensorVals);
    const label = cols[0].trim();
    if (label) labels.add(label);
  }

  const consistentLabel = labels.size === 1 ? [...labels][0] : null;
  return { rows, rejected, label: consistentLabel, totalLines: lines.length };
}

function loadTrainingSample() {
  const text = el('training-textarea').value;
  if (!text.trim()) {
    el('training-status').textContent = 'Paste a training sample first.';
    return;
  }
  const { rows, rejected, label, totalLines } = parseTrainingSample(text);
  if (rows.length === 0) {
    el('training-status').textContent = `Could not parse any valid rows out of ${totalLines} lines.`;
    log('Training sample load failed: 0 valid rows parsed');
    return;
  }

  // Loading a sample REPLACES the current character buffer -- this is
  // what lets you test predict -> paste next character -> predict ->
  // ... -> Predict Word end to end before hardware exists.
  resetCharacterBuffer();
  rows.forEach((r) => captureRow(r, {}));

  state.pendingGroundTruth = label;
  el('training-status').textContent =
    `Loaded ${rows.length} samples${rejected ? ` (${rejected} rejected)` : ''}` +
    `${label ? `  ·  Ground truth: ${label}` : ''}`;
  log(`Training sample loaded: ${rows.length} rows, ${rejected} rejected${label ? `, label=${label}` : ''}`);
}

// ---------------------------------------------------------------------
// Sensor source: MARKER (hardware/marker_bridge.py's local WS server)
// ---------------------------------------------------------------------
function connectMarker() {
  const url = el('marker-ws-url').value.trim();
  if (!url) { log('Enter the bridge WebSocket URL first.'); return; }
  if (state.markerWs) { state.markerWs.close(); }

  const ws = new WebSocket(url);
  state.markerWs = ws;

  ws.onopen = () => { setMarkerStatus(true); log(`Connected to marker bridge at ${url}`); };
  ws.onclose = () => { setMarkerStatus(false); log('Marker bridge disconnected'); state.markerWs = null; };
  ws.onerror = () => log('Marker bridge WebSocket error');
  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch (e) { return; }
    if (msg.type === 'sensor' && Array.isArray(msg.data)) {
      captureRow(msg.data, { flag: msg.flag });
    }
  };
}

function disconnectMarker() {
  if (state.markerWs) { state.markerWs.close(); state.markerWs = null; }
}

// ---------------------------------------------------------------------
// Source selector wiring
// ---------------------------------------------------------------------
function setSource(src) {
  state.source = src;
  document.querySelectorAll('.source-pill').forEach((p) => p.classList.toggle('active', p.dataset.src === src));
  document.querySelectorAll('.source-panel').forEach((p) => p.classList.remove('active'));
  el(`panel-${src}`).classList.add('active');
  const badge = el('badge-source');
  badge.className = `badge badge-${src}`;
  badge.textContent = src === 'demo' ? 'Demo' : src === 'training' ? 'Training Sample' : 'Marker';
}

document.querySelectorAll('input[name="source"]').forEach((input) => {
  input.addEventListener('change', () => setSource(input.value));
});

// ---------------------------------------------------------------------
// Model info (valid length band) -- fetched once, purely informational
// ---------------------------------------------------------------------
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model/info');
    if (!res.ok) return;
    const info = await res.json();
    if (info.min_raw_lines && info.max_raw_lines) {
      state.minRawLines = info.min_raw_lines;
      state.maxRawLines = info.max_raw_lines;
      el('meta-band').textContent = `${info.min_raw_lines}–${info.max_raw_lines}`;
    }
  } catch (e) {
    // Non-fatal -- the band display just stays at its default.
  }
}

// ---------------------------------------------------------------------
// Button wiring
// ---------------------------------------------------------------------
el('btn-start').onclick = () => send('start_session');
el('btn-stop').onclick = () => send('stop_session');
el('btn-reset').onclick = () => send('reset_session');
el('btn-predict').onclick = () => send('predict_character');
el('btn-commit').onclick = () => send('commit_word');

el('btn-clear-character').onclick = () => {
  // Client-side-only reset. There is no dedicated "clear just the
  // in-progress character" server message today (only reset_session,
  // which clears the whole session) -- this button clears the LOCAL
  // preview so you can discard a bad capture before predicting. If the
  // server's own buffer needs clearing too, use Predict Character (it
  // always clears the server buffer after predicting) or Reset.
  resetCharacterBuffer();
  log('Cleared local character preview (server-side buffer unaffected until next Predict Character or Reset)');
};

el('btn-clear-word').onclick = () => {
  // Same caveat as above: there's no dedicated "clear word, keep
  // session" server message, so this uses the existing reset_session
  // message, which also clears the in-progress character buffer.
  send('reset_session');
};

el('btn-sim-sample').onclick = () => captureRow(randomSensorRow(), {});
el('btn-sim-burst').onclick = () => {
  for (let i = 0; i < 40; i += 1) captureRow(randomSensorRow(), {});
};

el('btn-load-sample').onclick = loadTrainingSample;
el('btn-clear-training').onclick = () => {
  el('training-textarea').value = '';
  el('training-status').textContent = '';
  state.pendingGroundTruth = null;
};

el('btn-marker-connect').onclick = connectMarker;
el('btn-marker-disconnect').onclick = disconnectMarker;

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------
setSource('demo');
renderSparklines();
connect();
loadModelInfo();
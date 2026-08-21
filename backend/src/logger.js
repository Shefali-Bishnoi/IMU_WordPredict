/**
 * Minimal structured logger. Deliberately does not log raw sensor
 * arrays (per project logging guidance) -- callers should log counts/
 * lengths/latencies, never the arrays themselves.
 */
function ts() {
  return new Date().toISOString();
}

function fmt(level, msg) {
  return `[${ts()}] [${level}] ${msg}`;
}

export default {
  info: (msg) => console.log(fmt('INFO', msg)),
  warn: (msg) => console.warn(fmt('WARN', msg)),
  error: (msg) => console.error(fmt('ERROR', msg)),
  debug: (msg) => {
    if (process.env.DEBUG) console.log(fmt('DEBUG', msg));
  },
};
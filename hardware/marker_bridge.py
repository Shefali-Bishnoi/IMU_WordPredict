"""
hardware/marker_bridge.py

Pure hardware-to-browser adapter for the physical IMU marker. This script
does ONE job: read the marker's live serial stream, parse it into
9-value sensor rows using the SAME column layout the rest of this
project already uses (see config.py / preprocessing/io.py), and
broadcast those rows over a small local WebSocket server that the
frontend connects to directly.

WHY A SEPARATE LOCAL WEBSOCKET SERVER, NOT A DIRECT LINE INTO THE
NODE BACKEND'S SESSION STATE:

The Node backend's WordPredict session model (backend/session/
sessionManager.js, backend/websocket/wsServer.js) ties a session to the
specific browser WebSocket connection that created it -- there is no
existing mechanism for a second process (this bridge) to inject rows
into someone else's browser session. Rather than redesigning that
session/connection model (which every other client of the Node backend
already depends on), this bridge instead broadcasts rows on its OWN
tiny local WebSocket server. The frontend's MarkerWebSocketSource (see
frontend/app.js) connects to BOTH: the existing Node backend (for
session/predict/commit, unchanged) AND this bridge (purely to receive
raw sensor rows). When a row arrives from the bridge, the frontend
forwards it into the EXISTING 'sensor' WebSocket message the Node
backend already understands -- exactly the same call a Demo-source or
Training-Sample-source row makes. This is what keeps "one prediction
path, three interchangeable sources" true without touching the Node/
Python session contract at all.

    Marker  --serial-->  THIS BRIDGE  --local WS-->  Browser
                                                          |
                                                          | existing
                                                          | 'sensor' /
                                                          | 'predict_character'
                                                          | messages,
                                                          | UNCHANGED
                                                          v
                                                    Node backend
                                                          |
                                                          v
                                                  Python /stroke, /commit
                                                  (beam search, dictionary,
                                                   TCN model -- unchanged)

NO ML INFERENCE HAPPENS HERE. This script never imports tensorflow,
never calls the model, never talks to app/main.py directly.

RAW SCHEMA (reused, not re-derived): config.py already defines the exact
12-column, no-header raw schema this dataset/device uses:
    0: character label (not present/meaningful on a live device stream --
       see NOTE below)
    1: timestamp
    2-4: Accel X, Y, Z
    5-7: Gyro X, Y, Z
    8-10: Mag X, Y, Z
    11: writing flag (1 = actively writing, 0 = not)
This bridge reuses preprocessing/io.py's row-normalization helper
(_normalize_row) for the same 12/13-column quirks that module already
handles, instead of writing a second parser.

NOTE on live device format: the ORIGINAL recorded .txt files always
carry a real character label in column 0, because that's how the
dataset was collected (one file per known character). A LIVE marker
mid-stream has no way to know what character is being written -- that's
the whole point of this system. If your firmware's live serial output
omits column 0 (or sends a placeholder), set --label-col-present=false
and this script will insert a placeholder so the shared row-normalizer
still sees the expected column count. Check your actual firmware output
before assuming either way -- this is deliberately a CLI flag, not a
guess baked into the parser.

'ovf' / 'nan' HANDLING (documented limitation, read before relying on
this in production): the project's real cleaning logic
(preprocessing/clean.py's clean_sensor_matrix) does forward-fill THEN
back-fill across an entire already-complete stroke -- it can look both
backward and forward in time because by the time it runs, the whole
stroke already exists. This bridge is a streaming adapter -- it only
ever sees one row at a time and can't "look forward." So invalid
readings here are handled with LOCAL, causal forward-fill only (carry
the last valid value per channel; 0.0 if none has ever been seen yet),
logged as a warning. This is NOT identical to the server-side cleaning
and is called out explicitly rather than silently claimed equivalent --
a normal marker stream should essentially never hit this path (ovf/nan
are collection artifacts, not something well-functioning hardware emits
routinely), so this is a safety net, not a data-quality guarantee.

USAGE:
    pip install pyserial websockets
    python -m hardware.marker_bridge --serial-port COM5 --baud 115200
    python -m hardware.marker_bridge --serial-port /dev/ttyUSB0 --baud 115200 --ws-port 8765

Then, in the frontend, select "Marker" as the sensor source and it will
connect to ws://localhost:8765 by default (configurable in the UI).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FLAG_COL, LABEL_COL, NUM_RAW_COLS, SENSOR_COLS  # noqa: E402
from preprocessing.io import _normalize_row  # noqa: E402  (reused, not re-derived)

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    serial = None

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


# ---------------------------------------------------------------------------
# Transport abstraction -- Serial today, BLE is a clean drop-in later.
# Nothing downstream of MarkerTransport.read_lines() cares which one is
# in use.
# ---------------------------------------------------------------------------
class MarkerTransport:
    """Base interface: yields raw text lines from the marker device."""

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def read_line(self) -> Optional[str]:
        """Returns one decoded line (no trailing newline), or None on a
        transient read timeout (caller should just try again)."""
        raise NotImplementedError


class SerialMarkerTransport(MarkerTransport):
    def __init__(self, port: str, baud: int, timeout: float = 1.0):
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Run `pip install pyserial` first."
            )
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._conn: Optional["serial.Serial"] = None

    def open(self) -> None:
        self._conn = serial.Serial(self.port, self.baud, timeout=self.timeout)
        print(f"[bridge] opened serial port {self.port} @ {self.baud} baud")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def read_line(self) -> Optional[str]:
        if self._conn is None:
            raise RuntimeError("transport not open")
        raw = self._conn.readline()
        if not raw:
            return None  # timeout, no data this cycle
        return raw.decode("utf-8", errors="replace").strip()


class BLEMarkerTransport(MarkerTransport):
    """Not implemented -- extension point only. When BLE firmware support
    exists, implement open()/close()/read_line() here (e.g. via `bleak`)
    without touching the parser or the WebSocket broadcaster below; both
    already only depend on the MarkerTransport interface."""

    def __init__(self, device_address: str):
        self.device_address = device_address

    def open(self) -> None:
        raise NotImplementedError(
            "BLE transport is a documented extension point, not implemented yet. "
            "Use --transport serial until a BLE firmware format is confirmed."
        )

    def close(self) -> None:
        pass

    def read_line(self) -> Optional[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Parsing: reuse the project's own row-normalization, extract 9 channels.
# ---------------------------------------------------------------------------
class RowParser:
    """Causal (streaming, one-row-at-a-time) parser. Carries forward the
    last valid value per sensor channel for ovf/nan/unparsable readings
    -- see the module docstring's 'ovf/nan HANDLING' note for why this is
    NOT the same as the server-side ffill+bfill cleaning, and why that's
    an acceptable, documented trade-off for a live streaming adapter."""

    def __init__(self, label_col_present: bool = True):
        self.label_col_present = label_col_present
        self._last_valid = [0.0] * 9  # carried-forward per-channel value
        self.n_parsed = 0
        self.n_rejected = 0
        self.n_filled = 0

    def parse_line(self, line: str) -> Optional[dict]:
        """Returns {"sensor": [9 floats], "flag": 0/1/None} or None if
        the line couldn't be salvaged into a valid row at all (wrong
        shape after normalization, empty line, etc.) -- these are
        logged and skipped, never forwarded downstream."""
        line = line.strip()
        if not line:
            return None

        parts = line.split(",")
        if not self.label_col_present:
            # Live firmware with no label column -- insert a placeholder
            # so the shared _normalize_row() still sees the schema it
            # expects (label, timestamp, 9 sensor values, flag).
            parts = ["?"] + parts

        try:
            parts = _normalize_row(parts, filepath=Path("<serial>"), line_no=self.n_parsed + 1)
        except ValueError as e:
            self.n_rejected += 1
            print(f"[bridge] rejected malformed line ({e}): {line!r}")
            return None

        if len(parts) != NUM_RAW_COLS:
            self.n_rejected += 1
            print(f"[bridge] rejected line with unexpected column count "
                  f"({len(parts)} != {NUM_RAW_COLS}): {line!r}")
            return None

        sensor_fields = parts[SENSOR_COLS]
        row = list(self._last_valid)  # start from last-known-good, overwrite below
        any_filled = False
        for i, raw_val in enumerate(sensor_fields):
            val = self._to_float(raw_val)
            if val is None:
                any_filled = True
                # row[i] already carries the last valid value (causal fill)
            else:
                row[i] = val
                self._last_valid[i] = val

        if any_filled:
            self.n_filled += 1

        flag_val = None
        try:
            flag_val = int(float(parts[FLAG_COL]))
        except (ValueError, IndexError):
            pass

        self.n_parsed += 1
        return {"sensor": row, "flag": flag_val}

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in ("ovf", "nan"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        if f != f:  # NaN check without importing math/numpy here
            return None
        return f


# ---------------------------------------------------------------------------
# WebSocket broadcaster -- the ONLY thing the frontend talks to for
# marker data. One-to-many broadcast (usually just one browser tab).
# ---------------------------------------------------------------------------
class BridgeServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: "asyncio.Queue" = None  # created inside the loop

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        print(f"[bridge] browser connected ({len(self._clients)} client(s))")
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)
            print(f"[bridge] browser disconnected ({len(self._clients)} client(s))")

    async def _broadcast_loop(self) -> None:
        while True:
            payload = await self._queue.get()
            if not self._clients:
                continue
            message = json.dumps(payload)
            dead = []
            for ws in self._clients:
                try:
                    await ws.send(message)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def push_from_thread(self, payload: dict) -> None:
        """Called from the serial-reading background thread -- hands the
        payload to the asyncio loop thread-safely."""
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)

    async def run(self) -> None:
        if websockets is None:
            raise RuntimeError(
                "websockets is not installed. Run `pip install websockets` first."
            )
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[bridge] local WebSocket server listening on "
                  f"ws://{self.host}:{self.port} -- point the frontend's "
                  f"'Marker' sensor source at this URL")
            await self._broadcast_loop()


# ---------------------------------------------------------------------------
# Serial reader thread -- blocking pyserial reads happen off the asyncio
# loop, forwarding parsed rows into the broadcaster's queue.
# ---------------------------------------------------------------------------
def _serial_reader_thread(
    transport: MarkerTransport,
    parser: RowParser,
    server: BridgeServer,
    stop_event: threading.Event,
) -> None:
    transport.open()
    try:
        while not stop_event.is_set():
            try:
                line = transport.read_line()
            except Exception as e:  # noqa: BLE001
                print(f"[bridge] serial read error: {e} -- retrying in 1s")
                time.sleep(1.0)
                continue
            if line is None:
                continue  # read timeout, no data this cycle
            parsed = parser.parse_line(line)
            if parsed is None:
                continue
            server.push_from_thread({
                "type": "sensor",
                "data": parsed["sensor"],
                "flag": parsed["flag"],          # optional metadata, UI display only
                "source": "marker",
                "timestamp": time.time(),
            })
    finally:
        transport.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["serial", "ble"], default="serial")
    parser.add_argument("--serial-port", type=str, help="e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--ble-address", type=str, help="BLE device address (not implemented yet)")
    parser.add_argument("--ws-host", type=str, default="localhost")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument(
        "--label-col-present", type=str, default="false",
        choices=["true", "false"],
        help="Does the live serial line include a character-label column "
             "(like the recorded .txt files do)? Most live firmware "
             "streams will NOT have a meaningful label (the device "
             "doesn't know what character is being written) -- default "
             "false. Set true only if your firmware genuinely echoes one.",
    )
    args = parser.parse_args()

    if args.transport == "serial":
        if not args.serial_port:
            parser.error("--serial-port is required when --transport=serial")
        transport = SerialMarkerTransport(args.serial_port, args.baud)
    else:
        if not args.ble_address:
            parser.error("--ble-address is required when --transport=ble")
        transport = BLEMarkerTransport(args.ble_address)

    row_parser = RowParser(label_col_present=(args.label_col_present == "true"))
    server = BridgeServer(args.ws_host, args.ws_port)

    stop_event = threading.Event()
    reader = threading.Thread(
        target=_serial_reader_thread,
        args=(transport, row_parser, server, stop_event),
        daemon=True,
    )
    reader.start()

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print(
            f"\n[bridge] shutting down. rows parsed={row_parser.n_parsed} "
            f"rejected={row_parser.n_rejected} "
            f"channel-fills-used={row_parser.n_filled}"
        )


if __name__ == "__main__":
    main()
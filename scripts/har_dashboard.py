#!/usr/bin/env python3
"""Local browser dashboard for Grove Vision AI V2 HAR predictions.

The board continues to send its prediction text over USB serial. This program
owns that serial connection, parses activity/confidence values, and presents
them through a local web GUI instead of a serial terminal.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


ACTIVITY_NAMES = {
    "0": "Walking",
    "1": "Walking Upstairs",
    "2": "Walking Downstairs",
    "3": "Sitting",
    "4": "Standing",
    "5": "Laying",
    "walking": "Walking",
    "walking upstairs": "Walking Upstairs",
    "walking upstairs": "Walking Upstairs",
    "walking downstairs": "Walking Downstairs",
    "walking downstairs": "Walking Downstairs",
    "sitting": "Sitting",
    "standing": "Standing",
    "laying": "Laying",
    "lying": "Laying",
}

PREDICTION_RE = re.compile(
    r"(?:prediction|activity|result|class)\s*[:=]\s*"
    r"(?P<activity>[A-Za-z_ -]+|[0-5])",
    re.IGNORECASE,
)
CONFIDENCE_RE = re.compile(
    r"(?:confidence|conf|probability|score)\s*[:=]?\s*"
    r"(?P<confidence>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<percent>%)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Prediction:
    timestamp: float
    activity: str
    confidence: Optional[float]


class DashboardState:
    def __init__(self, serial_port: str) -> None:
        self._lock = threading.Lock()
        self.serial_port = serial_port
        self.connected = False
        self.status = "Waiting for serial device"
        self.last_line = ""
        self.last_prediction: Optional[Prediction] = None
        self.history: deque[Prediction] = deque(maxlen=60)
        self.revision = 0

    def set_connection(self, connected: bool, status: str) -> None:
        with self._lock:
            if self.connected != connected or self.status != status:
                self.connected = connected
                self.status = status
                self.revision += 1

    def set_last_line(self, line: str) -> None:
        with self._lock:
            self.last_line = line[-300:]

    def add_prediction(self, prediction: Prediction) -> None:
        with self._lock:
            self.last_prediction = prediction
            self.history.append(prediction)
            self.revision += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "serial_port": self.serial_port,
                "connected": self.connected,
                "status": self.status,
                "last_prediction": (
                    asdict(self.last_prediction) if self.last_prediction else None
                ),
                "history": [asdict(item) for item in self.history],
                "revision": self.revision,
                "server_time": time.time(),
            }


def canonical_activity(raw: str) -> Optional[str]:
    cleaned = re.sub(r"[_-]+", " ", raw).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in ACTIVITY_NAMES:
        return ACTIVITY_NAMES[cleaned]

    # Some firmware messages put confidence text directly after the label.
    for key, display_name in ACTIVITY_NAMES.items():
        if not key.isdigit() and cleaned.startswith(key):
            return display_name
    return None


def parse_prediction(line: str) -> Optional[Prediction]:
    activity_match = PREDICTION_RE.search(line)
    if not activity_match:
        return None

    activity = canonical_activity(activity_match.group("activity"))
    if activity is None:
        return None

    confidence: Optional[float] = None
    confidence_match = CONFIDENCE_RE.search(line)
    if confidence_match:
        value = float(confidence_match.group("confidence"))
        if confidence_match.group("percent") or value > 1.0:
            value /= 100.0
        confidence = max(0.0, min(1.0, value))

    return Prediction(time.time(), activity, confidence)


def serial_worker(
    state: DashboardState,
    serial_port: str,
    baudrate: int,
    stop_event: threading.Event,
) -> None:
    try:
        import serial  # type: ignore
    except ImportError:
        state.set_connection(
            False,
            "PySerial is missing; install it in the WSL virtual environment",
        )
        return

    while not stop_event.is_set():
        connection = None
        try:
            state.set_connection(False, f"Connecting to {serial_port}")
            connection = serial.Serial(serial_port, baudrate, timeout=1.0)
            state.set_connection(True, f"Connected to {serial_port}")

            while not stop_event.is_set():
                raw = connection.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                state.set_last_line(line)
                prediction = parse_prediction(line)
                if prediction:
                    state.add_prediction(prediction)

        except Exception as exc:  # SerialException varies between platforms.
            state.set_connection(False, f"Serial unavailable: {exc}")
            stop_event.wait(2.0)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass


def demo_worker(state: DashboardState, stop_event: threading.Event) -> None:
    activities = list(dict.fromkeys(ACTIVITY_NAMES.values()))
    index = 0
    phase = 0.0
    state.set_connection(True, "Demo data enabled")

    while not stop_event.is_set():
        if random.random() < 0.18:
            index = (index + random.choice((1, 2, 3))) % len(activities)
        phase += 0.7
        confidence = 0.78 + 0.14 * math.sin(phase) + random.uniform(-0.035, 0.035)
        confidence = max(0.52, min(0.99, confidence))
        state.add_prediction(Prediction(time.time(), activities[index], confidence))
        stop_event.wait(1.28)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Activity Recognition</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: rgba(17, 31, 51, 0.90);
      --panel-2: rgba(12, 24, 42, 0.94);
      --text: #f4f8ff;
      --muted: #91a3bb;
      --accent: #48d7b0;
      --accent-soft: rgba(72, 215, 176, 0.14);
      --danger: #ff718d;
      --grid: rgba(148, 163, 184, 0.16);
      --border: rgba(148, 163, 184, 0.18);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 8%, rgba(36, 99, 235, .20), transparent 34rem),
        radial-gradient(circle at 90% 86%, rgba(16, 185, 129, .14), transparent 30rem),
        var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif;
    }

    main { width: min(1180px, calc(100% - 36px)); margin: 0 auto; padding: 30px 0 42px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 24px; }
    h1 { margin: 0; font-size: clamp(1.25rem, 2.5vw, 1.75rem); letter-spacing: -.02em; }
    .eyebrow { color: var(--muted); text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 800; margin-bottom: 7px; }
    .status { display: inline-flex; align-items: center; gap: 9px; color: var(--muted); font-size: .86rem; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--danger); box-shadow: 0 0 0 5px rgba(255, 113, 141, .10); }
    .dot.online { background: var(--accent); box-shadow: 0 0 0 5px var(--accent-soft); }

    .grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(300px, .8fr); gap: 18px; }
    .card { border: 1px solid var(--border); border-radius: 22px; background: linear-gradient(145deg, var(--panel), var(--panel-2)); box-shadow: 0 24px 80px rgba(0,0,0,.24); }
    .hero { min-height: 350px; padding: clamp(24px, 5vw, 48px); display: flex; flex-direction: column; justify-content: space-between; overflow: hidden; position: relative; }
    .hero::after { content: ""; position: absolute; width: 270px; height: 270px; border-radius: 50%; right: -90px; bottom: -110px; background: var(--accent-soft); filter: blur(2px); }
    .activity-row { display: flex; align-items: center; gap: 22px; position: relative; z-index: 1; }
    .activity-icon { display: grid; place-items: center; width: 84px; height: 84px; flex: 0 0 auto; border-radius: 24px; background: var(--accent-soft); font-size: 2.7rem; }
    #activity { margin: 0; font-size: clamp(2.6rem, 7vw, 5.7rem); line-height: .98; letter-spacing: -.055em; text-wrap: balance; }
    .confidence-label { color: var(--muted); font-size: .82rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
    #confidence { margin-top: 5px; color: var(--accent); font-size: clamp(1.15rem, 2.3vw, 1.7rem); font-weight: 750; }
    .meter { width: 100%; height: 9px; margin-top: 12px; border-radius: 99px; background: rgba(148,163,184,.16); overflow: hidden; }
    .meter > div { height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, #2bb7ff, var(--accent)); transition: width .45s ease; }

    .side { padding: 24px; display: flex; flex-direction: column; gap: 20px; }
    .side h2, .chart-card h2 { margin: 0; font-size: 1rem; }
    .metric { padding: 16px; border-radius: 16px; background: rgba(148,163,184,.07); }
    .metric span { display: block; color: var(--muted); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
    .metric strong { display: block; margin-top: 5px; font-size: 1rem; overflow-wrap: anywhere; }
    .class-list { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .class-pill { padding: 9px 10px; border: 1px solid var(--border); border-radius: 12px; color: var(--muted); font-size: .78rem; }
    .class-pill.active { color: var(--text); border-color: var(--accent); background: var(--accent-soft); }

    .chart-card { grid-column: 1 / -1; padding: 24px; }
    .chart-head { display: flex; justify-content: space-between; align-items: end; gap: 18px; margin-bottom: 16px; }
    .chart-head p { margin: 4px 0 0; color: var(--muted); font-size: .84rem; }
    .chart-wrap { position: relative; height: 260px; width: 100%; }
    canvas { width: 100%; height: 100%; display: block; }
    #last-update { color: var(--muted); font-size: .78rem; white-space: nowrap; }

    @media (max-width: 780px) {
      main { width: min(100% - 24px, 680px); padding-top: 20px; }
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .hero { min-height: 310px; }
      .activity-row { align-items: flex-start; flex-direction: column; }
      .chart-card { grid-column: auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">Grove Vision AI V2 · Ethos-U55</div>
        <h1>Human Activity Recognition</h1>
      </div>
      <div class="status"><span id="status-dot" class="dot"></span><span id="status-text">Connecting…</span></div>
    </header>

    <section class="grid">
      <article class="card hero" aria-live="polite">
        <div>
          <div class="eyebrow">Current activity</div>
          <div class="activity-row">
            <div id="activity-icon" class="activity-icon" aria-hidden="true">⋯</div>
            <h2 id="activity">Waiting…</h2>
          </div>
        </div>
        <div>
          <div class="confidence-label">Model confidence</div>
          <div id="confidence">—</div>
          <div class="meter" role="progressbar" aria-label="Prediction confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div id="confidence-bar"></div>
          </div>
        </div>
      </article>

      <aside class="card side">
        <h2>System</h2>
        <div class="metric"><span>Serial device</span><strong id="serial-port">/dev/ttyACM0</strong></div>
        <div class="metric"><span>Input window</span><strong>128 × 9 · 50 Hz</strong></div>
        <div class="metric"><span>Inference stride</span><strong>64 samples · 1.28 s</strong></div>
        <div>
          <div class="eyebrow">Activity classes</div>
          <div id="class-list" class="class-list"></div>
        </div>
      </aside>

      <article class="card chart-card">
        <div class="chart-head">
          <div><h2>Confidence trend</h2><p>Most recent 60 predictions</p></div>
          <div id="last-update">No prediction received</div>
        </div>
        <div class="chart-wrap"><canvas id="chart" aria-label="Confidence history line graph"></canvas></div>
      </article>
    </section>
  </main>

  <script>
    const classes = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"];
    const icons = {"Walking":"🚶", "Walking Upstairs":"↗", "Walking Downstairs":"↘", "Sitting":"🪑", "Standing":"🧍", "Laying":"▬"};
    let snapshot = {history: [], last_prediction: null};

    const classList = document.getElementById("class-list");
    classes.forEach(name => {
      const item = document.createElement("div");
      item.className = "class-pill";
      item.dataset.activity = name;
      item.textContent = name;
      classList.appendChild(item);
    });

    function update(data) {
      snapshot = data;
      document.getElementById("serial-port").textContent = data.serial_port;
      document.getElementById("status-text").textContent = data.status;
      document.getElementById("status-dot").classList.toggle("online", data.connected);

      const prediction = data.last_prediction;
      if (prediction) {
        document.getElementById("activity").textContent = prediction.activity;
        document.getElementById("activity-icon").textContent = icons[prediction.activity] || "●";
        const value = prediction.confidence;
        const percent = value == null ? null : Math.round(value * 1000) / 10;
        document.getElementById("confidence").textContent = percent == null ? "Not reported" : `${percent.toFixed(1)}%`;
        document.getElementById("confidence-bar").style.width = percent == null ? "0%" : `${percent}%`;
        const meter = document.querySelector(".meter");
        meter.setAttribute("aria-valuenow", percent == null ? "0" : String(percent));
        document.getElementById("last-update").textContent = `Updated ${new Date(prediction.timestamp * 1000).toLocaleTimeString()}`;
        document.querySelectorAll(".class-pill").forEach(item => item.classList.toggle("active", item.dataset.activity === prediction.activity));
      }
      drawChart();
    }

    function drawChart() {
      const canvas = document.getElementById("chart");
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);

      const width = rect.width, height = rect.height;
      const pad = {left: 42, right: 14, top: 14, bottom: 26};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      ctx.clearRect(0, 0, width, height);
      ctx.font = "12px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";

      [0, 25, 50, 75, 100].forEach(value => {
        const y = pad.top + plotH * (1 - value / 100);
        ctx.strokeStyle = "rgba(148, 163, 184, .16)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
        ctx.fillStyle = "#91a3bb"; ctx.fillText(`${value}%`, pad.left - 8, y);
      });

      const points = (snapshot.history || []).filter(p => p.confidence != null);
      if (!points.length) {
        ctx.fillStyle = "#91a3bb";
        ctx.textAlign = "center";
        ctx.fillText("Confidence data will appear after the first prediction", pad.left + plotW / 2, pad.top + plotH / 2);
        return;
      }

      const xFor = i => pad.left + (points.length === 1 ? plotW : i * plotW / (points.length - 1));
      const yFor = p => pad.top + plotH * (1 - p.confidence);

      const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
      gradient.addColorStop(0, "rgba(72, 215, 176, .28)");
      gradient.addColorStop(1, "rgba(72, 215, 176, 0)");
      ctx.beginPath();
      points.forEach((point, i) => i ? ctx.lineTo(xFor(i), yFor(point)) : ctx.moveTo(xFor(i), yFor(point)));
      ctx.lineTo(xFor(points.length - 1), pad.top + plotH);
      ctx.lineTo(xFor(0), pad.top + plotH);
      ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();

      ctx.beginPath();
      points.forEach((point, i) => i ? ctx.lineTo(xFor(i), yFor(point)) : ctx.moveTo(xFor(i), yFor(point)));
      ctx.strokeStyle = "#48d7b0"; ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();

      const last = points[points.length - 1];
      ctx.beginPath(); ctx.arc(xFor(points.length - 1), yFor(last), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#f4f8ff"; ctx.fill();
    }

    const events = new EventSource("/events");
    events.onmessage = event => update(JSON.parse(event.data));
    events.onerror = () => {
      document.getElementById("status-text").textContent = "Dashboard connection interrupted";
      document.getElementById("status-dot").classList.remove("online");
    };
    new ResizeObserver(drawChart).observe(document.querySelector(".chart-wrap"));
  </script>
</body>
</html>
"""


def make_handler(state: DashboardState, stop_event: threading.Event):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "HARDashboard/1.0"

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/state":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/events":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while not stop_event.is_set():
                        payload = json.dumps(state.snapshot())
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        stop_event.wait(0.5)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args) -> None:
            # Keep the terminal focused on device/dashboard status.
            return

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Grove Vision AI V2 HAR predictions in a local web GUI."
    )
    parser.add_argument("--serial-port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate sample predictions without connecting to hardware.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    state = DashboardState("Demo mode" if args.demo else args.serial_port)

    worker_target = demo_worker if args.demo else serial_worker
    worker_args = (
        (state, stop_event)
        if args.demo
        else (state, args.serial_port, args.baudrate, stop_event)
    )
    worker = threading.Thread(target=worker_target, args=worker_args, daemon=True)
    worker.start()

    server = ThreadingHTTPServer(
        (args.host, args.http_port), make_handler(state, stop_event)
    )

    def request_shutdown(signum, frame) -> None:  # noqa: ARG001
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    print("HAR dashboard is running")
    print(f"Open: http://{args.host}:{args.http_port}")
    if args.demo:
        print("Input: demo predictions")
    else:
        print(f"Input: {args.serial_port} at {args.baudrate} baud")
        print("Close miniterm first; only one program can own the serial port.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop_event.set()
        server.server_close()
        worker.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

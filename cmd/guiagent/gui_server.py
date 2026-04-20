#!/usr/bin/env python3
"""
gui_server.py — Web UI for the GUI automation agent on port 5005.

Provides:
  GET  /            — browser UI (screenshot + live log + task input)
  POST /start       — {"task": "...", "max_iterations": 30}
  GET  /stream      — SSE event stream (screenshot, thought, action, result, done)
  GET  /status      — current agent state
  POST /stop        — request current run to stop (best-effort)

Run standalone:
  .venv/bin/python cmd/guiagent/gui_server.py [--port 5005] [--display :99]
"""

import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add cmd subpackages to path (same logic as server.py)
_HERE = Path(__file__).resolve().parent
_CMD  = _HERE.parent
for _sub in (_CMD, _CMD / "core", _CMD / "chain", _CMD / "blueteam",
             _CMD / "infra", _CMD / "guiagent"):
    s = str(_sub)
    if s not in sys.path:
        sys.path.insert(0, s)

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from gui_agent import GUIAgent

app  = Flask(__name__)
CORS(app)

# ── State ─────────────────────────────────────────────────────────────────────

_lock          = threading.Lock()
_clients: list = []          # list of queue.Queue — one per SSE client
_agent_thread  = None
_agent_state   = {
    "status":     "idle",    # idle | running | done | error
    "task":       "",
    "started_at": None,
    "finished_at": None,
    "success":    None,
    "summary":    "",
    "iterations": 0,
    "last_screenshot": None,  # base64 PNG string
}
_stop_requested = threading.Event()


# ── SSE broadcast ─────────────────────────────────────────────────────────────

def _broadcast(event_type: str, payload: dict):
    """Push an event to all connected SSE clients."""
    msg = json.dumps({"type": event_type, **payload})
    if event_type == "screenshot":
        # Keep last screenshot for late-joining clients
        _agent_state["last_screenshot"] = payload.get("image")
    dead = []
    with _lock:
        clients = list(_clients)
    for q in clients:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    if dead:
        with _lock:
            for q in dead:
                if q in _clients:
                    _clients.remove(q)


def _sse_format(data: str) -> str:
    return f"data: {data}\n\n"


# ── Agent runner ──────────────────────────────────────────────────────────────

def _run_agent(task: str, max_iterations: int, display: str):
    global _agent_state
    _agent_state.update({
        "status": "running",
        "task": task,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "success": None,
        "summary": "",
        "iterations": 0,
        "last_screenshot": None,
    })
    _broadcast("log", {"text": f"Starting task: {task}", "level": "info"})

    try:
        _stop_requested.clear()  # reset from any previous stop
        agent = GUIAgent(display=display, event_cb=_broadcast, stop_event=_stop_requested)
        result = agent.run(task, max_iterations=max_iterations)
        success = result.get("success", False)
        summary = result.get("summary", "")
        _agent_state.update({
            "status": "done",
            "success": success,
            "summary": summary,
            "iterations": len(agent.agent.react_trace),
            "finished_at": datetime.now().isoformat(),
        })
    except Exception as e:
        _agent_state.update({
            "status": "error",
            "success": False,
            "summary": str(e),
            "finished_at": datetime.now().isoformat(),
        })
        _broadcast("error", {"text": str(e)})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_UI, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/start", methods=["POST"])
def start():
    global _agent_thread
    data = request.get_json(silent=True) or {}
    task = (data.get("task") or "").strip()
    if not task:
        return jsonify({"error": "task is required"}), 400
    max_iter = int(data.get("max_iterations", 30))
    display  = data.get("display", ":99")

    with _lock:
        if _agent_state["status"] == "running":
            return jsonify({"error": "agent already running"}), 409

    _agent_thread = threading.Thread(
        target=_run_agent,
        args=(task, max_iter, display),
        daemon=True,
    )
    _agent_thread.start()
    return jsonify({"status": "started", "task": task})


@app.route("/stop", methods=["POST"])
def stop():
    _stop_requested.set()
    _broadcast("log", {"text": "Stop requested — will finish current action", "level": "warn"})
    return jsonify({"status": "stop_requested"})


@app.route("/status")
def status():
    s = dict(_agent_state)
    s.pop("last_screenshot", None)  # don't include image in status JSON
    return jsonify(s)


@app.route("/screenshot")
def screenshot():
    img = _agent_state.get("last_screenshot")
    if not img:
        return "", 204
    import base64
    data = base64.b64decode(img)
    return Response(data, mimetype="image/png")


@app.route("/stream")
def stream():
    """SSE endpoint — streams agent events to the browser."""
    client_q = queue.Queue(maxsize=200)
    with _lock:
        _clients.append(client_q)

    # Immediately send current state to new client
    init_events = [json.dumps({"type": "init", "state": {
        "status": _agent_state["status"],
        "task":   _agent_state["task"],
    }})]
    if _agent_state.get("last_screenshot"):
        init_events.append(json.dumps({
            "type": "screenshot",
            "image": _agent_state["last_screenshot"],
        }))

    @stream_with_context
    def generate():
        try:
            for ev in init_events:
                yield _sse_format(ev)
            # Heartbeat every 15s, events as they arrive
            while True:
                try:
                    msg = client_q.get(timeout=15)
                    yield _sse_format(msg)
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _lock:
                if client_q in _clients:
                    _clients.remove(client_q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GUI Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0f0f0f;
    --surface:  #1a1a1a;
    --border:   #2a2a2a;
    --text:     #e0e0e0;
    --dim:      #888;
    --accent:   #4a9eff;
    --thought:  #6ba3ff;
    --action:   #ffad33;
    --result:   #4caf7d;
    --vision:   #7ecac8;
    --done:     #b06bff;
    --error:    #ff5555;
    --warn:     #ffcc44;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 18px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  header h1 {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--accent);
  }
  .badge {
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .badge.idle    { background: #2a2a2a; color: var(--dim); }
  .badge.running { background: #1a3a5c; color: var(--accent); animation: pulse 1.4s ease-in-out infinite; }
  .badge.done    { background: #1a3a28; color: var(--result); }
  .badge.error   { background: #3a1a1a; color: var(--error); }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.55 } }

  .task-label {
    font-size: 0.78rem;
    color: var(--dim);
    max-width: 500px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ── Main layout ── */
  main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Screenshot panel ── */
  #screen-panel {
    flex: 1.3;
    background: #080808;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    border-right: 1px solid var(--border);
  }
  #screen-panel img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }
  #no-screen {
    color: var(--dim);
    font-size: 0.85rem;
    text-align: center;
    line-height: 1.8;
  }
  #screen-ts {
    position: absolute;
    bottom: 8px;
    right: 10px;
    font-size: 0.68rem;
    color: var(--dim);
    background: rgba(0,0,0,.5);
    padding: 2px 6px;
    border-radius: 4px;
  }

  /* ── Right panel ── */
  #right-panel {
    width: 420px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Task form ── */
  #task-form {
    padding: 14px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  #task-form textarea {
    width: 100%;
    height: 72px;
    resize: none;
    background: #111;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-family: inherit;
    font-size: 0.85rem;
    outline: none;
    transition: border-color .2s;
  }
  #task-form textarea:focus { border-color: var(--accent); }
  .form-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    align-items: center;
  }
  .form-row label {
    font-size: 0.78rem;
    color: var(--dim);
    white-space: nowrap;
  }
  .form-row input[type=number] {
    width: 60px;
    background: #111;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.82rem;
    outline: none;
  }
  #start-btn {
    margin-left: auto;
    padding: 6px 20px;
    border-radius: 6px;
    border: none;
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    font-size: 0.82rem;
    cursor: pointer;
    transition: opacity .15s;
  }
  #start-btn:hover:not(:disabled) { opacity: .85; }
  #start-btn:disabled { opacity: .4; cursor: default; }
  #stop-btn {
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--error);
    background: transparent;
    color: var(--error);
    font-size: 0.82rem;
    cursor: pointer;
    display: none;
  }
  #stop-btn:hover { background: rgba(255,85,85,.1); }

  /* ── Log ── */
  #log-header {
    padding: 6px 14px;
    font-size: 0.72rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  #clear-btn {
    background: none;
    border: none;
    color: var(--dim);
    cursor: pointer;
    font-size: 0.72rem;
    padding: 0;
  }
  #clear-btn:hover { color: var(--text); }

  #log {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
    scroll-behavior: smooth;
  }
  #log::-webkit-scrollbar { width: 4px; }
  #log::-webkit-scrollbar-track { background: transparent; }
  #log::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

  .log-entry {
    display: flex;
    gap: 8px;
    padding: 5px 14px;
    font-size: 0.8rem;
    line-height: 1.45;
    border-left: 2px solid transparent;
  }
  .log-entry:hover { background: rgba(255,255,255,.03); }

  .log-entry .icon { flex-shrink: 0; width: 14px; font-size: 0.9rem; line-height: 1.45; }
  .log-entry .body { flex: 1; min-width: 0; word-break: break-word; }
  .log-entry .iter {
    flex-shrink: 0;
    font-size: 0.7rem;
    color: var(--dim);
    padding-top: 2px;
  }

  .entry-thought  { border-color: var(--thought);  color: #c5d8ff; }
  .entry-action   { border-color: var(--action);   color: var(--action); }
  .entry-result   { border-color: var(--result);   color: #a8dfbe; }
  .entry-vision   { border-color: var(--vision);   color: #b8e8e7; }
  .entry-done     { border-color: var(--done);     color: var(--done); font-weight: 600; }
  .entry-error    { border-color: var(--error);    color: var(--error); }
  .entry-info     { border-color: #333;            color: var(--dim); }
  .entry-warn     { border-color: var(--warn);     color: var(--warn); }
  .entry-screenshot { border-color: #333; color: var(--dim); font-style: italic; }

  .entry-action .tool-name {
    font-family: 'Courier New', monospace;
    background: rgba(255,173,51,.12);
    padding: 0 4px;
    border-radius: 3px;
    font-size: 0.78rem;
  }
  .entry-action .args-json {
    font-family: 'Courier New', monospace;
    font-size: 0.75rem;
    color: #ccc;
    opacity: 0.8;
  }

  .result-ok   { color: var(--result); }
  .result-fail { color: var(--error); }
</style>
</head>
<body>

<header>
  <h1>⬜ GUI Agent</h1>
  <span id="status-badge" class="badge idle">Idle</span>
  <span id="task-label" class="task-label"></span>
</header>

<main>
  <div id="screen-panel">
    <div id="no-screen">No screenshot yet.<br>Submit a task to start.</div>
    <img id="screen-img" style="display:none" alt="agent screen">
    <span id="screen-ts"></span>
  </div>

  <div id="right-panel">
    <div id="task-form">
      <textarea id="task-input" placeholder="Describe what the agent should do on the desktop…"></textarea>
      <div class="form-row">
        <label for="iter-input">Max iterations:</label>
        <input type="number" id="iter-input" value="30" min="5" max="200">
        <button id="stop-btn" onclick="stopAgent()">Stop</button>
        <button id="start-btn" onclick="startAgent()">Start</button>
      </div>
    </div>

    <div id="log-header">
      <span>Live log</span>
      <button id="clear-btn" onclick="clearLog()">clear</button>
    </div>
    <div id="log"></div>
  </div>
</main>

<script>
const log      = document.getElementById('log');
const badge    = document.getElementById('status-badge');
const taskLbl  = document.getElementById('task-label');
const startBtn = document.getElementById('start-btn');
const stopBtn  = document.getElementById('stop-btn');
const screenImg = document.getElementById('screen-img');
const noScreen  = document.getElementById('no-screen');
const screenTs  = document.getElementById('screen-ts');

let evtSource = null;
let autoScroll = true;

log.addEventListener('scroll', () => {
  autoScroll = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
});

// ── SSE connection ─────────────────────────────────────────────────────────

function connect() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/stream');
  evtSource.onmessage = e => handleEvent(JSON.parse(e.data));
  evtSource.onerror   = () => setTimeout(connect, 3000);
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'init':
      setStatus(ev.state.status);
      if (ev.state.task) taskLbl.textContent = ev.state.task;
      break;

    case 'screenshot':
      showScreenshot(ev.image);
      addEntry('screenshot', '📷', 'Screenshot taken', null, null);
      break;

    case 'thought':
      addEntry('thought', '💭',
        `<strong>[${ev.iteration}]</strong> ${esc(ev.thought)}`,
        ev.iteration,
        ev.confidence != null ? `${ev.confidence}%` : null);
      break;

    case 'action':
      const argsStr = JSON.stringify(ev.args || {});
      addEntry('action', '▶',
        `<span class="tool-name">${esc(ev.tool)}</span> ` +
        `<span class="args-json">${esc(argsStr)}</span>`);
      break;

    case 'result':
      const ok = ev.success;
      const txt = ev.output || ev.error || '';
      addEntry('result', ok ? '✓' : '✗',
        `<span class="${ok ? 'result-ok' : 'result-fail'}">${esc(txt.slice(0,200))}</span>`);
      break;

    case 'vision':
      addEntry('vision', '👁', esc(ev.text.slice(0, 400)));
      break;

    case 'done':
      const icon = ev.success ? '✅' : '⚠️';
      addEntry('done', icon,
        `${ev.success ? 'Done' : 'Finished'}: ${esc(ev.summary || '')} — ${ev.iterations} iterations`);
      setStatus(ev.success ? 'done' : 'error');
      setRunning(false);
      break;

    case 'error':
      addEntry('error', '✗', esc(ev.text || ''));
      setStatus('error');
      setRunning(false);
      break;

    case 'log':
      addEntry(ev.level || 'info',
        ev.level === 'warn' ? '⚠' : ev.level === 'error' ? '✗' : 'ℹ',
        esc(ev.text || ''));
      break;
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────

function showScreenshot(b64) {
  screenImg.src = 'data:image/png;base64,' + b64;
  screenImg.style.display = 'block';
  noScreen.style.display  = 'none';
  const now = new Date();
  screenTs.textContent = now.toLocaleTimeString();
}

function addEntry(type, icon, html, iter, meta) {
  const el = document.createElement('div');
  el.className = `log-entry entry-${type}`;
  el.innerHTML =
    `<span class="icon">${icon}</span>` +
    `<span class="body">${html}</span>` +
    (meta   ? `<span class="iter">${meta}</span>` : '') ;
  log.appendChild(el);
  if (autoScroll) log.scrollTop = log.scrollHeight;
}

function setStatus(st) {
  badge.textContent = st.charAt(0).toUpperCase() + st.slice(1);
  badge.className   = `badge ${st}`;
}

function setRunning(running) {
  startBtn.disabled = running;
  stopBtn.style.display = running ? 'inline-block' : 'none';
  document.getElementById('task-input').disabled = running;
}

function clearLog() { log.innerHTML = ''; }

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Actions ────────────────────────────────────────────────────────────────

async function startAgent() {
  const task = document.getElementById('task-input').value.trim();
  if (!task) return;
  const maxIter = parseInt(document.getElementById('iter-input').value) || 30;
  clearLog();
  setStatus('running');
  setRunning(true);
  taskLbl.textContent = task;

  try {
    const r = await fetch('/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task, max_iterations: maxIter}),
    });
    const d = await r.json();
    if (!r.ok) {
      addEntry('error', '✗', esc(d.error || 'Failed to start'));
      setStatus('error');
      setRunning(false);
    }
  } catch (e) {
    addEntry('error', '✗', esc(String(e)));
    setStatus('error');
    setRunning(false);
  }
}

async function stopAgent() {
  await fetch('/stop', {method:'POST'});
}

// Allow Ctrl+Enter to submit
document.getElementById('task-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) startAgent();
});

// ── Boot ───────────────────────────────────────────────────────────────────

connect();

// Restore status from server on page load
fetch('/status').then(r => r.json()).then(s => {
  setStatus(s.status || 'idle');
  if (s.task) taskLbl.textContent = s.task;
  if (s.status === 'running') setRunning(true);
});
</script>
</body>
</html>"""


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GUI Agent web server")
    ap.add_argument("--port",    type=int, default=5005)
    ap.add_argument("--host",    default="0.0.0.0")
    ap.add_argument("--display", default=":99")
    opts = ap.parse_args()

    print(f"\n  GUI Agent server  →  http://{opts.host}:{opts.port}")
    print(f"  Display: {opts.display}")
    print(f"  Open your browser at http://10.0.0.58:{opts.port}\n")

    app.run(host=opts.host, port=opts.port, debug=False, threaded=True)

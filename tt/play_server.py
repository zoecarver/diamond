#!/usr/bin/env python3
"""
Diamond World Model play server. Serves browser UI, proxies frames/actions
to play.py via file-based IPC in /tmp/.

Usage: python play_server.py
Then open http://localhost:8000
"""
import os
import json
import gymnasium
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

PORT = 8000
FRAME_PATH = "/tmp/diamond_live_frame.bmp"
ACTION_PATH = "/tmp/diamond_action.json"
STATUS_PATH = "/tmp/diamond_status.json"
GAME_PATH = "/tmp/diamond_game.json"
RESET_PATH = "/tmp/diamond_reset.json"
_local_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "data")
DATA_DIR = _local_data_dir if os.path.isdir(_local_data_dir) else "/tmp/diamond_data"

# Atari action names (full 18-action space)
ATARI_ACTION_NAMES = [
    "noop", "fire", "up", "right", "left", "down",
    "upright", "upleft", "downright", "downleft",
    "upfire", "rightfire", "leftfire", "downfire",
    "uprightfire", "upleftfire", "downrightfire", "downleftfire",
]

# Browser key -> atari action index (full keymap, same as src/game/keymap.py)
# Single keys
ATARI_BROWSER_KEYMAP_SINGLE = {
    " ": 1, "w": 2, "d": 3, "a": 4, "s": 5,
}
# Key combos (sorted longest first for priority matching)
ATARI_BROWSER_KEYMAP_COMBOS = [
    (["w", "d", " "], 14),  # uprightfire
    (["w", "a", " "], 15),  # upleftfire
    (["s", "d", " "], 16),  # downrightfire
    (["s", "a", " "], 17),  # downleftfire
    (["w", " "], 10),       # upfire
    (["d", " "], 11),       # rightfire
    (["a", " "], 12),       # leftfire
    (["s", " "], 13),       # downfire
    (["w", "d"], 6),        # upright
    (["w", "a"], 7),        # upleft
    (["s", "d"], 8),        # downright
    (["s", "a"], 9),        # downleft
]

_keymap_cache = {}


def get_games():
    games = []
    for name in sorted(os.listdir(DATA_DIR)):
        if os.path.isfile(os.path.join(DATA_DIR, name, "initial_frames.pt")):
            games.append(name)
    return games


def get_keymap(game):
    if game in _keymap_cache:
        return _keymap_cache[game]

    env_id = f"{game}NoFrameskip-v4"
    try:
        env = gymnasium.make(env_id)
        action_meanings = [x.lower() for x in env.unwrapped.get_action_meanings()]
        env.close()
    except Exception:
        action_meanings = ATARI_ACTION_NAMES[:6]

    # Filter single keys to only valid actions for this game
    keymap = {}
    for key, atari_idx in ATARI_BROWSER_KEYMAP_SINGLE.items():
        action_name = ATARI_ACTION_NAMES[atari_idx]
        if action_name in action_meanings:
            keymap[key] = action_meanings.index(action_name)

    # Filter combos
    combos = []
    for keys, atari_idx in ATARI_BROWSER_KEYMAP_COMBOS:
        action_name = ATARI_ACTION_NAMES[atari_idx]
        if action_name in action_meanings:
            combos.append({"keys": keys, "action": action_meanings.index(action_name)})

    result = {
        "actions": action_meanings,
        "keymap": keymap,
        "combos": combos,
    }
    _keymap_cache[game] = result
    return result


HTML_PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Diamond World Model Player</title>
<style>
  html, body { margin:0; height:100%%; background:#111; color:#eee; font-family: system-ui, sans-serif; }
  #app { padding: 20px; max-width: 720px; margin: 0 auto; }
  h1 { margin: 0 0 16px 0; font-size: 22px; }
  .hud { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; font-size: 14px; }
  .hud select { background: #222; color: #eee; border: 1px solid #444; padding: 4px 8px;
                border-radius: 4px; font-size: 14px; }
  .hud .stat { color: #888; font-family: monospace; }
  #frame { image-rendering: pixelated; width: 640px; height: 640px; background:#222;
           display:block; border: 1px solid #333; }
  #loading { display:none; position:absolute; top:50%%; left:50%%; transform:translate(-50%%,-50%%);
             background:rgba(0,0,0,0.8); padding:20px 40px; border-radius:8px; font-size:18px; z-index:10; }
  .frame-wrap { position: relative; display: inline-block; }
  .controls { margin-top: 16px; display: flex; gap: 24px; align-items: flex-start; }
  .key-grid { display: grid; grid-template-columns: repeat(3, 48px); gap: 4px; }
  .key { width: 48px; height: 48px; display: flex; align-items: center; justify-content: center;
         background: #222; border: 1px solid #444; border-radius: 6px; font-size: 14px;
         font-family: monospace; transition: background 0.05s; user-select: none; }
  .key.active { background: #09f; border-color: #0bf; color: #fff; }
  .key.empty { visibility: hidden; }
  .key.hidden { display: none; }
  #currentAction { margin-top: 12px; font-family: monospace; font-size: 14px; color: #888; }
  #status { margin-top: 8px; font-family: monospace; font-size: 13px; color: #666; }
</style>
</head>
<body>
<div id="app">
  <h1>Diamond World Model Player</h1>
  <div class="hud">
    <label>Game: <select id="gameSelect"></select></label>
    <button id="resetBtn" style="background:#333; color:#eee; border:1px solid #555; padding:4px 12px;
            border-radius:4px; font-size:13px; cursor:pointer;">Reset</button>
    <span class="stat" id="fpsDisplay">FPS: --</span>
    <span class="stat" id="frameDisplay">Frame: --</span>
  </div>
  <div class="frame-wrap">
    <img id="frame" alt="Waiting for frames..." />
    <div id="loading">Loading...</div>
  </div>
  <div class="controls">
    <div>
      <div style="margin-bottom: 8px; font-size: 13px; color: #888;">Controls</div>
      <div class="key-grid">
        <div class="key empty"></div>
        <div class="key" data-key="w" id="k-w">W</div>
        <div class="key empty"></div>
        <div class="key" data-key="a" id="k-a">A</div>
        <div class="key" data-key="s" id="k-s">S</div>
        <div class="key" data-key="d" id="k-d">D</div>
      </div>
      <div style="margin-top: 8px;">
        <div class="key" data-key=" " id="k-space" style="width: 152px;">Space</div>
      </div>
    </div>
    <div>
      <div style="margin-bottom: 8px; font-size: 13px; color: #888;">Action names</div>
      <div id="actionList" style="font-family: monospace; font-size: 13px; line-height: 1.6;"></div>
    </div>
  </div>
  <div id="currentAction">Action: noop</div>
  <div id="status">Waiting for generator...</div>
</div>

<script>
let keymap = {};
let combos = [];
let actionNames = ["noop"];
let pressed = new Set();

const gameSelect = document.getElementById('gameSelect');
const frameImg = document.getElementById('frame');
const loadingEl = document.getElementById('loading');

// Load game list
fetch('/games').then(r => r.json()).then(games => {
  games.forEach(g => {
    const opt = document.createElement('option');
    opt.value = g; opt.textContent = g;
    gameSelect.appendChild(opt);
  });
  // Default to Breakout if available
  if (games.includes('Breakout')) gameSelect.value = 'Breakout';
  loadKeymap(gameSelect.value);
});

gameSelect.addEventListener('change', () => {
  loadingEl.style.display = 'block';
  fetch('/select_game', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({game: gameSelect.value})
  });
  loadKeymap(gameSelect.value);
});

document.getElementById('resetBtn').addEventListener('click', () => {
  loadingEl.style.display = 'block';
  fetch('/reset', {method: 'POST'});
});

function loadKeymap(game) {
  fetch('/keymap?game=' + encodeURIComponent(game))
    .then(r => r.json())
    .then(data => {
      keymap = data.keymap;
      combos = data.combos;
      actionNames = data.actions;
      rebuildActionList();
      updateKeyVisibility();
    });
}

function rebuildActionList() {
  const el = document.getElementById('actionList');
  el.innerHTML = '';
  actionNames.forEach((name, i) => {
    const div = document.createElement('div');
    div.id = 'act-' + i;
    div.textContent = i + ': ' + name;
    div.style.padding = '1px 4px';
    div.style.borderRadius = '3px';
    div.style.transition = 'background 0.05s';
    el.appendChild(div);
  });
}

function updateKeyVisibility() {
  // Show/hide keys based on whether they map to valid actions
  const validKeys = new Set(Object.keys(keymap));
  combos.forEach(c => c.keys.forEach(k => validKeys.add(k)));
  document.querySelectorAll('.key[data-key]').forEach(el => {
    const k = el.dataset.key;
    el.classList.toggle('hidden', !validKeys.has(k));
  });
}

function resolveAction() {
  // Try combos first (longest match wins, already sorted)
  for (const combo of combos) {
    if (combo.keys.every(k => pressed.has(k))) {
      return combo.action;
    }
  }
  // Single keys
  for (const key of pressed) {
    if (keymap[key] !== undefined) return keymap[key];
  }
  return 0;
}

function updateUI() {
  const action = resolveAction();
  document.querySelectorAll('.key[data-key]').forEach(el => {
    el.classList.toggle('active', pressed.has(el.dataset.key));
  });
  const name = actionNames[action] || 'unknown';
  document.getElementById('currentAction').textContent = 'Action: ' + name;
  // Highlight active action in list
  actionNames.forEach((_, i) => {
    const el = document.getElementById('act-' + i);
    if (el) {
      el.style.background = (i === action) ? '#09f' : '';
      el.style.color = (i === action) ? '#fff' : '';
    }
  });
}

const VALID_KEYS = new Set(['w','a','s','d',' ']);
document.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k === ' ') e.preventDefault();
  if (VALID_KEYS.has(k)) {
    pressed.add(k);
    updateUI();
    sendAction();
  }
});
document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  pressed.delete(k);
  updateUI();
  sendAction();
});

let actionPending = false;
let actionDirty = false;
function sendAction() {
  actionDirty = true;
  if (actionPending) return;
  actionPending = true;
  actionDirty = false;
  fetch('/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: resolveAction()})
  }).finally(() => {
    actionPending = false;
    if (actionDirty) sendAction();
  });
}

// Frame polling
let lastFrameIndex = -1;
async function pollFrames() {
  while (true) {
    try {
      const resp = await fetch('/frame?t=' + Date.now());
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        frameImg.onload = () => URL.revokeObjectURL(url);
        frameImg.src = url;
        loadingEl.style.display = 'none';
        const fps = resp.headers.get('X-FPS') || '--';
        const idx = resp.headers.get('X-Frame-Index') || '--';
        document.getElementById('fpsDisplay').textContent = 'FPS: ' + fps;
        document.getElementById('frameDisplay').textContent = 'Frame: ' + idx;
        document.getElementById('status').textContent = '';
      }
    } catch(e) {}
    await new Promise(r => setTimeout(r, 30));
  }
}
pollFrames();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            body = (HTML_PAGE % {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/games":
            body = json.dumps(get_games()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/frame":
            try:
                data = open(FRAME_PATH, "rb").read()
                fps, idx = "--", "--"
                try:
                    st = json.load(open(STATUS_PATH))
                    fps = "%.1f" % st.get("fps", 0)
                    idx = str(st.get("frame_index", 0))
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "image/bmp")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-FPS", fps)
                self.send_header("X-Frame-Index", idx)
                self.send_header("Access-Control-Expose-Headers", "X-FPS, X-Frame-Index")
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self.send_error(404, "No frame yet")

        elif path == "/keymap":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            game = params.get("game", "Breakout")
            from urllib.parse import unquote
            game = unquote(game)
            body = json.dumps(get_keymap(game)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        path = self.path.split("?")[0]

        if path == "/action":
            tmp = ACTION_PATH + ".tmp"
            with open(tmp, "w") as f:
                f.write(body.decode())
            os.rename(tmp, ACTION_PATH)
            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == "/select_game":
            tmp = GAME_PATH + ".tmp"
            with open(tmp, "w") as f:
                f.write(body.decode())
            os.rename(tmp, GAME_PATH)
            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif path == "/reset":
            tmp = RESET_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"reset": True}, f)
            os.rename(tmp, RESET_PATH)
            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        else:
            self.send_error(404)


if __name__ == "__main__":
    with open(ACTION_PATH, "w") as f:
        json.dump({"action": 0}, f)
    print("Diamond play server on http://0.0.0.0:%d" % PORT)
    print("Waiting for play.py to generate frames at %s" % FRAME_PATH)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()

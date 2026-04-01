#!/usr/bin/env python3
"""
Generate an interactive Markov chain step-through visualization.

Run demo_voice_generation.py first to train and save a model, then run this.

Usage:
    python generate_markov_viz.py

Output:
    markov_visualization.html — open this file in any browser.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "phoenixvoice"))

MODEL_FILE = Path(__file__).parent / "my_model.pkl"


def main():
    if not MODEL_FILE.exists():
        print(f"No model found at {MODEL_FILE.name}.")
        print("Run demo_voice_generation.py first to train and save a model.")
        sys.exit(1)

    print(f"Loading model from '{MODEL_FILE.name}'...")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    print("Serializing transitions...")
    transitions = {}
    for state, next_tokens in model.transitions.items():
        key = "|||".join(state)
        total = model.total_transitions[state]
        transitions[key] = {
            token: round(count / total, 6)
            for token, count in sorted(next_tokens.items(), key=lambda x: -x[1])
        }

    seen = set()
    start_states = []
    for s in model.start_states:
        k = "|||".join(s)
        if k not in seen:
            seen.add(k)
            start_states.append(k)

    model_data = {
        "transitions": transitions,
        "start_states": start_states,
        "order": model.order,
    }

    out_path = Path(__file__).parent / "markov_visualization.html"
    print("Generating HTML...")
    out_path.write_text(build_html(model_data), encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved to {out_path.name} ({size_kb:.0f} KB)")
    print("Open it in your browser to explore the model!")


def build_html(model_data):
    data_json = json.dumps(model_data, ensure_ascii=False)
    return _HTML_BEFORE + data_json + _HTML_AFTER


_HTML_BEFORE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Markov Chain Visualizer</title>
  <style>
    :root {
      --bg:      #070c18;
      --surface: #0f1825;
      --surf2:   #1a2535;
      --border:  #2d3f56;
      --text:    #e8f0fe;
      --dim:     #7a90a8;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem 3rem;
    }

    .wrap { max-width: 1080px; margin: 0 auto; }

    /* ── Header ── */
    header { text-align: center; margin-bottom: 2rem; }

    h1 {
      font-size: 2.1rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg,
        hsl(224,64%,55%) 0%,
        hsl(270,60%,62%) 35%,
        hsl(330,60%,55%) 65%,
        hsl(35,80%,55%)  100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.4rem;
    }

    header p { color: var(--dim); font-size: 0.95rem; }

    /* ── Panel ── */
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 0.9rem;
    }

    .panel-label {
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.3px;
      color: var(--dim);
      margin-bottom: 0.7rem;
    }

    /* ── Generated text ── */
    #gen-text {
      font-family: Georgia, 'Times New Roman', serif;
      font-size: 1.05rem;
      line-height: 1.85;
      min-height: 72px;
      max-height: 170px;
      overflow-y: auto;
      word-spacing: 2px;
    }

    #gen-text::-webkit-scrollbar      { width: 5px; }
    #gen-text::-webkit-scrollbar-track  { background: transparent; }
    #gen-text::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    .tok-past  { color: var(--dim); }
    .tok-state {
      color: #fde68a;
      background: rgba(251,191,36,0.13);
      border-bottom: 2px solid #fbbf24;
      padding: 1px 3px;
      border-radius: 3px 3px 0 0;
    }

    /* ── State chips ── */
    #state-chips {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .chip {
      background: var(--surf2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.35rem 0.85rem;
      font-family: 'SFMono-Regular', Consolas, monospace;
      font-size: 0.95rem;
      color: #fde68a;
    }

    .chip-sep { color: var(--dim); font-size: 1.3rem; user-select: none; }

    /* ── Connector ── */
    .connector {
      text-align: center;
      font-size: 1.8rem;
      color: var(--border);
      margin: 0.15rem 0;
      user-select: none;
    }

    /* ── Token cards ── */
    .legend {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.74rem;
      color: var(--dim);
      margin-bottom: 0.85rem;
    }

    .legend-bar {
      width: 180px;
      height: 10px;
      border-radius: 5px;
      flex-shrink: 0;
      background: linear-gradient(to right,
        hsl(224,64%,33%),
        hsl(180,70%,30%),
        hsl(270,60%,42%),
        hsl(35,80%,40%),
        hsl(348,72%,38%)
      );
    }

    #tokens-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      min-height: 58px;
    }

    .token-card {
      border-radius: 10px;
      padding: 0.55rem 0.9rem;
      min-width: 68px;
      text-align: center;
      border: 1px solid rgba(255,255,255,0.07);
      transition: transform 0.12s ease, box-shadow 0.12s ease;
      cursor: default;
    }

    .token-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 6px 18px rgba(0,0,0,0.55);
      border-color: rgba(255,255,255,0.2);
    }

    .token-card.chosen {
      transform: scale(1.18) translateY(-2px);
      box-shadow: 0 0 0 3px #fff, 0 0 22px rgba(255,255,255,0.22);
      border-color: white;
      z-index: 2;
      position: relative;
    }

    .tc-word {
      font-family: 'SFMono-Regular', Consolas, monospace;
      font-size: 0.88rem;
      font-weight: 700;
      color: #fff;
      word-break: break-all;
    }

    .tc-pct {
      font-size: 0.68rem;
      color: rgba(255,255,255,0.72);
      margin-top: 3px;
    }

    .overflow-note {
      color: var(--dim);
      font-size: 0.8rem;
      align-self: center;
      padding: 0 0.3rem;
    }

    /* ── Controls ── */
    .controls {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.7rem;
      flex-wrap: wrap;
      padding: 0.9rem 0 0.4rem;
    }

    button {
      background: var(--surf2);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 0.87rem;
      padding: 0.55rem 1.3rem;
      cursor: pointer;
      transition: background 0.12s, transform 0.1s, filter 0.12s;
      user-select: none;
    }
    button:hover  { background: #26384f; }
    button:active { transform: scale(0.95); }

    #step-btn {
      background: linear-gradient(135deg, hsl(224,64%,40%), hsl(270,60%,48%));
      border: none;
      font-weight: 600;
      padding: 0.55rem 2rem;
      font-size: 0.92rem;
    }
    #step-btn:hover { filter: brightness(1.15); }

    #auto-btn.playing {
      background: linear-gradient(135deg, hsl(160,65%,28%), hsl(160,65%,38%));
      border: none;
    }

    .speed-wrap {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      color: var(--dim);
      font-size: 0.8rem;
    }
    input[type=range] {
      accent-color: hsl(270,60%,55%);
      width: 88px;
      cursor: pointer;
    }

    /* ── Status ── */
    #status {
      text-align: center;
      color: var(--dim);
      font-size: 0.8rem;
      margin-top: 0.4rem;
      min-height: 1.1em;
      transition: color 0.2s;
    }
    #status.err { color: #f87171; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Markov Chain Visualizer</h1>
      <p>Step through text generation one token at a time &mdash; see exactly how the model decides what comes next.</p>
    </header>

    <div class="controls">
      <button id="reset-btn" onclick="resetModel()">&#8635; Reset</button>
      <button id="step-btn"  onclick="step()">Step &rarr;</button>
      <button id="auto-btn"  onclick="toggleAuto()">&#9654; Auto-play</button>
      <div class="speed-wrap">
        <label for="speed">Speed</label>
        <input type="range" id="speed" min="1" max="10" value="3">
      </div>
    </div>

    <div id="status"></div>

    <div class="panel">
      <div class="panel-label">Generated Text &mdash; current state highlighted in gold</div>
      <div id="gen-text"></div>
    </div>

    <div class="panel">
      <div class="panel-label">
        Current State &mdash; the last <span id="order-lbl">?</span> token(s) the model looks at
      </div>
      <div id="state-chips"></div>
    </div>

    <div class="connector">&#8595;</div>

    <div class="panel">
      <div class="panel-label">
        Possible Next Tokens
        <span id="opt-count" style="font-weight:400;text-transform:none;letter-spacing:0;margin-left:0.5rem"></span>
      </div>
      <div class="legend">
        <span>Less likely</span>
        <div class="legend-bar"></div>
        <span>More likely</span>
        <span style="margin-left:auto;font-style:italic">Colors are relative within this state</span>
      </div>
      <div id="tokens-grid"></div>
    </div>
  </div>

  <script>
  </script>

  <script type="application/json" id="model-data">"""

_HTML_AFTER = """
  </script>

  <script>
  "use strict";
  const MODEL_DATA = JSON.parse(document.getElementById('model-data').textContent);

  /* ── jewel-tone colour scale ──────────────────────────────────────────────
     Five stops from cool (sapphire) to warm (ruby).
     t = 0 → least likely in this state; t = 1 → most likely.          */
  const STOPS = [
    [0.00, 224, 64, 33],   // Sapphire
    [0.25, 180, 70, 30],   // Aquamarine / Tourmaline
    [0.50, 270, 60, 42],   // Amethyst
    [0.75,  35, 80, 40],   // Topaz / Amber
    [1.00, 348, 72, 38],   // Ruby / Garnet
  ];

  function jewelColor(t) {
    t = Math.max(0, Math.min(1, t));
    let lo = STOPS[0], hi = STOPS[STOPS.length - 1];
    for (let i = 0; i < STOPS.length - 1; i++) {
      if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { lo = STOPS[i]; hi = STOPS[i + 1]; break; }
    }
    const f = (hi[0] === lo[0]) ? 0 : (t - lo[0]) / (hi[0] - lo[0]);
    let dh = hi[1] - lo[1];
    if (dh >  180) dh -= 360;
    if (dh < -180) dh += 360;
    const h = (((lo[1] + f * dh) % 360) + 360) % 360;
    const s = lo[2] + f * (hi[2] - lo[2]);
    const l = lo[3] + f * (hi[3] - lo[3]);
    return `hsl(${h.toFixed(0)},${s.toFixed(0)}%,${l.toFixed(0)}%)`;
  }

  /* ── state ──────────────────────────────────────────────────────────────── */
  let tokens    = [];
  let curState  = [];
  let stepCount = 0;
  let autoTimer = null;
  const MAX_CARDS = 45;

  /* ── core logic ─────────────────────────────────────────────────────────── */
  function resetModel() {
    stopAuto();
    stepCount = 0;
    const keys = MODEL_DATA.start_states;
    const key  = keys[Math.floor(Math.random() * keys.length)];
    curState   = key.split('|||');
    tokens     = [...curState];
    render(null);
    setStatus(`Started fresh. Current state: "${curState.join(' ')}"`, false);
  }

  function step() {
    const key   = curState.join('|||');
    const nexts = MODEL_DATA.transitions[key];
    if (!nexts || !Object.keys(nexts).length) {
      setStatus('No transitions from this state — end of chain. Press &#8635; Reset to start again.', true);
      stopAuto();
      return;
    }
    const chosen = weightedSample(nexts);
    stepCount++;
    tokens.push(chosen);
    curState = [...curState.slice(1), chosen];
    render(chosen);
    const n = Object.keys(nexts).length;
    setStatus(`Step ${stepCount} \u2014 \u201c${chosen}\u201d chosen from ${n} option${n !== 1 ? 's' : ''}.`, false);
    if (chosen === '<END>') {
      stopAuto();
      setStatus(`Step ${stepCount} \u2014 reached <END> token. Press &#8635; Reset to start again.`, false);
    }
  }

  function weightedSample(nexts) {
    const entries = Object.entries(nexts);
    let r = Math.random(), cum = 0;
    for (const [tok, p] of entries) { cum += p; if (r < cum) return tok; }
    return entries[entries.length - 1][0];
  }

  /* ── auto-play ───────────────────────────────────────────────────────────── */
  function toggleAuto() { autoTimer ? stopAuto() : startAuto(); }

  function startAuto() {
    const speed = +document.getElementById('speed').value;
    const delay = Math.round(1050 - speed * 95);
    const btn = document.getElementById('auto-btn');
    btn.innerHTML = '&#9646;&#9646; Pause';
    btn.classList.add('playing');
    autoTimer = setInterval(() => {
      step();
      if (!MODEL_DATA.transitions[curState.join('|||')]) stopAuto();
    }, delay);
  }

  function stopAuto() {
    clearInterval(autoTimer);
    autoTimer = null;
    const btn = document.getElementById('auto-btn');
    btn.innerHTML = '&#9654; Auto-play';
    btn.classList.remove('playing');
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('speed').addEventListener('change', () => {
      if (autoTimer) { stopAuto(); startAuto(); }
    });
    resetModel();
  });

  /* ── rendering ───────────────────────────────────────────────────────────── */
  function render(chosen) {
    renderText();
    renderState();
    renderTokens(chosen);
  }

  function renderText() {
    const order = MODEL_DATA.order;
    const si    = tokens.length - order;
    const html  = tokens.map((t, i) =>
      `<span class="${i >= si ? 'tok-state' : 'tok-past'}">${esc(t)}</span>`
    ).join(' ');
    const el = document.getElementById('gen-text');
    el.innerHTML = html;
    el.scrollTop = el.scrollHeight;
  }

  function renderState() {
    document.getElementById('order-lbl').textContent = MODEL_DATA.order;
    const html = curState.map((t, i) => {
      const sep = i < curState.length - 1 ? '<span class="chip-sep">\u203a</span>' : '';
      return `<span class="chip">${esc(t)}</span>${sep}`;
    }).join('');
    document.getElementById('state-chips').innerHTML = html;
  }

  function renderTokens(chosen) {
    const key    = curState.join('|||');
    const nexts  = MODEL_DATA.transitions[key] || {};
    const sorted = Object.entries(nexts).sort((a, b) => b[1] - a[1]);

    const countEl = document.getElementById('opt-count');
    countEl.textContent = sorted.length
      ? `\u2014 ${sorted.length} option${sorted.length !== 1 ? 's' : ''}`
      : '';

    const grid = document.getElementById('tokens-grid');

    if (!sorted.length) {
      grid.innerHTML = '<span style="color:#f87171;font-style:italic">No transitions \u2014 end of chain</span>';
      return;
    }

    const maxP  = sorted[0][1];
    const minP  = sorted[sorted.length - 1][1];
    const range = maxP - minP || 0.001;

    const show = sorted.slice(0, MAX_CARDS);
    const rest = sorted.length - show.length;

    const cards = show.map(([tok, p]) => {
      const norm = (p - minP) / range;
      const bg   = jewelColor(norm);
      const cls  = tok === chosen ? ' chosen' : '';
      return `<div class="token-card${cls}" style="background:${bg}">
        <div class="tc-word">${esc(tok)}</div>
        <div class="tc-pct">${(p * 100).toFixed(1)}%</div>
      </div>`;
    }).join('');

    const more = rest > 0
      ? `<span class="overflow-note">+${rest} more</span>`
      : '';

    grid.innerHTML = cards + more;
  }

  /* ── utilities ───────────────────────────────────────────────────────────── */
  function setStatus(msg, err) {
    const el = document.getElementById('status');
    el.innerHTML = msg;
    el.classList.toggle('err', err);
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()

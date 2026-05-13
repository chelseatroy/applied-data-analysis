#!/usr/bin/env python3
"""
Getflix Session 8 — Instructor Dashboard

A four-tab web UI that polls the class Google Sheet and queries every
student's live ngrok server in real time.

  Tab 1 — Live          auto-refreshing comparison table
  Tab 2 — Compare       pick any user ID; see all models' recommendations
  Tab 3 — Ensemble      select servers; get a consensus recommendation list
  Tab 4 — Rate & Rec    rate movies yourself; each model finds your nearest
                        neighbour and recommends from there

Usage:
    python session_8_dashboard.py
    Open http://localhost:5055 and project that tab.

Optional flags:
    --port      Port to serve on (default: 5055; overridden by $PORT env var)
    --interval  Browser auto-refresh interval in seconds (default: 15)
    --user-id   Default user ID for Live and Compare tabs (default: 42)
    --genre     Default genre for forecast column (default: Drama)
    --timeout   Per-request timeout in seconds (default: 4)
"""

import argparse
import csv
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1tcw2AkRCE3A2dMtvDi74Ly1K-7qVQPZ_LjrObDnry84/"
    "export?format=csv&gid=748395572"
)
NAME_COL = "Name"
URL_COL  = "URL"

CONFIG = {"user_id": 42, "genre": "Drama", "timeout": 4, "interval": 15}
GLOBAL_TIMEOUT = 8  # seconds before /data abandons slow servers

_FALLBACK_MOVIES = [
    {"movie_id": 50,  "title": "Star Wars (1977)"},
    {"movie_id": 172, "title": "Empire Strikes Back, The (1980)"},
    {"movie_id": 181, "title": "Return of the Jedi (1983)"},
    {"movie_id": 174, "title": "Raiders of the Lost Ark (1981)"},
    {"movie_id": 127, "title": "Godfather, The (1972)"},
    {"movie_id": 98,  "title": "Silence of the Lambs, The (1991)"},
    {"movie_id": 56,  "title": "Pulp Fiction (1994)"},
    {"movie_id": 1,   "title": "Toy Story (1995)"},
    {"movie_id": 7,   "title": "Twelve Monkeys (1995)"},
    {"movie_id": 100, "title": "Fargo (1996)"},
    {"movie_id": 258, "title": "Contact (1997)"},
    {"movie_id": 204, "title": "Back to the Future (1985)"},
]


# ---------------------------------------------------------------------------
# Sheet + server helpers
# ---------------------------------------------------------------------------

def fetch_rows():
    try:
        r = requests.get(SHEET_URL, timeout=8)
        r.raise_for_status()
    except Exception as e:
        return [], str(e)
    reader = csv.DictReader(io.StringIO(r.text))
    rows = []
    for row in reader:
        name = row.get(NAME_COL, "").strip()
        url  = row.get(URL_COL,  "").strip()
        if name and url:
            if not url.startswith("http"):
                url = "https://" + url
            rows.append((name, url.rstrip("/")))
    return rows, None


def get(url, path, timeout):
    try:
        r = requests.get(url + path, timeout=timeout)
        return (r.json(), None) if r.status_code == 200 else (None, f"HTTP {r.status_code}")
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception:
        return None, "offline"


def query_server(name, url, user_id, genre, timeout):
    result = {"name": name, "url": url}

    health, err = get(url, "/health", timeout)
    if err:
        result["status"] = err
        return result
    result["status"] = "online"

    c   = health.get("clustering",  {})
    rec = health.get("recommender", {})
    result["cluster_model"] = c.get("model_type",    "?")
    result["cluster_fill"]  = c.get("fill_strategy", "?")
    result["rec_model"]     = rec.get("algo_type",   "?")

    data, err = get(url, f"/cluster?user_id={user_id}", timeout)
    result["cluster_label"] = data["cluster"] if data else f"({err})"

    data, err = get(url, f"/recommend?user_id={user_id}&n=5", timeout)
    if data:
        recs = data.get("recommendations", [])
        result["recs"] = [
            {"title": r["title"], "score": r["predicted_rating"]}
            for r in recs
        ]
    else:
        result["recs"] = []
        result["rec_err"] = err

    data, err = get(url, f"/forecast?genre={genre}&steps=1", timeout)
    if data:
        fc = data.get("forecast", [])
        result["forecast"] = round(fc[0], 3) if fc else "—"
    else:
        result["forecast"] = f"({err})"

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/data")
def data():
    user_id = request.args.get("user_id", type=int,  default=CONFIG["user_id"])
    genre   = request.args.get("genre",   default=CONFIG["genre"])
    timeout = CONFIG["timeout"]

    rows, sheet_err = fetch_rows()
    results = []
    if rows:
        with ThreadPoolExecutor(max_workers=len(rows)) as executor:
            future_to_name = {
                executor.submit(query_server, n, u, user_id, genre, timeout): n
                for n, u in rows
            }
            try:
                for future in as_completed(future_to_name, timeout=GLOBAL_TIMEOUT):
                    results.append(future.result())
            except FuturesTimeoutError:
                for future, name in future_to_name.items():
                    if name not in {r["name"] for r in results}:
                        results.append({"name": name, "status": "timeout"})
    online  = [r for r in results if r.get("status") == "online"]
    labels  = sorted(set(str(r.get("cluster_label", "?")) for r in online))

    return jsonify({
        "results":    results,
        "online":     len(online),
        "total":      len(results),
        "labels":     labels,
        "sheet_err":  sheet_err,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
        "user_id":    user_id,
        "genre":      genre,
        "interval":   CONFIG["interval"],
    })


@app.route("/servers")
def servers():
    rows, err = fetch_rows()
    result = []
    for name, url in rows:
        try:
            r = requests.get(f"{url}/health", timeout=CONFIG["timeout"])
            online = r.status_code == 200
        except Exception:
            online = False
        result.append({"name": name, "url": url, "online": online})
    return jsonify({"servers": result, "sheet_err": err})


@app.route("/ensemble", methods=["POST"])
def ensemble():
    body    = request.get_json(silent=True) or {}
    user_id = body.get("user_id", CONFIG["user_id"])
    urls    = body.get("urls", [])
    n       = body.get("n", 10)

    if not urls:
        return jsonify({"error": "no servers selected"}), 400

    tally = {}
    for url in urls:
        try:
            r = requests.get(
                f"{url}/recommend?user_id={user_id}&n=50",
                timeout=CONFIG["timeout"],
            )
            if r.status_code == 200:
                for rec in r.json().get("recommendations", []):
                    mid = rec["movie_id"]
                    if mid not in tally:
                        tally[mid] = {"title": rec["title"], "scores": []}
                    tally[mid]["scores"].append(rec["predicted_rating"])
        except Exception:
            pass

    n_servers = len(urls)
    recs = sorted(
        [
            {
                "movie_id":  mid,
                "title":     info["title"],
                "avg_score": round(sum(info["scores"]) / len(info["scores"]), 2),
                "n_servers": len(info["scores"]),
                "consensus": round(len(info["scores"]) / n_servers, 2),
            }
            for mid, info in tally.items()
        ],
        key=lambda x: (-x["n_servers"], -x["avg_score"]),
    )
    return jsonify({
        "recommendations":  recs[:n],
        "n_servers_queried": n_servers,
        "user_id":          user_id,
    })


@app.route("/sample_movies")
def sample_movies():
    rows, _ = fetch_rows()
    for _, url in rows:
        try:
            r = requests.get(f"{url}/movies?n=12", timeout=CONFIG["timeout"])
            if r.status_code == 200:
                return jsonify(r.json())
        except Exception:
            continue
    return jsonify(_FALLBACK_MOVIES)


@app.route("/try_it", methods=["POST"])
def try_it():
    body    = request.get_json(silent=True) or {}
    ratings = body.get("ratings", {})
    n       = body.get("n", 5)

    rows, _ = fetch_rows()
    results = []
    for name, url in rows:
        try:
            r = requests.post(
                f"{url}/recommend_new_user",
                json={"ratings": ratings, "n": n},
                timeout=CONFIG["timeout"] * 3,
            )
            if r.status_code == 200:
                d = r.json()
                results.append({
                    "name":            name,
                    "status":          "ok",
                    "nearest_user":    d.get("nearest_user"),
                    "similarity":      d.get("similarity"),
                    "recommendations": d.get("recommendations", [])[:3],
                })
            else:
                results.append({"name": name, "status": f"HTTP {r.status_code}"})
        except Exception:
            results.append({"name": name, "status": "offline"})

    return jsonify({"results": results})


@app.route("/")
def index():
    return render_template_string(HTML, **CONFIG)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Getflix Session 8</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }

header {
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 0.8rem 2rem; display: flex; align-items: center; justify-content: space-between;
}
header h1 { font-size: 1.2rem; color: #58a6ff; }
.meta { color: #8b949e; font-size: 0.82rem; display: flex; gap: 1.2rem; align-items: center; }
.badge { background: #21262d; border: 1px solid #30363d; border-radius: 12px; padding: 0.15rem 0.6rem; font-size: 0.78rem; }
.badge.good { border-color: #238636; color: #3fb950; }

/* tabs */
.tab-nav { background: #161b22; border-bottom: 1px solid #30363d; display: flex; padding: 0 2rem; }
.tab-btn {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: #8b949e; padding: 0.7rem 1.2rem; cursor: pointer; font-size: 0.88rem;
}
.tab-btn:hover { color: #e6edf3; }
.tab-btn.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.tab-pane { display: none; padding: 1.5rem 2rem; }
.tab-pane.active { display: block; }

/* tables */
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
thead th {
  background: #161b22; color: #8b949e; font-weight: 600; text-align: left;
  padding: 0.55rem 0.9rem; border-bottom: 1px solid #30363d; white-space: nowrap;
}
tbody tr { border-bottom: 1px solid #21262d; }
tbody tr:hover { background: #161b22; }
tbody td { padding: 0.55rem 0.9rem; color: #c9d1d9; vertical-align: top; }
td.name  { color: #f0f6fc; font-weight: 600; white-space: nowrap; }
td.model { color: #58a6ff; }
td.score { color: #3fb950; font-weight: 600; }
td.cluster-label { font-size: 1.1rem; font-weight: 700; color: #f0f6fc; text-align: center; }
.rec-list { list-style: none; }
.rec-list li { font-size: 0.82rem; color: #c9d1d9; line-height: 1.6; }
.rec-list li span { color: #3fb950; font-weight: 600; margin-left: 4px; }

/* dot */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.dot.online  { background: #3fb950; }
.dot.offline { background: #f85149; }

/* summary bar */
.summary {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 0.8rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.9rem; color: #c9d1d9; min-height: 2.2rem;
}
.summary strong { color: #f0f6fc; }
.sheet-err { color: #e3b341; font-size: 0.8rem; margin-bottom: 0.6rem; }
.empty { color: #8b949e; padding: 2.5rem; text-align: center; }

/* controls */
.controls { display: flex; gap: 1rem; align-items: flex-end; margin-bottom: 1.2rem; flex-wrap: wrap; }
.field { display: flex; flex-direction: column; gap: 0.3rem; }
.field label { font-size: 0.78rem; color: #8b949e; }
.field input[type=number], .field select {
  background: #21262d; border: 1px solid #30363d; border-radius: 6px;
  color: #e6edf3; padding: 0.4rem 0.7rem; font-size: 0.88rem; width: 120px;
}
.btn {
  background: #238636; border: none; border-radius: 6px; color: #fff;
  padding: 0.45rem 1.1rem; font-size: 0.88rem; cursor: pointer;
}
.btn:hover { background: #2ea043; }
.btn.secondary { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; }
.btn.secondary:hover { background: #30363d; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.countdown { color: #58a6ff; font-size: 0.82rem; }

/* server checkboxes */
.server-grid { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 1.2rem; }
.server-chip {
  background: #21262d; border: 1px solid #30363d; border-radius: 16px;
  padding: 0.3rem 0.8rem; font-size: 0.82rem; cursor: pointer; display: flex; align-items: center; gap: 0.4rem;
}
.server-chip input { accent-color: #58a6ff; }
.server-chip.online { border-color: #238636; }

/* ensemble results */
.consensus-bar {
  height: 6px; background: #21262d; border-radius: 3px; margin-top: 4px; width: 80px; display: inline-block;
}
.consensus-fill { height: 100%; background: #58a6ff; border-radius: 3px; }

/* movie grid */
.movie-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-bottom: 1.5rem; }
.movie-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 0.8rem;
}
.movie-card .title { font-size: 0.85rem; color: #f0f6fc; margin-bottom: 0.5rem; line-height: 1.3; min-height: 2.4rem; }
.stars { display: flex; gap: 3px; }
.star-btn {
  background: none; border: none; font-size: 1.1rem; cursor: pointer;
  color: #30363d; padding: 0; line-height: 1; transition: color 0.1s;
}
.star-btn.lit { color: #e3b341; }
.star-btn:hover { color: #e3b341; }

/* try-it results */
.try-result-card {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
}
.try-result-card .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.try-result-card .header strong { color: #f0f6fc; }
.try-result-card .neighbour { font-size: 0.8rem; color: #8b949e; }
.try-result-card .neighbour span { color: #58a6ff; }

/* loading spinner */
.loading { color: #8b949e; padding: 2rem; text-align: center; font-size: 0.9rem; }
</style>
</head>
<body>

<header>
  <h1>🎬 Getflix — Session 8</h1>
  <div class="meta">
    <span id="online-badge" class="badge">— / — online</span>
    <span>updated <strong id="updated-at">—</strong></span>
    <span class="countdown">refresh in <strong id="countdown">{{ interval }}</strong>s</span>
  </div>
</header>

<nav class="tab-nav">
  <button class="tab-btn active" data-tab="live">Live</button>
  <button class="tab-btn" data-tab="compare">Compare</button>
  <button class="tab-btn" data-tab="ensemble">Ensemble</button>
  <button class="tab-btn" data-tab="rate">Rate &amp; Recommend</button>
</nav>

<!-- ======================================================= TAB 1: LIVE -->
<div id="tab-live" class="tab-pane active">
  <div id="live-sheet-err" class="sheet-err" style="display:none"></div>
  <div id="live-summary" class="summary">Waiting for data…</div>
  <div id="live-content"><div class="empty">Waiting for students to submit their URLs…</div></div>
</div>

<!-- ==================================================== TAB 2: COMPARE -->
<div id="tab-compare" class="tab-pane">
  <div class="controls">
    <div class="field">
      <label>User ID</label>
      <input type="number" id="compare-uid" value="{{ user_id }}" min="1" max="943">
    </div>
    <div class="field">
      <label>Genre (forecast)</label>
      <input type="text" id="compare-genre" value="{{ genre }}" style="width:100px">
    </div>
    <button class="btn" onclick="runCompare()">Compare</button>
  </div>
  <div id="compare-content"><div class="empty">Enter a user ID and click Compare.</div></div>
</div>

<!-- =================================================== TAB 3: ENSEMBLE -->
<div id="tab-ensemble" class="tab-pane">
  <div class="controls">
    <div class="field">
      <label>User ID</label>
      <input type="number" id="ensemble-uid" value="{{ user_id }}" min="1" max="943">
    </div>
    <div class="field">
      <label>Recommendations</label>
      <input type="number" id="ensemble-n" value="10" min="1" max="50" style="width:80px">
    </div>
    <button class="btn secondary" onclick="loadServers()">Refresh servers</button>
    <button class="btn" onclick="runEnsemble()">Run ensemble</button>
  </div>
  <div id="server-grid" class="server-grid"><div class="empty">Loading servers…</div></div>
  <div id="ensemble-content"></div>
</div>

<!-- ================================================= TAB 4: RATE & REC -->
<div id="tab-rate" class="tab-pane">
  <p style="color:#8b949e;font-size:0.88rem;margin-bottom:1.2rem;">
    Rate as many movies as you like (1–5 stars). Each student server will find
    your nearest neighbour in its training data and recommend from there.
  </p>
  <div id="movie-grid" class="movie-grid"><div class="loading">Loading movies…</div></div>
  <div style="display:flex;gap:1rem;align-items:center;margin-bottom:1.5rem;">
    <button class="btn" onclick="runTryIt()">Get Recommendations</button>
    <button class="btn secondary" onclick="clearRatings()">Clear ratings</button>
    <span id="rating-count" style="color:#8b949e;font-size:0.85rem;">0 movies rated</span>
  </div>
  <div id="tryit-content"></div>
</div>

<script>
const INTERVAL = {{ interval }};
const ratings  = {};
let liveTimer  = null;
let liveCountdown = INTERVAL;

// ------------------------------------------------------------------ TABS
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'live') startLive();
    else stopLive();
    if (btn.dataset.tab === 'ensemble') loadServers();
    if (btn.dataset.tab === 'rate') loadMovies();
  });
});

// ------------------------------------------------------------ LIVE TAB
function startLive() {
  refreshLive();
  liveTimer = setInterval(() => {
    liveCountdown--;
    document.getElementById('countdown').textContent = liveCountdown;
    if (liveCountdown <= 0) { refreshLive(); liveCountdown = INTERVAL; }
  }, 1000);
}
function stopLive() { clearInterval(liveTimer); liveTimer = null; }

function refreshLive() {
  liveCountdown = INTERVAL;
  document.getElementById('countdown').textContent = liveCountdown;
  fetch('/data').then(r => r.json()).then(renderLive).catch(console.error);
}

function renderLive(data) {
  document.getElementById('updated-at').textContent = data.updated_at;
  const badge = document.getElementById('online-badge');
  badge.textContent = `${data.online} / ${data.total} online`;
  badge.className = 'badge' + (data.online === data.total && data.total > 0 ? ' good' : '');

  const err = document.getElementById('live-sheet-err');
  if (data.sheet_err) { err.textContent = '⚠ Sheet: ' + data.sheet_err; err.style.display = 'block'; }
  else err.style.display = 'none';

  const summary = document.getElementById('live-summary');
  if (data.results.length === 0) { summary.innerHTML = 'Waiting for students to submit their URLs…'; }
  else if (data.labels.length > 1) {
    summary.innerHTML = `User <strong>${data.user_id}</strong> landed in <strong>${data.labels.length}</strong> different clusters: <strong>${data.labels.join(', ')}</strong>`;
  } else if (data.labels.length === 1) {
    summary.innerHTML = `User <strong>${data.user_id}</strong> landed in cluster <strong>${data.labels[0]}</strong> on every server.`;
  } else { summary.innerHTML = 'No servers online.'; }

  document.getElementById('live-content').innerHTML = renderTable(data.results, data.user_id, data.genre);
}

// --------------------------------------------------------- COMPARE TAB
function runCompare() {
  const uid   = document.getElementById('compare-uid').value   || 42;
  const genre = document.getElementById('compare-genre').value || 'Drama';
  document.getElementById('compare-content').innerHTML = '<div class="loading">Querying servers…</div>';
  fetch(`/data?user_id=${uid}&genre=${genre}`)
    .then(r => r.json())
    .then(data => {
      document.getElementById('compare-content').innerHTML = renderTable(data.results, uid, genre);
    }).catch(() => {
      document.getElementById('compare-content').innerHTML = '<div class="empty">Error fetching data.</div>';
    });
}

// --------------------------------------------------------- TABLE RENDERER (shared by live + compare)
function renderTable(results, userId, genre) {
  if (!results.length) return '<div class="empty">No servers yet.</div>';
  const rows = results.map(r => {
    const dot = `<span class="dot ${r.status === 'online' ? 'online' : 'offline'}"></span>`;
    if (r.status !== 'online') {
      return `<tr><td>${dot}</td><td class="name">${r.name}</td><td colspan="6" style="color:#f85149">${r.status}</td></tr>`;
    }
    const recHtml = (r.recs || []).map((rec, i) =>
      `<li>${i+1}. ${rec.title}<span>${rec.score}</span></li>`
    ).join('');
    return `<tr>
      <td>${dot}</td>
      <td class="name">${r.name}</td>
      <td class="model">${r.cluster_model} / ${r.cluster_fill}</td>
      <td class="model">${r.rec_model}</td>
      <td class="cluster-label">${r.cluster_label}</td>
      <td><ul class="rec-list">${recHtml}</ul></td>
      <td>${r.forecast}</td>
    </tr>`;
  }).join('');

  return `<table>
    <thead><tr>
      <th></th><th>Student</th><th>Cluster model / fill</th><th>Rec model</th>
      <th style="text-align:center">Cluster<br><small>user ${userId}</small></th>
      <th>Top recs for user ${userId}</th>
      <th>${genre} forecast</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// -------------------------------------------------------- ENSEMBLE TAB
function loadServers() {
  document.getElementById('server-grid').innerHTML = '<div class="loading">Loading…</div>';
  fetch('/servers').then(r => r.json()).then(data => {
    if (!data.servers.length) {
      document.getElementById('server-grid').innerHTML = '<div class="empty">No servers in sheet yet.</div>';
      return;
    }
    document.getElementById('server-grid').innerHTML = data.servers.map(s => `
      <label class="server-chip ${s.online ? 'online' : ''}">
        <input type="checkbox" class="server-check" value="${s.url}" ${s.online ? 'checked' : ''}>
        <span class="dot ${s.online ? 'online' : 'offline'}"></span>
        ${s.name}
      </label>
    `).join('');
  });
}

function runEnsemble() {
  const uid   = parseInt(document.getElementById('ensemble-uid').value) || 42;
  const n     = parseInt(document.getElementById('ensemble-n').value)   || 10;
  const urls  = [...document.querySelectorAll('.server-check:checked')].map(c => c.value);
  if (!urls.length) { alert('Select at least one server.'); return; }
  document.getElementById('ensemble-content').innerHTML = '<div class="loading">Running ensemble across ' + urls.length + ' servers…</div>';
  fetch('/ensemble', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: uid, urls, n}),
  }).then(r => r.json()).then(data => {
    const rows = (data.recommendations || []).map((rec, i) => {
      const barW = Math.round(rec.consensus * 80);
      return `<tr>
        <td style="color:#8b949e">${i+1}</td>
        <td>${rec.title}</td>
        <td class="score">${rec.avg_score}</td>
        <td>${rec.n_servers} / ${data.n_servers_queried}
          <span class="consensus-bar"><span class="consensus-fill" style="width:${barW}px"></span></span>
        </td>
      </tr>`;
    }).join('');
    document.getElementById('ensemble-content').innerHTML = `
      <p style="color:#8b949e;font-size:0.82rem;margin-bottom:0.8rem;">
        Consensus recommendations for user <strong style="color:#f0f6fc">${data.user_id}</strong>
        across <strong style="color:#f0f6fc">${data.n_servers_queried}</strong> servers.
        Sorted by how many servers recommended the movie, then by average predicted rating.
      </p>
      <table>
        <thead><tr><th>#</th><th>Movie</th><th>Avg score</th><th>Server agreement</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }).catch(() => {
    document.getElementById('ensemble-content').innerHTML = '<div class="empty">Error running ensemble.</div>';
  });
}

// ------------------------------------------------------ RATE & REC TAB
function loadMovies() {
  if (document.getElementById('movie-grid').querySelector('.movie-card')) return; // already loaded
  fetch('/sample_movies').then(r => r.json()).then(movies => {
    document.getElementById('movie-grid').innerHTML = movies.map(m => `
      <div class="movie-card">
        <div class="title">${m.title}</div>
        <div class="stars">
          ${[1,2,3,4,5].map(s =>
            `<button class="star-btn" data-movie="${m.movie_id}" data-score="${s}" onclick="setRating(${m.movie_id}, ${s})">★</button>`
          ).join('')}
        </div>
      </div>
    `).join('');
  });
}

function setRating(movieId, score) {
  ratings[movieId] = score;
  document.querySelectorAll(`.star-btn[data-movie="${movieId}"]`).forEach(btn => {
    btn.classList.toggle('lit', parseInt(btn.dataset.score) <= score);
  });
  document.getElementById('rating-count').textContent = Object.keys(ratings).length + ' movies rated';
}

function clearRatings() {
  Object.keys(ratings).forEach(k => delete ratings[k]);
  document.querySelectorAll('.star-btn').forEach(b => b.classList.remove('lit'));
  document.getElementById('rating-count').textContent = '0 movies rated';
  document.getElementById('tryit-content').innerHTML = '';
}

function runTryIt() {
  if (!Object.keys(ratings).length) { alert('Rate at least one movie first.'); return; }
  document.getElementById('tryit-content').innerHTML = '<div class="loading">Querying all servers…</div>';
  fetch('/try_it', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ratings, n: 5}),
  }).then(r => r.json()).then(data => {
    const cards = (data.results || []).map(r => {
      if (r.status !== 'ok') {
        return `<div class="try-result-card">
          <div class="header"><strong>${r.name}</strong><span style="color:#f85149">${r.status}</span></div>
        </div>`;
      }
      const recs = (r.recommendations || []).map((rec, i) =>
        `<li>${i+1}. ${rec.title}<span class="score" style="color:#3fb950;margin-left:6px">${rec.predicted_rating}</span></li>`
      ).join('');
      return `<div class="try-result-card">
        <div class="header">
          <strong>${r.name}</strong>
          <span class="neighbour">nearest neighbour: user <span>${r.nearest_user}</span> (sim ${r.similarity})</span>
        </div>
        <ul class="rec-list">${recs}</ul>
      </div>`;
    }).join('');
    document.getElementById('tryit-content').innerHTML = cards ||
      '<div class="empty">No servers responded.</div>';
  }).catch(() => {
    document.getElementById('tryit-content').innerHTML = '<div class="empty">Error contacting servers.</div>';
  });
}

// ----------------------------------------------------------------- INIT
startLive();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getflix Session 8 dashboard")
    parser.add_argument("--port",     type=int, default=5055)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--user-id",  type=int, default=42)
    parser.add_argument("--genre",    default="Drama")
    parser.add_argument("--timeout",  type=int, default=4)
    args = parser.parse_args()

    CONFIG["user_id"]  = args.user_id
    CONFIG["genre"]    = args.genre
    CONFIG["timeout"]  = args.timeout
    CONFIG["interval"] = args.interval

    port = int(os.environ.get("PORT", args.port))
    print(f"\n  Getflix Session 8 Dashboard")
    print(f"  Open http://localhost:{port} and project that tab.\n")
    app.run(host="0.0.0.0", port=port, debug=False)

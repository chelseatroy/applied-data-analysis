"""
Generate monitoring.html from logs/requests.jsonl.

Reads every prediction the Flask app has served and produces a self-contained
dashboard showing request volume, model usage, and a basic data-drift signal
(predicted rating distribution vs. training distribution).

Run any time to refresh:
    python generate_monitoring.py
"""
import json
import os
from collections import Counter, defaultdict

LOG_FILE  = os.path.join(os.path.dirname(__file__), "logs", "requests.jsonl")
CACHE_REC = os.path.join(os.path.dirname(__file__), "recommender_data.json")
OUT_FILE  = os.path.join(os.path.dirname(__file__), "monitoring.html")


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def load_entries():
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_training_distribution():
    """Rating counts from the training data (1-5 stars), for drift comparison."""
    if not os.path.exists(CACHE_REC):
        return {}
    with open(CACHE_REC) as f:
        blob = json.load(f)
    return blob.get("__meta__", {}).get("rating_counts", {})


def summarise(entries):
    if not entries:
        return None

    endpoints = Counter(e["endpoint"] for e in entries)
    timestamps = sorted(e["timestamp"] for e in entries)

    # Hourly buckets  "2026-05-04T14"
    hourly = Counter(e["timestamp"][:13] for e in entries)
    hours_sorted = sorted(hourly)

    # /cluster
    cluster_entries = [e for e in entries if e["endpoint"] == "/cluster"]
    cluster_labels  = Counter(
        str(e["response"].get("cluster")) for e in cluster_entries
        if e.get("response")
    )
    cluster_users = Counter(
        e["params"].get("user_id", "?") for e in cluster_entries
    )

    # /recommend
    rec_entries = [e for e in entries if e["endpoint"] == "/recommend"]
    predicted_ratings = []
    rec_users = Counter()
    for e in rec_entries:
        uid = e["params"].get("user_id", "?")
        rec_users[uid] += 1
        recs = (e.get("response") or {}).get("recommendations", [])
        predicted_ratings.extend(r["predicted_rating"] for r in recs)

    # Bucket predicted ratings into 0.5-wide bins (0.5, 1.0, ..., 5.0)
    pred_buckets = defaultdict(int)
    for r in predicted_ratings:
        bucket = round(round(r * 2) / 2, 1)
        pred_buckets[str(bucket)] += 1

    # /forecast
    fc_entries = [e for e in entries if e["endpoint"] == "/forecast"]
    fc_genres = Counter(e["params"].get("genre", "?") for e in fc_entries)

    return {
        "total_requests": len(entries),
        "first_request":  timestamps[0] if timestamps else "",
        "last_request":   timestamps[-1] if timestamps else "",
        "endpoints":      dict(endpoints),
        "hourly":         {"hours": hours_sorted, "counts": [hourly[h] for h in hours_sorted]},
        "cluster": {
            "total":  len(cluster_entries),
            "labels": dict(cluster_labels),
            "top_users": dict(cluster_users.most_common(10)),
        },
        "recommend": {
            "total":          len(rec_entries),
            "n_predictions":  len(predicted_ratings),
            "pred_buckets":   dict(pred_buckets),
            "top_users":      dict(rec_users.most_common(10)),
        },
        "forecast": {
            "total":  len(fc_entries),
            "genres": dict(fc_genres),
        },
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Getflix Monitoring Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 1rem 2rem; }
  header h1 { font-size: 1.3rem; color: #58a6ff; }
  header p  { font-size: 0.84rem; color: #8b949e; margin-top: 0.2rem; }
  main { max-width: 960px; margin: 2rem auto; padding: 0 1.5rem 4rem; }
  h2 { font-size: 1.1rem; color: #f0f6fc; margin-bottom: 0.8rem; }
  h3 { font-size: 0.92rem; color: #8b949e; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
  p, li { color: #c9d1d9; line-height: 1.7; font-size: 0.9rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin: 1.2rem 0; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.2rem 0; }
  @media (max-width: 640px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem 1.4rem; }
  .stat-num { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .stat-lbl { font-size: 0.78rem; color: #8b949e; margin-top: 0.2rem; }
  .section { margin: 2rem 0; }
  .bar-chart { margin: 0.5rem 0; }
  .bar-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.83rem; }
  .bar-label { width: 110px; text-align: right; color: #8b949e; flex-shrink: 0; font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; background: #21262d; border-radius: 3px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .bar-value { width: 45px; color: #8b949e; font-size: 0.78rem; flex-shrink: 0; }
  .drift-note { font-size: 0.82rem; color: #8b949e; margin-top: 0.8rem; padding: 0.6rem 0.8rem;
    background: #21262d; border-radius: 5px; border-left: 3px solid #58a6ff; }
  .no-data { color: #8b949e; font-style: italic; font-size: 0.88rem; }
  .good  { color: #7ee787; }
  .warn  { color: #e3b341; }
  .ts-svg { width: 100%; height: 120px; overflow: visible; display: block; }
  code { color: #7ee787; font-size: 0.85rem; }
</style>
</head>
<body>
<header>
  <h1>Getflix — Inference Monitor</h1>
  <p id="header-sub"></p>
</header>
<main>

<div id="no-data-msg" style="display:none">
  <div class="card" style="margin-top:2rem;text-align:center;padding:2.5rem">
    <p style="font-size:1rem;color:#8b949e">No requests logged yet.</p>
    <p style="margin-top:0.6rem;font-size:0.85rem;color:#8b949e">
      Start the Flask app (<code>python flask_app/app.py</code>), make some predictions
      via <code>flask_app/static/index.html</code>, then regenerate this page.
    </p>
  </div>
</div>

<div id="dashboard">
  <!-- Summary stats -->
  <div class="grid-3" id="summary-stats"></div>

  <!-- Request timeline -->
  <div class="section">
    <h2>Request Timeline</h2>
    <div class="card">
      <svg class="ts-svg" id="timeline-svg" viewBox="0 0 880 100"></svg>
      <div id="timeline-legend" style="font-size:0.78rem;color:#8b949e;margin-top:0.4rem"></div>
    </div>
  </div>

  <!-- Endpoint breakdown -->
  <div class="section">
    <h2>By Endpoint</h2>
    <div class="grid-3" id="endpoint-cards"></div>
  </div>

  <!-- Recommender drift -->
  <div class="section" id="rec-section">
    <h2>Recommender — Predicted Rating Distribution</h2>
    <div class="card">
      <h3>Served predictions vs. training data</h3>
      <div class="grid-2">
        <div>
          <p style="font-size:0.8rem;color:#8b949e;margin-bottom:0.5rem">Predicted ratings (live)</p>
          <div class="bar-chart" id="pred-dist"></div>
        </div>
        <div>
          <p style="font-size:0.8rem;color:#8b949e;margin-bottom:0.5rem">Actual ratings (training data)</p>
          <div class="bar-chart" id="train-dist"></div>
        </div>
      </div>
      <div class="drift-note" id="drift-note"></div>
    </div>
  </div>

  <!-- Per-endpoint detail -->
  <div class="section">
    <h2>Detail</h2>
    <div class="grid-2">
      <div class="card" id="cluster-detail">
        <h3>Cluster assignments</h3>
        <div class="bar-chart" id="cluster-labels"></div>
      </div>
      <div class="card" id="forecast-detail">
        <h3>Forecast queries by genre</h3>
        <div class="bar-chart" id="fc-genres"></div>
      </div>
    </div>
    <div class="card" style="margin-top:1rem" id="users-detail">
      <h3>Most-queried users (recommender)</h3>
      <div class="bar-chart" id="top-users"></div>
    </div>
  </div>
</div>

</main>
<script>
const STATS  = <<<MONITORING_STATS>>>;
const TRAIN  = <<<TRAINING_DIST>>>;

function barChart(containerId, data, color, maxVal) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const entries = Object.entries(data).sort((a,b) => b[1]-a[1]);
  if (!entries.length) { el.innerHTML = '<p class="no-data">No data</p>'; return; }
  const max = maxVal || Math.max(...entries.map(e => e[1]));
  el.innerHTML = entries.map(([k,v]) => `
    <div class="bar-row">
      <div class="bar-label" title="${k}">${k}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${max ? (v/max*100).toFixed(1) : 0}%;background:${color}"></div>
      </div>
      <div class="bar-value">${v.toLocaleString()}</div>
    </div>`).join("");
}

function renderDashboard() {
  if (!STATS) {
    document.getElementById("no-data-msg").style.display = "block";
    document.getElementById("dashboard").style.display = "none";
    return;
  }

  document.getElementById("header-sub").textContent =
    STATS.total_requests + " requests · " +
    STATS.first_request.slice(0,16).replace("T"," ") + " → " +
    STATS.last_request.slice(0,16).replace("T"," ");

  // Summary stats
  const ep = STATS.endpoints;
  document.getElementById("summary-stats").innerHTML = [
    { num: STATS.total_requests.toLocaleString(), lbl: "Total requests" },
    { num: Object.keys(ep).length, lbl: "Endpoints active" },
    { num: STATS.recommend.n_predictions.toLocaleString(), lbl: "Predictions served" },
  ].map(s => `<div class="card"><div class="stat-num">${s.num}</div><div class="stat-lbl">${s.lbl}</div></div>`).join("");

  // Timeline
  renderTimeline();

  // Endpoint cards
  const epInfo = {
    "/cluster":   { label: "Cluster",     color: "#58a6ff", count: STATS.cluster.total },
    "/recommend": { label: "Recommend",   color: "#7ee787", count: STATS.recommend.total },
    "/forecast":  { label: "Forecast",    color: "#e3b341", count: STATS.forecast.total },
  };
  document.getElementById("endpoint-cards").innerHTML = Object.entries(epInfo).map(([path, info]) => `
    <div class="card">
      <div class="stat-num" style="color:${info.color}">${info.count}</div>
      <div class="stat-lbl">${info.label} requests</div>
      <div style="font-size:0.78rem;color:#8b949e;margin-top:0.3rem"><code>${path}</code></div>
    </div>`).join("");

  // Predicted vs training distribution
  barChart("pred-dist",  STATS.recommend.pred_buckets, "#7ee787");
  const trainFormatted = {};
  for (const [star, cnt] of Object.entries(TRAIN)) {
    trainFormatted[star + " ★"] = cnt;
  }
  barChart("train-dist", trainFormatted, "#58a6ff");

  // Drift note
  const predEntries = Object.entries(STATS.recommend.pred_buckets);
  const driftEl = document.getElementById("drift-note");
  if (predEntries.length === 0) {
    driftEl.textContent = "No recommender predictions logged yet.";
  } else {
    const predMean = predEntries.reduce((s,[k,v]) => s + parseFloat(k)*v, 0) /
                     predEntries.reduce((s,[,v]) => s + v, 0);
    const trainEntries = Object.entries(TRAIN);
    const trainMean = trainEntries.reduce((s,[k,v]) => s + parseFloat(k)*v, 0) /
                      trainEntries.reduce((s,[,v]) => s + v, 0);
    const diff = Math.abs(predMean - trainMean).toFixed(2);
    const cls  = parseFloat(diff) < 0.3 ? "good" : "warn";
    driftEl.innerHTML =
      `Mean predicted rating: <strong>${predMean.toFixed(2)}</strong> &nbsp;·&nbsp; ` +
      `Mean training rating: <strong>${trainMean.toFixed(2)}</strong> &nbsp;·&nbsp; ` +
      `Difference: <strong class="${cls}">${diff}</strong>. ` +
      (parseFloat(diff) < 0.3
        ? "Predicted ratings are close to the training distribution — no obvious drift."
        : "Predicted ratings are drifting from the training distribution. Consider retraining or checking your model.");
  }

  // Cluster labels
  barChart("cluster-labels", STATS.cluster.labels, "#58a6ff");

  // Forecast genres
  barChart("fc-genres", STATS.forecast.genres, "#e3b341");

  // Top users (recommender)
  barChart("top-users", STATS.recommend.top_users, "#7ee787");
}

function renderTimeline() {
  const { hours, counts } = STATS.hourly;
  if (!hours.length) return;
  const W=880, H=100, PL=8, PR=8, PT=8, PB=20;
  const n = hours.length;
  const maxC = Math.max(...counts, 1);
  const barW = Math.max(2, (W-PL-PR)/n - 1);
  const toX = i => PL + i * (W-PL-PR)/n;
  const toH = v => (v/maxC) * (H-PT-PB);

  let svg = counts.map((c,i) => {
    const x = toX(i);
    const h = toH(c);
    return `<rect x="${x.toFixed(1)}" y="${(H-PB-h).toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="#58a6ff" fill-opacity="0.8"/>`;
  }).join("");

  // Label every Nth hour
  const step = Math.max(1, Math.floor(n/8));
  svg += hours.filter((_,i) => i%step===0).map((h,j) => {
    const i = j * step;
    return `<text x="${(toX(i)+barW/2).toFixed(1)}" y="${H-4}" fill="#8b949e" font-size="9" text-anchor="middle">${h.slice(5)}</text>`;
  }).join("");

  document.getElementById("timeline-svg").innerHTML = svg;
  document.getElementById("timeline-legend").textContent =
    `${n} hour${n===1?"":"s"} of activity · peak: ${Math.max(...counts)} requests/hour`;
}

renderDashboard();
</script>
</body>
</html>
"""


def generate(log_path=LOG_FILE, out_path=OUT_FILE):
    entries = load_entries() if log_path == LOG_FILE else _load_entries_from(log_path)
    training_dist = load_training_distribution()
    stats = summarise(entries)

    stats_json = json.dumps(stats, separators=(",", ":"))
    train_json = json.dumps(training_dist, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("<<<MONITORING_STATS>>>", stats_json)
    html = html.replace("<<<TRAINING_DIST>>>", train_json)

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Written: {out_path}")
    if stats:
        print(f"  {stats['total_requests']} requests logged")
        print(f"  Endpoints: {stats['endpoints']}")
    else:
        print("  No log entries found — dashboard shows placeholder")


def _load_entries_from(path):
    entries = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries


if __name__ == "__main__":
    generate()

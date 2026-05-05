"""
Local inference server for the Getflix Adventure walkthroughs.

Start with:  python flask_app/app.py
Then open:   flask_app/static/index.html  in your browser.

Models are loaded lazily from flask_app/models/ on first request.
Run a Metaflow flow (clustering_flow.py, recommender_flow.py, timeseries_flow.py)
to produce the pickles, then restart the server.

Every successful prediction is appended to logs/requests.jsonl automatically.
Run  python generate_monitoring.py  to turn those logs into a dashboard.
"""
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "ml-100k")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE   = os.path.join(LOGS_DIR, "requests.jsonl")

_cache = {}

_ML_ENDPOINTS = {"/cluster", "/recommend", "/forecast"}


# ---------------------------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------------------------

def load_model(name):
    if name not in _cache:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


def load_ratings():
    if "ratings" not in _cache:
        path = os.path.join(DATA_DIR, "u.data")
        _cache["ratings"] = pd.read_csv(
            path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
        )
    return _cache["ratings"]


def load_movies():
    if "movies" not in _cache:
        path = os.path.join(DATA_DIR, "u.item")
        cols = ["movie_id", "title"] + [f"g{i}" for i in range(19)]
        _cache["movies"] = pd.read_csv(
            path, sep="|", names=cols, encoding="latin-1"
        )[["movie_id", "title"]]
    return _cache["movies"]


# ---------------------------------------------------------------------------
# Request logging — fires automatically after every successful ML response
# ---------------------------------------------------------------------------

@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def _log_request(response):
    if request.path not in _ML_ENDPOINTS or response.status_code != 200:
        return response
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint":  request.path,
            "params":    dict(request.args),
            "response":  response.get_json(),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging must never break the response
    return response


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Health check — shows what models are currently deployed
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """
    Returns metadata about the models currently deployed in flask_app/models/.
    Useful for knowing which Metaflow run is live without inspecting pickle files.
    """
    result = {}
    for name in ("clustering", "recommender", "timeseries"):
        pkl = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(pkl):
            result[name] = {"status": "missing"}
            continue
        trained_at = datetime.fromtimestamp(os.path.getmtime(pkl)).isoformat()
        bundle = load_model(name)
        if bundle is None:
            result[name] = {"status": "error", "trained_at": trained_at}
            continue
        info = {"status": "ok", "trained_at": trained_at}
        if name == "clustering":
            info["fill_strategy"] = bundle.get("fill_strategy")
            info["model_type"]    = type(bundle.get("model")).__name__
            info["n_users"]       = len(bundle.get("user_ids", []))
        elif name == "recommender":
            info["algo_type"] = type(bundle.get("algo")).__name__
        elif name == "timeseries":
            info["genres"] = sorted(bundle.get("forecasts", {}).keys())
        result[name] = info
    return jsonify(result)


# ---------------------------------------------------------------------------
# Clustering endpoint
# ---------------------------------------------------------------------------

@app.route("/cluster")
def cluster():
    """
    ?user_id=<int>
    Returns the cluster label for that user.
    """
    bundle = load_model("clustering")
    if bundle is None:
        return jsonify({"error": "clustering.pkl not found — run clustering_flow.py first"}), 503

    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "user_id parameter required"}), 400

    user_ids = bundle["user_ids"]
    if user_id not in user_ids:
        return jsonify({"error": f"user {user_id} not in training data"}), 404

    ratings = load_ratings()
    pivot = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
    fill = bundle["fill_strategy"]
    if fill == "zero":
        filled = pivot.fillna(0)
    else:
        filled = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)

    scaler = bundle["scaler"]
    scaled = scaler.transform(filled)
    idx = list(filled.index).index(user_id)
    user_vec = scaled[idx].reshape(1, -1)

    model = bundle["model"]
    if hasattr(model, "predict"):
        label = int(model.predict(user_vec)[0])
    else:
        label = int(model.labels_[idx])

    return jsonify({"user_id": user_id, "cluster": label})


# ---------------------------------------------------------------------------
# Recommender endpoint
# ---------------------------------------------------------------------------

@app.route("/recommend")
def recommend():
    """
    ?user_id=<int>&n=<int>
    Returns top-n movie recommendations for the user.
    """
    bundle = load_model("recommender")
    if bundle is None:
        return jsonify({"error": "recommender.pkl not found — run recommender_flow.py first"}), 503

    user_id = request.args.get("user_id", type=int)
    n = request.args.get("n", default=10, type=int)
    if user_id is None:
        return jsonify({"error": "user_id parameter required"}), 400

    algo    = bundle["algo"]
    movies  = load_movies()
    ratings = load_ratings()
    all_movie_ids = ratings["movie_id"].unique()
    rated      = set(ratings[ratings["user_id"] == user_id]["movie_id"])
    candidates = [m for m in all_movie_ids if m not in rated]

    preds = [(m, algo.predict(user_id, m).est) for m in candidates]
    preds.sort(key=lambda x: -x[1])

    results = []
    for movie_id, score in preds[:n]:
        title_row = movies[movies["movie_id"] == movie_id]
        title = title_row["title"].values[0] if len(title_row) else f"Movie {movie_id}"
        results.append({
            "movie_id":        int(movie_id),
            "title":           title,
            "predicted_rating": round(score, 2),
        })

    return jsonify({"user_id": user_id, "recommendations": results})


# ---------------------------------------------------------------------------
# Forecast endpoint
# ---------------------------------------------------------------------------

@app.route("/forecast")
def forecast():
    """
    ?genre=<str>&steps=<int>
    Returns ARIMA forecast for average weekly ratings in the given genre.
    """
    bundle = load_model("timeseries")
    if bundle is None:
        return jsonify({"error": "timeseries.pkl not found — run timeseries_flow.py first"}), 503

    genre = request.args.get("genre", "Drama")
    steps = request.args.get("steps", default=6, type=int)

    forecasts = bundle.get("forecasts", {})
    if genre not in forecasts:
        return jsonify({
            "error": f"No forecast for genre '{genre}'. Available: {list(forecasts.keys())}"
        }), 404

    fc = forecasts[genre]
    return jsonify({
        "genre":           genre,
        "steps":           steps,
        "forecast":        fc["forecast"][:steps],
        "conf_int_lower":  fc["conf_int_lower"][:steps],
        "conf_int_upper":  fc["conf_int_upper"][:steps],
        "periods":         fc["periods"][:steps],
    })


if __name__ == "__main__":
    print("Starting Getflix Adventure inference server...")
    print("Open flask_app/static/index.html in your browser.")
    print("Check /health to see which models are deployed.")
    app.run(debug=True, port=5050)

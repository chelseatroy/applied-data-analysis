"""
Penguin species inference server.

Endpoints:
    POST /predict   — classify a penguin from its measurements
    GET  /health    — metadata about the currently deployed model
    GET  /reload    — hot-swap the model without restarting the server
"""
import os
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder="static", static_url_path="")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "penguin_model.pkl")

_model_cache = None


@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            return None
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache


# ── /predict ──────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    m = load_model()
    if m is None:
        return jsonify({
            "error": "No model deployed yet. Run penguin_flow.py first.",
        }), 503

    data = request.get_json(silent=True) or {}
    required = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm",
                "body_mass_g", "island", "sex"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = [
            float(data["bill_length_mm"]),
            float(data["bill_depth_mm"]),
            float(data["flipper_length_mm"]),
            float(data["body_mass_g"]),
            float(m["island_map"][data["island"]]),
            float(m["sex_map"][data["sex"]]),
        ]
    except KeyError as e:
        return jsonify({"error": f"Unknown value for {e}. "
                                  f"Islands: {list(m['island_map'])}  "
                                  f"Sex: {list(m['sex_map'])}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Bad numeric value: {e}"}), 400

    probs    = m["model"].predict_proba([features])[0]
    pred_idx = int(probs.argmax())
    species  = m["species"]

    return jsonify({
        "species":      species[pred_idx],
        "confidence":   round(float(probs[pred_idx]), 3),
        "probabilities": {s: round(float(p), 3)
                          for s, p in zip(species, probs)},
        "model_depth":  m["max_depth"],
        "model_accuracy": round(m["accuracy"], 3),
        "trained_at":   m["trained_at"],
    })


# ── /health ───────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    m = load_model()
    if m is None:
        return jsonify({
            "status":  "no_model",
            "message": "Run penguin_flow.py first, then GET /reload.",
        })
    return jsonify({
        "status":    "ok",
        "max_depth": m["max_depth"],
        "accuracy":  round(m["accuracy"], 3),
        "n_train":   m["n_train"],
        "n_test":    m["n_test"],
        "n_dropped": m["n_dropped"],
        "trained_at": m["trained_at"],
    })


# ── /reload ───────────────────────────────────────────────────────────────────

@app.route("/reload")
def reload_model():
    """Force the server to re-read the pickle from disk without restarting."""
    global _model_cache
    _model_cache = None
    m = load_model()
    if m is None:
        return jsonify({"status": "no_model"})
    return jsonify({
        "status":    "reloaded",
        "max_depth": m["max_depth"],
        "accuracy":  round(m["accuracy"], 3),
        "trained_at": m["trained_at"],
    })


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Penguin inference server...")
    print("Open flask_app/static/index.html in your browser.")
    print("Check /health to see which model is deployed.")
    print(" * Running on http://127.0.0.1:5051")
    app.run(port=5051, debug=False)

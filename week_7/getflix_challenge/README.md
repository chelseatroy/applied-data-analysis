# Getflix Adventure

You're an ML engineer at a movie streaming service called Getflix. 
Your job is to build and deploy three machine learning models 
that power a movie recommendation site. By the end of this activity 
you'll have a live local web app serving real predictions from models
you trained yourself.

---

## Phase 1 — Start Your Environment

Run this **once** from the `getflix_challenge/` directory:

```bash
bash setup.sh
```

This creates a virtual environment in `.venv/`, installs all dependencies, and
downloads the MovieLens 100K dataset (~5 MB; please make sure you have the space on your machine for this). 
When it finishes:

```bash
source .venv/bin/activate
```

You need to activate the environment every time you open a new terminal. All
subsequent commands assume it is active. 

You'll know the venv is active because your command line prompt will start with .venv with parentheses around it.

---

## Phase 2 — Work Through the Walkthroughs

Open the data explorer first to get familiar with the dataset:

```bash
open explore.html
```

Then work through the three walkthroughs **in order**. Each one walks you
through some decisions an ML engineer makes when building that type of model —
fill strategy, algorithm family, hyperparameter choice — and explains the
tradeoffs along the way.

```bash
open clustering_walkthrough.html    # Step 1 of 3
open recommender_walkthrough.html   # Step 2 of 3
open timeseries_walkthrough.html    # Step 3 of 3
```

At the end of each walkthrough, the final screen shows a Metaflow command
pre-filled with your choices. Copy it; you'll run it in Phase 3.

---

## Phase 3 — Train Your Models

Run the command shown at the end of each walkthrough. The commands look like
these (your choices will fill in the placeholders):

```bash
python clustering_flow.py run --fill zero --model_key kmeans_k5
python recommender_flow.py run --model_key svd_f50
python timeseries_flow.py run --transform raw --p 1 --d 1 --q 1
```

Each command is a [Metaflow](https://metaflow.org) pipeline — the same kind of
step-based, versioned pipeline used in production ML systems. You'll see each
step print as it runs. When a flow finishes, it saves a pickle to
`flask_app/models/`:

| Flow | Saves |
|------|-------|
| `clustering_flow.py` | `flask_app/models/clustering.pkl` |
| `recommender_flow.py` | `flask_app/models/recommender.pkl` |
| `timeseries_flow.py` | `flask_app/models/timeseries.pkl` |

Metaflow records every run. To see your history at any time:

```python
from metaflow import Flow
for run in Flow('ClusteringFlow'):
    print(run.id, run.successful)
```

(Replace `ClusteringFlow` with `RecommenderFlow` or `TimeseriesFlow` as needed.)

> **Note:** The app ships with precomputed default models, so it works before
> you run any flows. Running your own flow replaces the default with the model
> you chose.

---

## Phase 4 — Start the App

In a separate terminal (with the environment activated), start the Flask server:

```bash
python flask_app/app.py
```

You should see:

```
Starting Getflix Adventure inference server...
Open flask_app/static/index.html in your browser.
Check /health to see which models are deployed.
 * Running on http://127.0.0.1:5050
```

Leave this terminal running. Then open the frontend:

```bash
open flask_app/static/index.html
```

---

## Phase 5 — Try the Endpoints

The frontend at `flask_app/static/index.html` gives you a GUI for all three
models. You can also query the endpoints directly.

### Cluster — which taste group does a user belong to?

```
GET http://localhost:5050/cluster?user_id=42
```

```json
{"cluster": 2, "user_id": 42}
```

### Recommend — top movies for a user they haven't seen

```
GET http://localhost:5050/recommend?user_id=42&n=5
```

```json
{
  "user_id": 42,
  "recommendations": [
    {"movie_id": 318, "title": "Schindler's List (1993)", "predicted_rating": 4.71},
    ...
  ]
}
```

### Forecast — predicted average rating trend for a genre

```
GET http://localhost:5050/forecast?genre=Drama&steps=4
```

```json
{
  "genre": "Drama",
  "forecast": [3.82, 3.79, 3.84, 3.81],
  "conf_int_lower": [3.51, 3.44, 3.47, 3.42],
  "conf_int_upper": [4.13, 4.14, 4.21, 4.20],
  "periods": ["1998-05-03", "1998-05-10", "1998-05-17", "1998-05-24"]
}
```

Available genres: `Drama`, `Comedy`, `Action`, `Thriller`, `Romance`.

### Health — what models are currently deployed?

```
GET http://localhost:5050/health
```

```json
{
  "clustering":  {"status": "ok", "model_type": "AgglomerativeClustering", "fill_strategy": "zero", "n_users": 943, "trained_at": "..."},
  "recommender": {"status": "ok", "algo_type": "SVD", "trained_at": "..."},
  "timeseries":  {"status": "ok", "genres": ["Action", "Comedy", "Drama", "Romance", "Thriller"], "trained_at": "..."}
}
```

This tells you which Metaflow run is currently live without opening any files.
If you train a new model and restart the server, `/health` will reflect the
change.

---

## Phase 6 — Monitoring

Every successful prediction is automatically logged to `logs/requests.jsonl`.
After making some requests through the frontend or via the endpoints above,
generate the monitoring dashboard:

```bash
python generate_monitoring.py
open monitoring.html
```

The dashboard shows:

- **Request timeline** — requests per hour across all endpoints
- **Endpoint breakdown** — how many calls each model has served
- **Predicted vs. training distribution** — are the recommender's predicted
  ratings drifting away from the distribution of actual ratings in the training
  data? A large gap is a signal that something may have changed.
- **Most-queried users and genres** — who and what is driving traffic

Regenerate it any time to refresh:

```bash
python generate_monitoring.py && open monitoring.html
```

### Error Analysis and One Improvement

Pick either the **clustering model** or the **recommender model** (you do not have to do both). Your goal is to understand where it currently falls short, choose one change to try, and evaluate what happened.

**Step 1 — Analyze the errors.**

Look at what the model is actually doing wrong. Some starting points:

- For the **recommender**: examine the residuals from the flow's evaluation
  output. Which users have the highest prediction error? Look at their rating
  histories — are there patterns (sparse raters, genre specialists, contrarians
  who rate differently from everyone else)?

- For **clustering**: look at which users end up in each cluster. Do the
  clusters feel coherent when you inspect the top-rated genres for users in each
  group? Are there users whose cluster assignment seems wrong given their
  history?

You are not limited to these angles. The goal is to look at the data carefully
enough that you can point to something specific.

Commit your error analysis work (scripts, notebooks, output files, whatever you
produce) so it is visible in your git history.

**Step 2 — Choose one thing to try.**

Based on what you found, pick one change to the model — a different
hyperparameter, a different fill strategy, a preprocessing decision, a different
algorithm — that you believe is motivated by the errors you saw. You do not need
to be right. You need to be able to explain *why* the thing you chose was a
reasonable response to what you observed.

Train the new model via the appropriate Metaflow flow and deploy it to the app.

**Step 3 — Evaluate.**

Compare the results before and after. Did the metric improve? Did it stay the
same? Did something else change (e.g., the recommendations look more coherent
even if RMSE didn't drop)?

**In `homework_7_reflections.md`, write:**

- What errors or patterns you found in your analysis
- What change you chose to try, and why that change was a reasonable response to
  what you saw
- What happened when you tried it — and what you think that means

---

### Model Versioning and Rollback

Right now, running a Metaflow flow overwrites `flask_app/models/clustering.pkl`
(or `recommender.pkl` or `timeseries.pkl`). There is no way to recover a
previous model without re-training.

**Your task:**

1. Modify the relevant flow(s) so that each training run saves a **versioned**
   model file alongside the generic one. The versioning scheme is up to you —
   timestamp, run ID, incrementing number, something else — but it should be
   consistent and unambiguous.

2. Add a `/rollback` endpoint to `flask_app/app.py`. It should load the
   previous version of the model (however you define "previous") and make it the
   active one, without restarting the server. The response should confirm which
   version is now active.

**In `homework_7_reflections.md`, write:**

- What versioning scheme you chose and why
- Why a rollback endpoint matters in a production system — what scenario would
  cause you to use it?


---

## File Map

```
getflix_challenge/
  setup.sh                        Phase 1 — one-time environment setup
  requirements.txt                all dependencies

  explore.html                    Phase 2 — pre-activity data explorer
  clustering_walkthrough.html     Phase 2 — Step 1 of 3
  recommender_walkthrough.html    Phase 2 — Step 2 of 3
  timeseries_walkthrough.html     Phase 2 — Step 3 of 3

  clustering_flow.py              Phase 3 — trains clustering model
  recommender_flow.py             Phase 3 — trains recommender model
  timeseries_flow.py              Phase 3 — trains time series forecasts

  flask_app/
    app.py                        Phase 4 — inference server (port 5050)
    static/index.html             Phase 5 — web frontend
    models/                       Phase 3 output — trained pickles land here

  generate_monitoring.py          Phase 6 — builds monitoring.html from logs
  monitoring.html                 Phase 6 — regenerated dashboard
  logs/requests.jsonl             Phase 6 — auto-written by the Flask server

  data/ml-100k/                   MovieLens 100K dataset (downloaded by setup.sh)

  homework_7_reflections.md       Homework — your error analysis and design decisions
```

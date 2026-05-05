# Penguin Species Classifier — Operational ML Activity

You're an ML engineer. Your job is to train a decision tree classifier, deploy it
behind an inference server, query it, then swap in a new model and watch the
predictions change.

By the end you'll have run a real Metaflow pipeline, served predictions from a
live Flask endpoint, and seen firsthand how a hyperparameter change affects what
a deployed model actually says.

---

## Phase 1 — Start Your Environment

Run once from the `penguins_operation/` directory:

```bash
bash setup.sh
source .venv/bin/activate
```

---

## Phase 2 — Train Your First Model

```bash
python penguin_flow.py run --max_depth 2
```

Watch what prints:
- How many rows were **dropped** due to missing values, and how many remain
- The **decision tree rules** — with depth 2 you can read the whole tree
- The **per-class accuracy** on the held-out test set

Metaflow records every run. See the full history any time:

```python
from metaflow import Flow
for run in Flow('PenguinFlow'):
    print(run.id, run.successful)
```

When the flow finishes it saves a pickle file to `flask_app/models/penguin_model.pkl`.

---

## Phase 3 — Start the Inference Server

In a separate terminal (with the environment activated):

```bash
python flask_app/app.py
```

You should see:
```
Starting Penguin inference server...
Open flask_app/static/index.html in your browser.
Check /health to see which model is deployed.
 * Running on http://127.0.0.1:5051
```

Leave this terminal running. Then open the frontend:

```bash
open flask_app/static/index.html
```

---

## Phase 4 — Query the Model

The frontend gives you a form for all six features. The **model badge** in the
header (click it) shows what's currently deployed — depth, accuracy, training
time — without opening any files.

You can also query the endpoints directly.

### Predict — classify a penguin from its measurements

```
POST http://localhost:5051/predict
Content-Type: application/json

{
  "bill_length_mm": 46.1,
  "bill_depth_mm": 13.2,
  "flipper_length_mm": 211,
  "body_mass_g": 4500,
  "island": "Biscoe",
  "sex": "female"
}
```

```json
{
  "species": "Gentoo",
  "confidence": 1.0,
  "probabilities": {"Adelie": 0.0, "Chinstrap": 0.0, "Gentoo": 1.0},
  "model_depth": 2,
  "model_accuracy": 0.913,
  "trained_at": "2026-05-04T20:00:00+00:00"
}
```

### Health — what model is deployed?

```
GET http://localhost:5051/health
```

```json
{
  "status": "ok",
  "max_depth": 2,
  "accuracy": 0.913,
  "n_train": 266,
  "n_test": 67,
  "n_dropped": 11,
  "trained_at": "..."
}
```

---

## Phase 5 — Retrain with a Different Depth

**Keep the Flask server running.** Back in your first terminal:

```bash
python penguin_flow.py run --max_depth 5
```

Notice:
- The **rules** are longer and more specific now
- The **test accuracy** may be higher — or the same, or lower

Once the flow finishes, tell the running server to pick up the new model:

```bash
curl http://localhost:5051/reload
```

Or just hit `/reload` in your browser. No restart required.

---

## Phase 6 — Compare

Use the frontend's **Prediction Log** table to make the same predictions under
both models. Interesting cases to try:

| Bill L | Bill D | Flipper | Mass | Island | Sex | Expected |
|--------|--------|---------|------|--------|-----|----------|
| 39.1 | 18.7 | 181 | 3750 | Torgersen | male | Adelie |
| 46.5 | 17.9 | 192 | 3500 | Dream | female | Chinstrap |
| 46.1 | 13.2 | 211 | 4500 | Biscoe | female | Gentoo |
| 42.0 | 17.0 | 195 | 3900 | Dream | male | ??? |

## Questions to Discuss:

Component overview:

  - What does the metaflow flow accomplish? What does the @step decorator appear to mean?
  - Which elements of the model training are configurable via arguments to the flow?
  - Can you think of some other model training options that aren't configurable in this flow?
  - What does the inference server do?
  - What does the `/health` endpoint in the inference server show you that you can't tell just from running the model?

On what the model is actually doing:

  - The depth=2 tree prints rules you can read in full. The depth=5 tree is
  harder to follow. If you had to explain a prediction to a non-technical
  stakeholder, which would you prefer, and what does that tell you about the
  tradeoff between interpretability and performance?
  - Island and sex are encoded as integers (Biscoe=0, Dream=1, Torgersen=2).
  What assumption does that encoding make that a decision tree doesn't care
  about but a linear model would?

On the train/test split:

  - The split uses stratify=y. Why does that matter here? What would happen to
  your evaluation if you didn't stratify and one class ended up underrepresented
   in the test set by chance?
  - 11 rows were dropped for missing values. Is that a problem? What factors
  would make you more or less worried about it?

On evaluation:

  - The classification report shows per-class precision and recall, not just
  overall accuracy. When does overall accuracy mislead you, and does it mislead
  you here?
  - You trained two models. The test accuracy might be higher for depth=5, the
  same, or even lower. What are the possible explanations for each of those
  outcomes?

On the operational design:

  - /health exists as a separate endpoint from /predict. What does it tell you
  that you couldn't get by just calling /predict? Why would you want that
  information separately?
  - /reload lets you hot-swap a model without restarting the server. When is
  that useful? Can you think of a scenario where you'd specifically not want to
  allow hot-swapping?
  - The pickle payload saves accuracy, n_train, trained_at, and other metadata
  alongside the model object itself. Why bundle metadata with the model? What
  goes wrong in a production system if you don't?

On the probabilities:

  - /predict returns both a species prediction and a probabilities dictionary.
  The prediction is just argmax of the probabilities. Can you think of a case
  where you'd want to use the full probability distribution rather than just the
   top prediction?

---

## File Map

```
penguins_operation/
  setup.sh                   Phase 1 — one-time environment setup
  requirements.txt           all dependencies

  penguin_flow.py            Phase 2 & 5 — Metaflow training pipeline

  flask_app/
    app.py                   Phase 3 — inference server (port 5051)
    static/index.html        Phase 4 — web frontend
    models/                  Phase 2 output — trained pickle lands here

  README.md                  this file
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify a penguin from its measurements |
| `/health` | GET | Metadata about the currently deployed model |
| `/reload` | GET | Hot-swap a new pickle without restarting Flask |

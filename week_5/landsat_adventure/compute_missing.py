#!/usr/bin/env python3
"""
compute_missing.py
Adds Naive Bayes preprocessing data, learning_curves, and (if imblearn is
installed) SMOTE results to landsat_data.json.

Run:  python compute_missing.py
      pip install imbalanced-learn  # for SMOTE, then re-run

- Uses n_jobs=1 throughout (no parallelism)
- No SVM anywhere — see the "Why No SVM?" note in the adventure HTML
- Writes cache after each combo so you can interrupt and resume
"""

import json
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

CACHE = "landsat_data.json"
CLASS_LABELS = [1, 2, 3, 4, 5, 7]
MODELS = ["lr", "knn", "dt", "nb", "rf"]

NB_VAR_SMOOTHING = [1e-9, 1e-7, 1e-5, 1e-3, 0.1]


def get_scaler(key):
    return {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": None}[key]


def make_model(key, params=None):
    p = params or {}
    if key == "lr":  return LogisticRegression(max_iter=2000, solver="saga", **p)
    if key == "knn": return KNeighborsClassifier(**p)
    if key == "dt":  return DecisionTreeClassifier(random_state=42, **p)
    if key == "nb":  return GaussianNB(**p)
    if key == "rf":  return RandomForestClassifier(random_state=42, **p)


def eval_result(yte, yp):
    cm = confusion_matrix(yte, yp, labels=CLASS_LABELS).tolist()
    pr, rc, _, _ = precision_recall_fscore_support(
        yte, yp, labels=CLASS_LABELS, zero_division=0)
    return {
        "score":     round(float((yp == yte).mean()), 4),
        "cm":        cm,
        "precision": [round(float(x), 3) for x in pr],
        "recall":    [round(float(x), 3) for x in rc],
    }


def save(blob):
    with open(CACHE, "w") as f:
        json.dump(blob, f)
    print("  → saved", flush=True)


print("Loading cache …")
with open(CACHE) as f:
    blob = json.load(f)

print("Loading dataset …")
ds = fetch_openml(name="satimage", version=1, as_frame=True)
X = ds.data.values.astype(float)
y = np.array([int(lbl.replace(".", "")) for lbl in ds.target.values])

# ── Naive Bayes preprocessing data (cv, test_cms, hp) ─────────────────────────
# The generator's compute_all() is only run when there's no cache at all.
# We add NB's data here directly so the generator can embed it in the HTML.
print("\n── Naive Bayes preprocessing data ──")
for scaler_key in ["standard", "minmax", "none"]:
    for stratify_key in ["yes", "no"]:
        pk = f"{scaler_key}_{stratify_key}"
        prep = blob["preprocessing"][pk]

        # CV scores
        changed = False
        for nf in [3, 5, 10]:
            if "nb" in prep["cv"].get(str(nf), {}):
                print(f"  {pk}/cv/{nf} nb already cached, skipping")
                continue
            strat_y = y if stratify_key == "yes" else None
            Xtr, _, ytr, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            Xtr_s = scaler.fit_transform(Xtr) if scaler else Xtr.copy()
            cv_obj = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
            scores = cross_val_score(make_model("nb"), Xtr_s, ytr, cv=cv_obj, scoring="accuracy")
            prep["cv"][str(nf)]["nb"] = {
                "mean": round(float(scores.mean()), 4),
                "std":  round(float(scores.std()),  4),
            }
            print(f"  {pk}/cv/{nf} nb done ({scores.mean():.3f})")
            changed = True

        # Test-set confusion matrix
        if "nb" not in prep.get("test_cms", {}):
            strat_y = y if stratify_key == "yes" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            if scaler:
                Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
            clf = make_model("nb"); clf.fit(Xtr_s, ytr)
            prep["test_cms"]["nb"] = confusion_matrix(yte, clf.predict(Xte_s), labels=CLASS_LABELS).tolist()
            print(f"  {pk}/test_cms nb done")
            changed = True

        # HP grid (var_smoothing only — 1D)
        if "nb" not in prep.get("hp", {}):
            strat_y = y if stratify_key == "yes" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            if scaler:
                Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
            results = []
            for vs in NB_VAR_SMOOTHING:
                clf = GaussianNB(var_smoothing=vs); clf.fit(Xtr_s, ytr)
                yp = clf.predict(Xte_s)
                acc = float((yp == yte).mean())
                cm = confusion_matrix(yte, yp, labels=CLASS_LABELS).tolist()
                pr, rc, _, _ = precision_recall_fscore_support(yte, yp, labels=CLASS_LABELS, zero_division=0)
                results.append({
                    "params":    {"var_smoothing": vs},
                    "score":     round(acc, 4),
                    "cm":        cm,
                    "precision": [round(float(x), 3) for x in pr],
                    "recall":    [round(float(x), 3) for x in rc],
                })
            best_idx = max(range(len(results)), key=lambda i: results[i]["score"])
            prep["hp"]["nb"] = {
                "results":   results,
                "best_idx":  best_idx,
                "x":         "var_smoothing",
                "x_vals":    NB_VAR_SMOOTHING,
                "hue":       None,
                "hue_vals":  None,
            }
            print(f"  {pk}/hp nb done (best var_smoothing={NB_VAR_SMOOTHING[best_idx]}, acc={results[best_idx]['score']:.3f})")
            changed = True

        if changed:
            save(blob)

# ── NB in improvements (ndmi and weighted) ────────────────────────────────────
print("\n── Naive Bayes in improvements ──")
GREY_LABELS = {3, 4, 7}

def add_ndmi(X_raw):
    cols = []
    for p in range(9):
        b3 = X_raw[:, p * 4 + 2].astype(float)
        b4 = X_raw[:, p * 4 + 3].astype(float)
        cols.append(((b3 - b4) / (b3 + b4 + 1e-8)).reshape(-1, 1))
    return np.hstack([X_raw, np.hstack(cols)])

for scaler_key in ["standard", "minmax", "none"]:
    for stratify_key in ["yes", "no"]:
        pk = f"{scaler_key}_{stratify_key}"
        imp = blob["improvements"][pk]
        changed = False

        # ndmi — NB can use engineered features fine
        if "nb" not in imp.get("ndmi", {}):
            strat_y = y if stratify_key == "yes" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            Xtr_aug = add_ndmi(Xtr); Xte_aug = add_ndmi(Xte)
            scaler = get_scaler(scaler_key)
            if scaler:
                Xtr_aug_s = scaler.fit_transform(Xtr_aug)
                Xte_aug_s = scaler.transform(Xte_aug)
            else:
                Xtr_aug_s, Xte_aug_s = Xtr_aug.copy(), Xte_aug.copy()
            clf = make_model("nb"); clf.fit(Xtr_aug_s, ytr)
            imp["ndmi"]["nb"] = eval_result(yte, clf.predict(Xte_aug_s))
            print(f"  {pk}/ndmi nb done ({imp['ndmi']['nb']['score']:.3f})")
            changed = True

        # weighted — GaussianNB doesn't support class_weight
        if "nb" not in imp.get("weighted", {}):
            imp["weighted"]["nb"] = {"unsupported": True}
            print(f"  {pk}/weighted nb → unsupported")
            changed = True

        if changed:
            save(blob)

# ── Learning curves ────────────────────────────────────────────────────────────
print("\n── Learning curves ──")
train_fracs = [0.10, 0.25, 0.50, 0.75, 1.00]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

if "learning_curves" not in blob:
    blob["learning_curves"] = {}

for scaler_key in ["standard", "minmax", "none"]:
    for stratify_key in ["yes", "no"]:
        pk = f"{scaler_key}_{stratify_key}"
        strat_y = y if stratify_key == "yes" else None
        Xtr, _, ytr, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat_y)
        scaler = get_scaler(scaler_key)
        Xtr_s = scaler.fit_transform(Xtr) if scaler else Xtr.copy()
        if pk not in blob["learning_curves"]:
            blob["learning_curves"][pk] = {}
        for mk in MODELS:
            if mk in blob["learning_curves"].get(pk, {}):
                print(f"  {pk}/{mk} already cached, skipping")
                continue
            print(f"  {pk}/{mk} …", end="", flush=True)
            sizes, tr_sc, val_sc = learning_curve(
                make_model(mk), Xtr_s, ytr,
                train_sizes=train_fracs, cv=cv,
                scoring="accuracy", n_jobs=1)
            blob["learning_curves"][pk][mk] = {
                "sizes":      [int(s) for s in sizes],
                "train_mean": [round(float(v), 4) for v in tr_sc.mean(axis=1)],
                "train_std":  [round(float(v), 4) for v in tr_sc.std(axis=1)],
                "val_mean":   [round(float(v), 4) for v in val_sc.mean(axis=1)],
                "val_std":    [round(float(v), 4) for v in val_sc.std(axis=1)],
            }
            print(" done")
            save(blob)

# ── SMOTE ──────────────────────────────────────────────────────────────────────
if not HAS_SMOTE:
    print("\nSMOTE skipped — imbalanced-learn not installed.")
    print("  pip install imbalanced-learn  then re-run this script.")
else:
    print("\n── SMOTE ──")
    for scaler_key in ["standard", "minmax", "none"]:
        for stratify_key in ["yes", "no"]:
            pk = f"{scaler_key}_{stratify_key}"
            if blob["improvements"].get(pk, {}).get("smote"):
                print(f"  {pk} smote already cached, skipping")
                continue
            strat_y = y if stratify_key == "yes" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            if scaler:
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
            print(f"  {pk} SMOTE resampling …", end="", flush=True)
            sm = SMOTE(random_state=42, k_neighbors=5)
            Xtr_res, ytr_res = sm.fit_resample(Xtr_s, ytr)
            print(f" {len(ytr)}→{len(ytr_res)} samples", end="", flush=True)
            smote_res = {}
            for mk in MODELS:
                clf = make_model(mk)
                clf.fit(Xtr_res, ytr_res)
                smote_res[mk] = eval_result(yte, clf.predict(Xte_s))
                print(".", end="", flush=True)
            blob["improvements"][pk]["smote"] = smote_res
            print(" done")
            save(blob)

print("\nDone. Run: python generate_landsat_adventure.py  to regenerate the HTML.")

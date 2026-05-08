#!/usr/bin/env python3
"""
generate_landsat_adventure.py

Generates landsat_adventure.html — a self-contained choose-your-own-adventure
ML walkthrough using the Statlog (Landsat Satellite) dataset.

Run:  python generate_landsat_adventure.py
Out:  landsat_adventure.html  (no internet required to view)

First run takes ~3 min (excludes SVM — see note in HTML). Caches to landsat_data.json
so subsequent re-runs to fix HTML are instant.
"""

import json, os, sys, numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import warnings; warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_LABELS = [1, 2, 3, 4, 5, 7]
CLASS_NAMES  = ["Red Soil", "Cotton Crop", "Grey Soil",
                "Damp Grey Soil", "Soil w/ Vegetation", "Very Damp Grey Soil"]
MODEL_NAMES  = {"lr": "Logistic Reg.", "knn": "k-NN (k=5)",
                "dt": "Decision Tree", "nb": "Naive Bayes", "rf": "Random Forest"}
GREY_LABELS  = {3, 4, 7}   # Grey Soil, Damp Grey Soil, Very Damp Grey Soil

HP_GRIDS = {
    "lr":  {"params": [{"C": c, "penalty": p}
                       for c in [0.001, 0.01, 0.1, 1, 10, 100] for p in ["l1","l2"]],
            "x":"C",     "x_vals":[0.001,0.01,0.1,1,10,100],
            "hue":"penalty", "hue_vals":["l1","l2"]},
    "knn": {"params": [{"n_neighbors": k, "metric": m}
                       for k in [1,3,5,7,11,15,21] for m in ["euclidean","manhattan"]],
            "x":"n_neighbors", "x_vals":[1,3,5,7,11,15,21],
            "hue":"metric",    "hue_vals":["euclidean","manhattan"]},
    "dt":  {"params": [{"max_depth": d, "min_samples_leaf": l}
                       for d in [2,3,5,8,12,None] for l in [1,2,5,10]],
            "x":"max_depth",      "x_vals":[2,3,5,8,12,"null"],
            "hue":"min_samples_leaf", "hue_vals":[1,2,5,10]},
    "nb":  {"params": [{"var_smoothing": v}
                       for v in [1e-9, 1e-7, 1e-5, 1e-3, 0.1]],
            "x":"var_smoothing", "x_vals":[1e-9, 1e-7, 1e-5, 1e-3, 0.1],
            "hue": None, "hue_vals": None},
    "rf":  {"params": [{"n_estimators": n, "max_depth": d}
                       for n in [10,50,100,200] for d in [3,5,10,None]],
            "x":"n_estimators", "x_vals":[10,50,100,200],
            "hue":"max_depth",  "hue_vals":[3,5,10,"null"]},
}

# ── Model factory ─────────────────────────────────────────────────────────────

def make_model(key, params=None):
    p = {k: (None if v == "null" else v) for k, v in (params or {}).items()}
    if key == "lr":  return LogisticRegression(max_iter=2000, solver="saga", **p)
    if key == "knn": return KNeighborsClassifier(**p)
    if key == "dt":  return DecisionTreeClassifier(random_state=42, **p)
    if key == "nb":  return GaussianNB(**p)
    if key == "rf":  return RandomForestClassifier(random_state=42, **p)

def get_scaler(key):
    return {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": None}[key]

def add_ndmi(X_raw):
    """Append 9 NDMI features — (B3−B4)/(B3+B4) per pixel — to a raw feature matrix.
    Features are ordered B1-Px1,B2-Px1,B3-Px1,B4-Px1, B1-Px2, … so for pixel p (0-indexed):
    B3 = X[:,p*4+2], B4 = X[:,p*4+3]."""
    cols = []
    for p in range(9):
        b3 = X_raw[:, p * 4 + 2].astype(float)
        b4 = X_raw[:, p * 4 + 3].astype(float)
        cols.append(((b3 - b4) / (b3 + b4 + 1e-8)).reshape(-1, 1))
    return np.hstack([X_raw, np.hstack(cols)])

def make_weighted(key):
    p = {"class_weight": "balanced"}
    if key == "lr":  return LogisticRegression(max_iter=2000, solver="saga", **p)
    if key == "dt":  return DecisionTreeClassifier(random_state=42, **p)
    if key == "rf":  return RandomForestClassifier(random_state=42, **p)
    return None  # knn and nb do not support class_weight

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

# ── Computation ───────────────────────────────────────────────────────────────

def compute_all(X, y):
    blob = {}

    # --- overview ---
    blob["overview"] = {
        "n_samples":    int(X.shape[0]),
        "n_features":   int(X.shape[1]),
        "class_counts": {str(cl): int((y == cl).sum()) for cl in CLASS_LABELS},
        "class_names":  CLASS_NAMES,
        "class_labels": CLASS_LABELS,
        "nan_count":    int(np.isnan(X).sum()),
        "sample_rows":  X[:5, :].round(1).tolist(),
        "feature_names":[f"B{b}-Px{p}" for p in range(1,10) for b in range(1,5)],
    }

    # --- regression dead-end ---
    Xtr, Xte, ytr, yte = train_test_split(X, y.astype(float), test_size=0.2, random_state=42)
    sc0 = StandardScaler()
    Xtr_s = sc0.fit_transform(Xtr); Xte_s = sc0.transform(Xte)
    reg = LinearRegression(); reg.fit(Xtr_s, ytr); yp = reg.predict(Xte_s)
    blob["regression_demo"] = {
        "r2": round(float(reg.score(Xte_s, yte)), 3),
        "sample_preds": [[round(float(a),0), round(float(b),2)]
                         for a, b in zip(yte[:10].tolist(), yp[:10].tolist())],
    }

    # --- preprocessing combos ---
    blob["preprocessing"] = {}
    for scaler_key in ["standard", "minmax", "none"]:
        for stratify_key in ["yes", "no"]:
            pk = f"{scaler_key}_{stratify_key}"
            print(f"\n── {pk} ──", flush=True)

            strat_y = y if stratify_key == "yes" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)

            scaler = get_scaler(scaler_key)
            if scaler:
                Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr.copy(), Xte.copy()

            # CV comparison
            cv_data = {}
            for nf in [3, 5, 10]:
                cv_obj = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
                cv_data[str(nf)] = {}
                for mk in ["lr","knn","dt","nb","rf"]:
                    scores = cross_val_score(make_model(mk), Xtr_s, ytr,
                                             cv=cv_obj, scoring="accuracy")
                    cv_data[str(nf)][mk] = {
                        "mean": round(float(scores.mean()), 4),
                        "std":  round(float(scores.std()),  4),
                    }
                    print(".", end="", flush=True)

            # Default test-set CMs
            test_cms = {}
            for mk in ["lr","knn","dt","nb","rf"]:
                clf = make_model(mk); clf.fit(Xtr_s, ytr)
                yp = clf.predict(Xte_s)
                test_cms[mk] = confusion_matrix(yte, yp, labels=CLASS_LABELS).tolist()

            # HP tuning
            hp_data = {}
            for mk in ["lr","knn","dt","nb","rf"]:
                print(f"\n  HP {mk}", end="", flush=True)
                grid   = HP_GRIDS[mk]
                results = []
                for params in grid["params"]:
                    actual = {k: (None if v == "null" else v) for k, v in params.items()}
                    clf = make_model(mk, actual); clf.fit(Xtr_s, ytr)
                    yp  = clf.predict(Xte_s)
                    acc = float((yp == yte).mean())
                    cm  = confusion_matrix(yte, yp, labels=CLASS_LABELS).tolist()
                    pr, rc, _, _ = precision_recall_fscore_support(
                        yte, yp, labels=CLASS_LABELS, zero_division=0)
                    results.append({
                        "params":    {k: ("null" if v is None else v) for k,v in actual.items()},
                        "score":     round(acc, 4),
                        "cm":        cm,
                        "precision": [round(float(x),3) for x in pr],
                        "recall":    [round(float(x),3) for x in rc],
                    })
                    print(".", end="", flush=True)
                best_idx = max(range(len(results)), key=lambda i: results[i]["score"])
                hp_data[mk] = {
                    "results":   results,
                    "best_idx":  best_idx,
                    "x":         grid["x"],
                    "x_vals":    grid["x_vals"],
                    "hue":       grid["hue"],
                    "hue_vals":  grid["hue_vals"],
                }

            blob["preprocessing"][pk] = {
                "cv":       cv_data,
                "test_cms": test_cms,
                "hp":       hp_data,
            }

    return blob

# ── Learning curves ───────────────────────────────────────────────────────────

def compute_learning_curves(X, y, cache_path=None, cache_blob=None):
    existing = (cache_blob or {}).get("learning_curves", {})
    blob = dict(existing)
    train_fracs = [0.10, 0.25, 0.50, 0.75, 1.00]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for scaler_key in ["standard", "minmax", "none"]:
        for stratify_key in ["yes", "no"]:
            pk = f"{scaler_key}_{stratify_key}"
            strat_y = y if stratify_key == "yes" else None
            Xtr, _, ytr, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            Xtr_s = scaler.fit_transform(Xtr) if scaler else Xtr.copy()
            if pk not in blob:
                blob[pk] = {}
            for mk in ["lr", "knn", "dt", "rf"]:
                if mk in blob.get(pk, {}):
                    print(f"  {pk}/{mk} already cached, skipping", flush=True)
                    continue
                sizes, tr_sc, val_sc = learning_curve(
                    make_model(mk), Xtr_s, ytr,
                    train_sizes=train_fracs, cv=cv,
                    scoring="accuracy", n_jobs=1)
                blob[pk][mk] = {
                    "sizes":      [int(s) for s in sizes],
                    "train_mean": [round(float(v), 4) for v in tr_sc.mean(axis=1)],
                    "train_std":  [round(float(v), 4) for v in tr_sc.std(axis=1)],
                    "val_mean":   [round(float(v), 4) for v in val_sc.mean(axis=1)],
                    "val_std":    [round(float(v), 4) for v in val_sc.std(axis=1)],
                }
                print(f"  {pk}/{mk} learning curve done", flush=True)
                if cache_path and cache_blob is not None:
                    cache_blob["learning_curves"] = blob
                    with open(cache_path, "w") as f:
                        json.dump(cache_blob, f)
    return blob


# ── Feature importances ───────────────────────────────────────────────────────

def compute_feature_importances(X, y):
    blob = {}
    for scaler_key in ["standard", "minmax", "none"]:
        for stratify_key in ["yes", "no"]:
            pk = f"{scaler_key}_{stratify_key}"
            strat_y = y if stratify_key == "yes" else None
            Xtr, _, ytr, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            Xtr_s = scaler.fit_transform(Xtr) if scaler else Xtr.copy()
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(Xtr_s, ytr)
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(Xtr_s, ytr)
            blob[pk] = {
                "rf": [round(float(x), 5) for x in rf.feature_importances_],
                "dt": [round(float(x), 5) for x in dt.feature_importances_],
            }
            print(f"  {pk} importances done", flush=True)
    return blob


# ── Improvement strategies ────────────────────────────────────────────────────

def compute_improvements(X, y):
    blob = {}
    grey_list = sorted(GREY_LABELS)

    for scaler_key in ["standard", "minmax", "none"]:
        for stratify_key in ["yes", "no"]:
            pk = f"{scaler_key}_{stratify_key}"
            print(f"\n── improvements {pk} ──", flush=True)
            strat_y = y if stratify_key == "yes" else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat_y)
            scaler = get_scaler(scaler_key)
            if scaler:
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
            else:
                Xtr_s, Xte_s = Xtr.copy(), Xte.copy()

            # ── A: NDMI feature engineering ───────────────────────────────────
            Xtr_aug = add_ndmi(Xtr)
            Xte_aug = add_ndmi(Xte)
            sc2 = get_scaler(scaler_key)
            if sc2:
                Xtr_aug_s = sc2.fit_transform(Xtr_aug)
                Xte_aug_s = sc2.transform(Xte_aug)
            else:
                Xtr_aug_s, Xte_aug_s = Xtr_aug.copy(), Xte_aug.copy()
            ndmi = {}
            for mk in ["lr", "knn", "dt", "nb", "rf"]:
                clf = make_model(mk)
                clf.fit(Xtr_aug_s, ytr)
                ndmi[mk] = eval_result(yte, clf.predict(Xte_aug_s))
                print(".", end="", flush=True)

            # ── B: class_weight='balanced' ────────────────────────────────────
            weighted = {}
            for mk in ["lr", "knn", "dt", "nb", "rf"]:
                if mk in ("knn", "nb"):
                    weighted[mk] = {"unsupported": True}
                    continue
                clf = make_weighted(mk)
                clf.fit(Xtr_s, ytr)
                weighted[mk] = eval_result(yte, clf.predict(Xte_s))
                print(".", end="", flush=True)

            # ── C: hierarchical (RF throughout) ──────────────────────────────
            ytr_bin = np.where(np.isin(ytr, grey_list), 1, 0)
            yte_bin = np.where(np.isin(yte, grey_list), 1, 0)
            s1 = RandomForestClassifier(n_estimators=100, random_state=42)
            s1.fit(Xtr_s, ytr_bin)
            yp_bin = s1.predict(Xte_s)
            s1_acc = float((yp_bin == yte_bin).mean())

            gm = np.isin(ytr, grey_list)
            s2 = RandomForestClassifier(n_estimators=100, random_state=42)
            s2.fit(Xtr_s[gm], ytr[gm])
            s3 = RandomForestClassifier(n_estimators=100, random_state=42)
            s3.fit(Xtr_s[~gm], ytr[~gm])

            yp_comb = np.zeros(len(yte), dtype=int)
            grey_pred = yp_bin == 1
            if grey_pred.any():
                yp_comb[grey_pred]  = s2.predict(Xte_s[grey_pred])
            if (~grey_pred).any():
                yp_comb[~grey_pred] = s3.predict(Xte_s[~grey_pred])
            hier = {"stage1_accuracy": round(s1_acc, 4), **eval_result(yte, yp_comb)}
            print("H", end="", flush=True)

            # ── Misclassified grey soil samples (raw features for visualization)
            clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_rf.fit(Xtr_s, ytr)
            yp_rf = clf_rf.predict(Xte_s)
            confused = np.isin(yte, grey_list) & (yp_rf != yte)
            samples = []
            for idx in np.where(confused)[0][:6]:
                samples.append({
                    "actual":       CLASS_NAMES[CLASS_LABELS.index(int(yte[idx]))],
                    "pred":         CLASS_NAMES[CLASS_LABELS.index(int(yp_rf[idx]))],
                    "features_raw": Xte[idx].round(1).tolist(),
                })

            blob[pk] = {
                "ndmi":               ndmi,
                "weighted":           weighted,
                "hierarchical":       hier,
                "misclassified_grey": samples,
            }
    return blob

# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Land Cover Classification — Choose Your Own Adventure</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8f5fb;color:#1e1030}
header{background:linear-gradient(135deg,#5a2d82,#3a1a5c);color:#fff;padding:14px 32px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 10px rgba(0,0,0,.3);position:sticky;top:0;z-index:200}
header h1{font-size:1.25rem;font-weight:700}
#prog-wrap{display:flex;gap:6px;align-items:center}
#reset-btn{background:none;border:1.5px solid rgba(255,255,255,.45);color:rgba(255,255,255,.9);border-radius:6px;padding:5px 13px;cursor:pointer;font-size:.8rem;margin-left:14px;transition:all .15s}
#reset-btn:hover{background:rgba(255,255,255,.15);border-color:#fff}
.pdot{width:10px;height:10px;border-radius:50%;background:rgba(255,255,255,.3);transition:all .2s}
.pdot.done{background:#4aaa7a}
.pdot.active{background:#fff;transform:scale(1.3)}
#content{max-width:1100px;margin:0 auto;padding:24px 24px 110px}
.step-title{font-size:1.4rem;font-weight:700;color:#5a2d82;margin-bottom:6px}
.step-sub{font-size:.95rem;color:#4a3060;line-height:1.7;margin-bottom:18px}
.two-col{display:flex;gap:20px;flex-wrap:wrap}
.col-left{flex:1 1 380px}
.col-right{flex:1 1 320px}
.panel{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.09);overflow:hidden;margin-bottom:18px}
.panel-hdr{background:#ede8f5;border-bottom:1px solid #c0a8e0;padding:8px 16px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#5a2d82}
.panel-body{padding:16px}
/* choice cards */
.choices{display:flex;flex-direction:column;gap:10px;margin-bottom:16px}
.choice-card{border:2px solid #c0a8e0;border-radius:10px;padding:12px 16px;cursor:pointer;transition:all .15s;background:#fff}
.choice-card:hover{border-color:#5a2d82;background:#f3eef9}
.choice-card.selected{border-color:#5a2d82;background:#ede8f5}
.choice-card.bad.selected{border-color:#ca8a04;background:#fef9c3}
.choice-card.dead.selected{border-color:#dc2626;background:#fee2e2}
.choice-title{font-weight:700;font-size:.95rem;color:#3a1a5c;margin-bottom:2px}
.choice-body{font-size:.85rem;color:#5a3a7a;line-height:1.5}
/* warning / dead-end banners */
.warn-box{background:#fef9c3;border:1px solid #ca8a04;border-radius:8px;padding:12px 16px;margin-top:10px;font-size:.88rem;line-height:1.6;color:#78350f}
.warn-box strong{color:#92400e}
.dead-box{background:#fee2e2;border:1px solid #dc2626;border-radius:8px;padding:12px 16px;margin-top:10px;font-size:.88rem;line-height:1.6;color:#7f1d1d}
.dead-box strong{display:block;font-size:1rem;margin-bottom:6px}
/* code panel */
.code-panel{background:#1e1030;border-radius:10px;padding:14px 18px;margin-top:12px}
.code-panel pre{font-family:'Menlo','Courier New',monospace;font-size:.82rem;line-height:1.7;color:#c9b8e8;white-space:pre-wrap}
.code-kw{color:#d397fa}
.code-str{color:#86efac}
.code-cm{color:#6b7280;font-style:italic}
/* confusion matrix */
.cm-wrap{overflow-x:auto}
table.cm{border-collapse:collapse;font-size:.78rem}
table.cm th{background:#ede8f5;color:#5a2d82;padding:5px 8px;font-weight:700;text-align:center;white-space:nowrap}
/* tooltips — JS-positioned fixed div escapes overflow containers */
.tip{cursor:help;text-decoration:underline dotted #9b6acd}
table.cm td{padding:5px 10px;text-align:center;font-weight:600;white-space:nowrap}
table.cm .row-lbl{background:#ede8f5;color:#5a2d82;font-weight:700;text-align:right;padding-right:10px}
/* bar chart SVG */
.barchart svg{display:block;width:100%}
/* hp heatmap */
table.hmap{border-collapse:collapse;font-size:.8rem;cursor:pointer}
table.hmap th{background:#ede8f5;color:#5a2d82;padding:5px 10px;font-weight:700;text-align:center}
table.hmap td{padding:5px 12px;text-align:center;font-weight:600;transition:outline .1s}
table.hmap td:hover{outline:2px solid #5a2d82}
table.hmap td.sel{outline:3px solid #5a2d82}
/* pr bar */
.pr-bar-wrap{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:.82rem}
.pr-bar-label{width:110px;text-align:right;color:#4a3060;flex-shrink:0;font-size:.78rem}
.pr-bar-bg{flex:1;background:#ede8f5;border-radius:4px;height:18px;position:relative}
.pr-bar-fg{height:100%;border-radius:4px;transition:width .3s}
.pr-bar-val{position:absolute;right:4px;top:0;line-height:18px;font-size:.75rem;font-weight:700;color:#3a1a5c}
/* nav */
.nav-bar{position:fixed;bottom:0;left:0;right:0;display:flex;justify-content:center;align-items:center;gap:24px;padding:12px 0 16px;background:#f8f5fb;border-top:1px solid #d0c0e8;z-index:100}
.nav-bar button{background:#5a2d82;color:#fff;border:none;padding:11px 32px;font-size:.97rem;font-weight:600;border-radius:8px;cursor:pointer;transition:background .15s,transform .1s}
.nav-bar button:hover{background:#46236a;transform:translateY(-1px)}
.nav-bar button:disabled{background:#c0a8e0;cursor:default;transform:none}
.info-tag{display:inline-block;background:#3b82f6;color:#fff;padding:1px 8px;border-radius:4px;font-size:.78rem;font-weight:700;margin-left:6px}
.tag-g{background:#16a34a}
.tag-r{background:#dc2626}
.model-cm-grid{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}
.model-cm-card{flex:1 1 280px}
.model-cm-title{font-size:.82rem;font-weight:700;color:#5a2d82;margin-bottom:5px}
.section-label{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#8b6aaa;margin:14px 0 6px}
</style>
</head>
<body>
<header>
  <h1>🌍 Land Cover Classification — Choose Your Own Adventure</h1>
  <div style="display:flex;align-items:center">
    <div id="prog-wrap"></div>
    <button id="reset-btn" onclick="resetAdventure()" title="Clear all selections and start over">↺ Reset</button>
  </div>
</header>
<div id="content"></div>
<div class="nav-bar">
  <button id="back-btn" onclick="goBack()">← Back</button>
  <button id="next-btn" onclick="goNext()">Continue →</button>
</div>
<script>
const DATA = <<<DATA_JSON>>>;

const STEP_COUNT = 13;
let state = {
  step:0, scaler:null, stratify:null, cv_folds:null,
  model:null, hp_idx:null,
  label_choice:null, nan_choice:null, cat_choice:null, prob_type:null,
  metric:null, improvement:null,
};

const MODEL_COLORS = {lr:"#3b82f6",knn:"#10b981",dt:"#f59e0b",nb:"#06b6d4",rf:"#8b5cf6"};
const MODEL_NAMES  = {lr:"Logistic Reg.",knn:"k-NN (k=5)",dt:"Decision Tree",nb:"Naive Bayes",rf:"Random Forest"};
// Always-predict-Red-Soil baseline (1531/6430=0.238; Macro F1=0.064; Weighted F1=0.092)
const DUMMY_BASELINE = {accuracy:0.238, macro_f1:0.064, weighted_f1:0.092};

const HP_TIPS = {
  C:                "Inverse regularization strength. Smaller C = stronger regularization — the model is penalized more for large coefficients, producing a simpler decision boundary. Larger C = weaker regularization — the model fits the training data more closely but may overfit.",
  penalty:          "Regularization type. L1 (Lasso) can shrink some coefficients to exactly zero, effectively removing features. L2 (Ridge) shrinks all coefficients toward zero but rarely eliminates them — better when most features are informative.",
  n_neighbors:      "Number of nearest training examples to consult when classifying a new point. k=1 memorizes the training set (high variance, low bias). Larger k smooths decision boundaries — a point is classified by the majority of a wider neighborhood.",
  metric:           "Distance function used to find nearest neighbors. Euclidean = straight-line distance in feature space. Manhattan = sum of absolute differences along each axis. Both are sensitive to feature scale, which is why k-NN benefits from scaling.",
  max_depth:        "Maximum number of splits from root to leaf. Deeper trees fit the training data more precisely but risk overfitting — they can learn noise. Shallower trees generalize better. None = grow until all leaves are pure.",
  min_samples_leaf: "Minimum number of training samples required to form a leaf node. Higher values prevent the tree from creating tiny leaves that memorize noise. Acts as a smoothing constraint on the decision boundary.",
  n_estimators:     "Number of decision trees in the forest. More trees average out more variance and produce more stable predictions, but with diminishing returns — going from 10→50 helps far more than 150→200.",
  var_smoothing:    "Adds this fraction of the largest per-feature variance to every class's variance estimate. Prevents zero-probability predictions when a feature value wasn't seen for a given class during training. Larger values smooth more aggressively.",
  l1:               "Lasso regularization — penalizes the sum of absolute coefficient values. Tends to zero out less-useful features entirely, producing sparse models.",
  l2:               "Ridge regularization — penalizes the sum of squared coefficient values. Shrinks all coefficients toward zero without eliminating them. Usually the better default when most features contribute signal.",
  euclidean:        "Straight-line distance: √(Σ(a−b)²). Treats all dimensions equally. Sensitive to large-scale features dominating — use with a scaler.",
  manhattan:        "Sum of absolute differences: Σ|a−b|. Less sensitive to outliers in individual features than Euclidean distance.",
};
function hpTip(key){ const t=HP_TIPS[String(key)]; return t?` class="tip" data-tip="${t}"`:""; }

function getKey(){ return state.scaler+"_"+state.stratify; }
function getPrep(){ return DATA.preprocessing[getKey()]; }

function choose(field, val){ state[field]=val; render(); }
function chooseHP(idx){ state.hp_idx=idx; render(); }

function nextEnabled(){
  const s=state.step;
  if(s===0)  return true;
  if(s===1)  return true;
  if(s===2)  return state.label_choice && state.nan_choice && state.cat_choice;
  if(s===3)  return state.stratify !== null;
  if(s===4)  return state.scaler !== null;
  if(s===5)  return state.prob_type === "classification";
  if(s===6)  return state.metric !== null;
  if(s===7)  return state.cv_folds !== null;
  if(s===8)  return state.model !== null;
  if(s===9)  return state.hp_idx !== null;
  if(s===10) return true;
  if(s===11) return state.improvement !== null;
  if(s===12) return false;
  return false;
}

function goNext(){ if(nextEnabled()){ state.step++; window.scrollTo(0,0); render(); } }
function resetAdventure(){
  Object.assign(state,{step:0,scaler:null,stratify:null,cv_folds:null,model:null,hp_idx:null,
    label_choice:null,nan_choice:null,cat_choice:null,prob_type:null,metric:null,improvement:null});
  try{ localStorage.removeItem("landsat_state_v2"); }catch(e){}
  window.scrollTo(0,0); render();
}
function goBack(){
  state.step--;
  // reset downstream choices when going back past a branching point
  if(state.step<=3){ state.scaler=null; state.metric=null; state.cv_folds=null; state.model=null; state.hp_idx=null; }
  if(state.step<=4){ state.metric=null; state.cv_folds=null; state.model=null; state.hp_idx=null; }
  if(state.step<=5){ state.prob_type=null; state.metric=null; state.cv_folds=null; state.model=null; state.hp_idx=null; }
  if(state.step<=6){ state.metric=null; state.cv_folds=null; state.model=null; state.hp_idx=null; }
  if(state.step<=7){ state.cv_folds=null; state.model=null; state.hp_idx=null; }
  if(state.step<=8){ state.model=null; state.hp_idx=null; }
  if(state.step<=9){ state.hp_idx=null; }
  if(state.step<=10){ state.improvement=null; }
  window.scrollTo(0,0); render();
}

function render(){
  try { localStorage.setItem("landsat_state_v2", JSON.stringify(state)); } catch(e){}
  document.getElementById("content").innerHTML = renderStep(state.step);
  const nb=document.getElementById("next-btn");
  const bb=document.getElementById("back-btn");
  nb.disabled = !nextEnabled();
  if(state.step===12) nb.style.display="none"; else nb.style.display="";
  bb.disabled = state.step===0;
  // progress dots
  let dots="";
  for(let i=0;i<STEP_COUNT;i++){
    let cls="pdot";
    if(i<state.step) cls+=" done";
    else if(i===state.step) cls+=" active";
    dots+=`<div class="${cls}"></div>`;
  }
  document.getElementById("prog-wrap").innerHTML=dots;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function card(title,body,field,val,cls=""){
  const sel = state[field]===val ? " selected" : "";
  const badCls = cls ? " "+cls : "";
  return `<div class="choice-card${sel}${badCls}" onclick="choose('${field}','${val}')">
    <div class="choice-title">${title}</div>
    <div class="choice-body">${body}</div>
  </div>`;
}

function cmTable(cm, classNames){
  const n=cm.length;
  const flat=cm.flat(); const maxV=Math.max(...flat)||1;
  let h=`<div class="cm-wrap"><table class="cm"><thead><tr><th>Actual \\ Pred</th>`;
  classNames.forEach(c=>{ h+=`<th class="tip" data-tip="${classTip(c)}">${c}</th>`; });
  h+="</tr></thead><tbody>";
  cm.forEach((row,i)=>{
    h+=`<tr><td class="row-lbl tip" data-tip="${classTip(classNames[i])}">${classNames[i]}</td>`;
    row.forEach((v,j)=>{
      const t=v/maxV;
      let bg,fg;
      if(i===j){
        const g=Math.round(212-(212-26)*t); bg=`rgb(${Math.round(26+(212-26)*(1-t))},${g},${Math.round(64+(120-64)*(1-t))})`;
        fg=t>0.45?"#fff":"#1e4030";
      } else {
        bg=`rgb(${Math.round(248-(248-90)*t)},${Math.round(245-(245-45)*t)},${Math.round(251-(251-130)*t)})`;
        fg=t>0.55?"#fff":"#2e1a48";
      }
      h+=`<td style="background:${bg};color:${fg}">${v}</td>`;
    });
    h+="</tr>";
  });
  h+="</tbody></table></div>";
  return h;
}

function cvBarSvg(foldKey){
  const d=getPrep().cv[foldKey];
  const models=["lr","knn","dt","nb","rf"];
  const sorted=[...models].sort((a,b)=>d[b].mean-d[a].mean);
  const W=460,barH=28,gap=10,padL=100,padR=60,padT=20;
  const H=padT+(barH+gap)*(models.length+1);
  const minV=0.0, maxV=1.0, scale=W-padL-padR;
  let svg=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px">`;
  sorted.forEach((mk,i)=>{
    const y=padT+i*(barH+gap);
    const bw=Math.max(0,(d[mk].mean-minV)/(maxV-minV)*scale);
    const ew=(d[mk].std/(maxV-minV))*scale;
    const ex=padL+bw;
    svg+=`<rect x="${padL}" y="${y}" width="${bw}" height="${barH}" fill="${MODEL_COLORS[mk]}" rx="4"/>`;
    svg+=`<line x1="${ex-ew}" x2="${ex+ew}" y1="${y+barH/2}" y2="${y+barH/2}" stroke="#333" stroke-width="2"/>`;
    svg+=`<line x1="${ex-ew}" x2="${ex-ew}" y1="${y+5}" y2="${y+barH-5}" stroke="#333" stroke-width="1.5"/>`;
    svg+=`<line x1="${ex+ew}" x2="${ex+ew}" y1="${y+5}" y2="${y+barH-5}" stroke="#333" stroke-width="1.5"/>`;
    svg+=`<text x="${padL-6}" y="${y+barH/2+4}" text-anchor="end" font-size="12" fill="#4a3060">${MODEL_NAMES[mk]}</text>`;
    svg+=`<text x="${padL+bw+ew+6}" y="${y+barH/2+4}" font-size="12" font-weight="bold" fill="#3a1a5c">${d[mk].mean.toFixed(3)}</text>`;
  });
  const by=padT+models.length*(barH+gap);
  const bbw=DUMMY_BASELINE.accuracy*scale;
  svg+=`<rect x="${padL}" y="${by}" width="${bbw}" height="${barH}" fill="#c8c0d8" rx="4" stroke="#8b6aaa" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  svg+=`<text x="${padL-6}" y="${by+barH/2+4}" text-anchor="end" font-size="11" fill="#8b6aaa">Dummy baseline</text>`;
  svg+=`<text x="${padL+bbw+6}" y="${by+barH/2+4}" font-size="12" font-weight="bold" fill="#8b6aaa">${DUMMY_BASELINE.accuracy.toFixed(3)}</text>`;
  svg+=`<line x1="${padL}" x2="${padL}" y1="${padT-4}" y2="${H-4}" stroke="#c0a8e0" stroke-width="1"/>`;
  svg+=`</svg>`;
  return `<div class="barchart">${svg}</div>`;
}

function hpHeatmap(mk){
  const hp=getPrep().hp[mk];
  const xVals=hp.x_vals, hueVals=hp.hue_vals;
  const xKey=hp.x, hueKey=hp.hue;
  const scores=hp.results.map(r=>r.score);
  const minS=Math.min(...scores), maxS=Math.max(...scores);
  // 1-D case (e.g. Naive Bayes — only var_smoothing, no second axis)
  if(!hueVals){
    let h=`<table class="hmap"><thead><tr><th${hpTip(xKey)}>${xKey}</th>`;
    xVals.forEach(v=>{ h+=`<th${hpTip(v)}>${v}</th>`; }); h+="</tr></thead><tbody><tr><td style='background:#ede8f5;color:#5a2d82;font-weight:700'>accuracy</td>";
    xVals.forEach((xv,idx)=>{
      const sc=hp.results[idx].score;
      const t=(sc-minS)/(maxS-minS||1);
      const bg=`rgb(${Math.round(248-(248-90)*t)},${Math.round(245-(245-45)*t)},${Math.round(251-(251-130)*t)})`;
      const fg=t>0.55?"#fff":"#2e1a48";
      const sel=state.hp_idx===idx?" sel":"";
      const best=idx===hp.best_idx?" ★":"";
      h+=`<td class="${sel}" style="background:${bg};color:${fg}" onclick="chooseHP(${idx})">${sc.toFixed(3)}${best}</td>`;
    });
    h+="</tr></tbody></table>"; return h;
  }
  let h=`<table class="hmap"><thead><tr><th><span${hpTip(hueKey)}>${hueKey}</span> \\ <span${hpTip(xKey)}>${xKey}</span></th>`;
  xVals.forEach(v=>{ h+=`<th${hpTip(v)}>${v==="null"?"None":v}</th>`; }); h+="</tr></thead><tbody>";
  hueVals.forEach(hv=>{
    h+=`<tr><td style="background:#ede8f5;color:#5a2d82;font-weight:700"${hpTip(hv)}>${hv}</td>`;
    xVals.forEach(xv=>{
      const idx=hp.results.findIndex(r=>
        String(r.params[xKey])=== String(xv) && String(r.params[hueKey])===String(hv));
      if(idx===-1){ h+=`<td>—</td>`; return; }
      const sc=hp.results[idx].score;
      const t=(sc-minS)/(maxS-minS||1);
      const bg=`rgb(${Math.round(248-(248-90)*t)},${Math.round(245-(245-45)*t)},${Math.round(251-(251-130)*t)})`;
      const fg=t>0.55?"#fff":"#2e1a48";
      const sel=state.hp_idx===idx?" sel":"";
      const best=idx===hp.best_idx?" ★":"";
      h+=`<td class="${sel}" style="background:${bg};color:${fg}" onclick="chooseHP(${idx})">${sc.toFixed(3)}${best}</td>`;
    });
    h+="</tr>";
  });
  h+="</tbody></table>";
  return h;
}

function prBars(precision, recall, classNames){
  let h="";
  classNames.forEach((cn,i)=>{
    const pr=precision[i], rc=recall[i];
    h+=`<div class="section-label">${cn}</div>`;
    h+=prBar("Precision",pr,"#3b82f6");
    h+=prBar("Recall",rc,"#10b981");
  });
  return h;
}

function prBar(label, val, color){
  const pct=Math.round(val*100);
  return `<div class="pr-bar-wrap">
    <div class="pr-bar-label">${label}</div>
    <div class="pr-bar-bg"><div class="pr-bar-fg" style="width:${pct}%;background:${color}"></div>
    <div class="pr-bar-val">${val.toFixed(2)}</div></div>
  </div>`;
}

function codeBlock(code){
  return `<div class="code-panel"><pre>${code}</pre></div>`;
}

function featureTip(name){
  const bandDesc = {
    "B1": "Band 1 — Visible green (0.52–0.60 µm)",
    "B2": "Band 2 — Visible red (0.63–0.69 µm)",
    "B3": "Band 3 — Near-infrared (0.76–0.90 µm)",
    "B4": "Band 4 — Shortwave-infrared (1.55–1.75 µm)",
  };
  const pixelDesc = {
    "Px1":"Top-left",   "Px2":"Top-center",    "Px3":"Top-right",
    "Px4":"Middle-left","Px5":"Center ★ (pixel being classified)","Px6":"Middle-right",
    "Px7":"Bottom-left","Px8":"Bottom-center",  "Px9":"Bottom-right",
  };
  const [band, px] = name.split("-");
  const b = bandDesc[band] || band;
  const p = pixelDesc[px]  || px;
  return `${b} · ${p} of 3×3 neighborhood`;
}

function confusionNote(actual, pred){
  const grey = new Set(["Grey Soil","Damp Grey Soil","Very Damp Grey Soil"]);
  if(grey.has(actual) && grey.has(pred)){
    return `Both are grey soil — the distinction is <strong>moisture level</strong>, which is detectable
    because water molecules strongly absorb shortwave-infrared light (Band 4, 1.55–1.75&nbsp;µm).
    Dry grey soil is bright in B4; damp soil is darker; very damp soil is darker still.
    In nature this is a continuous gradient, so pixels near the boundary between moisture levels
    are genuinely ambiguous — not a modeling failure.<br><br>
    <em>Why keep three classes instead of merging them?</em> Because soil moisture is agriculturally
    important. A model that distinguishes "damp" from "very damp" from orbit — with no ground sensors —
    enables irrigation planning, flood risk mapping, and crop stress monitoring at continental scale.
    The classes are worth separating because they <em>are</em> separable most of the time.`;
  }
  return `These two classes share some spectral similarity across the 36 band-pixel features — a physical
  property of how these surfaces reflect light, not a modeling failure.`;
}

function classTip(name){
  const tips = {
    "Red Soil":            "Bare or sparsely vegetated red-hued soil, common in arid regions. High reflectance in red and near-infrared bands due to iron oxide content.",
    "Cotton Crop":         "Active cotton cultivation. Dense canopy during growing season; high near-infrared reflectance from healthy leaf structure.",
    "Grey Soil":           "Dry grey soil with little or no vegetation cover. Moderate, relatively flat reflectance across all spectral bands.",
    "Damp Grey Soil":      "Grey soil with elevated moisture content. Moist soil absorbs more light, appearing darker across bands than dry grey soil.",
    "Soil w/ Vegetation":  "Partially vegetated soil — mixed spectral signature between bare soil and healthy vegetation. Near-infrared elevated but lower than pure crop.",
    "Very Damp Grey Soil": "Grey soil with very high moisture or surface water nearby. Strongly suppressed near-infrared; very dark in shortwave-infrared band.",
  };
  return tips[name] || name;
}

// ── Step renderers ────────────────────────────────────────────────────────────

function renderStep(n){
  return [step0,step1,step2,step3,step4,step5,step6,step7,step8,step9,step10,step11,step12][n]();
}

function step0(){
  const o=DATA.overview;
  return `<div class="step-title">Welcome to the Land Cover Classification Adventure</div>

<div class="two-col" style="margin-bottom:18px">
<div class="col-left">
<div class="panel"><div class="panel-hdr">What is a Landsat Satellite?</div><div class="panel-body" style="font-size:.9rem;line-height:1.8;color:#2e1a48">
<strong>Landsat</strong> is a joint NASA/USGS program — the longest-running enterprise for acquisition of satellite imagery of Earth, running continuously since 1972. Landsat satellites orbit about 705 km above Earth's surface, completing a full scan of the planet every 16 days.<br><br>
Their purpose: track how Earth's land surface changes over time — forests shrinking, cities expanding, glaciers retreating, crops rotating. Because Landsat data is freely available to anyone, it has become one of the most important datasets in environmental science, agriculture, urban planning, and disaster response.
</div></div>

<div class="panel"><div class="panel-hdr">What is Multispectral Imaging?</div><div class="panel-body" style="font-size:.9rem;line-height:1.8;color:#2e1a48">
A standard camera captures three channels of light: <strong>red, green, and blue</strong> — the colors human eyes can see. A multispectral sensor captures many more channels, including wavelengths <em>invisible to the human eye</em>, such as near-infrared and shortwave-infrared.<br><br>
Each channel is called a <strong>spectral band</strong>. Different land surface types — healthy vegetation, bare soil, water, urban pavement — absorb and reflect light differently across these bands. Healthy plants, for example, strongly absorb red light (for photosynthesis) but reflect near-infrared light, making them easy to distinguish from dry or dead vegetation in multispectral imagery even when they look similar to the human eye.<br><br>
The image to the right is a <strong>false-color composite</strong>: bands are mapped to red, green, and blue channels in a way that makes vegetation (near-infrared) appear bright red, revealing crop patterns invisible in natural color.
</div></div>
</div>

<div class="col-right" style="display:flex;flex-direction:column;gap:14px">
<div class="panel"><div class="panel-hdr">Landsat 9 at NASA's Kennedy Space Center, 2021</div>
<div class="panel-body" style="padding:0">
<img src="https://assets.science.nasa.gov/dynamicimage/assets/science/missions/landsat/landsat-9-mission-page/KSC-20210714-PH-RNB02_0025~orig.jpg"
     alt="Technicians remove protective covers from the Landsat 9 spacecraft inside the Integrated Processing Facility at Kennedy Space Center"
     style="width:100%;display:block;border-radius:0 0 10px 10px"
     onerror="this.parentElement.innerHTML='<p style=padding:16px;color:#8b6aaa;font-size:.85rem>Image requires an internet connection to load. &lt;a href=https://landsat.gsfc.nasa.gov/multimedia/graphics-library/landsat-9-graphics/ target=_blank&gt;View on NASA.gov&lt;/a&gt;</p>'">
<p style="font-size:.75rem;color:#8b6aaa;padding:8px 12px;margin:0">Credit: NASA/Robert Nath. Public domain.</p>
</div></div>

<div class="panel"><div class="panel-hdr">False-Color Multispectral Image — Center-Pivot Irrigation, Washington State</div>
<div class="panel-body" style="padding:0">
<img src="https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/thumbnails/image/falsecolor%20-%208-20-2019.jpg"
     alt="False-color Landsat 8 composite showing center-pivot irrigated crops — vegetation appears red because near-infrared is mapped to the red channel"
     style="width:100%;display:block;border-radius:0 0 10px 10px"
     onerror="this.parentElement.innerHTML='<p style=padding:16px;color:#8b6aaa;font-size:.85rem>Image requires an internet connection to load. &lt;a href=https://www.usgs.gov/media/images/center-pivot-irrigation-false-color-landsat-8 target=_blank&gt;View on USGS.gov&lt;/a&gt;</p>'">
<p style="font-size:.75rem;color:#8b6aaa;padding:8px 12px;margin:0">Landsat 8, August 2019. Bands 6-5-2 mapped to RGB. Credit: USGS. Public domain.</p>
</div></div>
</div>
</div>

<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">This Dataset: Six Australian Land Cover Types</div><div class="panel-body">
<p style="font-size:.85rem;color:#4a3060;margin-bottom:10px">The <strong>Statlog (Landsat Satellite)</strong> dataset contains ${o.n_samples.toLocaleString()} pixels from a multispectral image of farmland in Australia, each described by ${o.n_features} features (4 spectral bands measured across a 3×3 pixel neighborhood). Your goal: predict which of six land cover types the center pixel belongs to.</p>
<p style="font-size:.8rem;color:#8b6aaa;border-top:1px solid #ede8f5;padding-top:8px;margin-bottom:10px">The three grey soil classes are split by <strong>moisture level</strong>. Soil moisture is detectable via shortwave-infrared absorption — wet soil absorbs more light and appears darker in Band 4. Distinguishing moisture levels matters for irrigation planning and flood risk mapping.</p>
${Object.entries(o.class_counts).map(([k,v],i)=>`<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
<div style="width:14px;height:14px;border-radius:3px;background:${["#e74c3c","#f1c40f","#95a5a6","#7f8c8d","#27ae60","#1a6e8e"][i]};flex-shrink:0"></div>
<div style="flex:1;font-size:.88rem"><strong><span class="tip" data-tip="${classTip(o.class_names[i])}">${o.class_names[i]}</span></strong></div>
<div style="width:60px;text-align:right;color:#8b6aaa;font-size:.82rem">${v.toLocaleString()} px</div>
</div>`).join("")}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Your Journey (10 Steps)</div><div class="panel-body" style="font-size:.88rem;line-height:1.9;color:#4a3060">
1. Explore the data<br>2. Clean the data<br>3. Choose a split strategy<br>4. Choose a scaling strategy<br>5. Choose a problem type<br>6. Choose an evaluation metric<br>7. Cross-validate and compare models<br>8. Select the best model<br>9. Tune hyperparameters<br>10. Error analysis<br>11. Try to improve
</div></div>
<div class="panel"><div class="panel-hdr">Why Does This Matter?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
Automated land cover classification from Landsat imagery is used to monitor <strong>deforestation</strong>, track <strong>agricultural change</strong>, map <strong>urban sprawl</strong>, and measure <strong>wildfire damage</strong> — all at continental scale. Building a model that can reliably distinguish soil types and vegetation from spectral measurements is genuinely useful science.
</div></div>
</div></div>`;
}

function step1(){
  const o=DATA.overview;
  const rows=o.sample_rows;
  const fnames=o.feature_names;
  let tableH=`<div style="overflow-x:auto"><table class="cm" style="white-space:nowrap"><thead><tr><th>Row</th>`;
  fnames.forEach(f=>{tableH+=`<th class="tip" data-tip="${featureTip(f)}">${f}</th>`;});
  tableH+=`</tr></thead><tbody>`;
  rows.forEach((row,i)=>{
    tableH+=`<tr><td class="row-lbl">${i+1}</td>`;
    row.forEach(v=>{tableH+=`<td>${v}</td>`;});
    tableH+=`</tr>`;
  });
  tableH+="</tbody></table></div>";

  const counts=o.class_counts; const names=o.class_names; const labels=o.class_labels;
  const vals=labels.map(l=>counts[String(l)]);
  const maxV=Math.max(...vals);
  const colors=["#e74c3c","#f1c40f","#95a5a6","#7f8c8d","#27ae60","#1a6e8e"];
  const W=420,padL=130,padR=60,barH=26,gap=8,padT=16;
  const H=padT+(barH+gap)*6;
  let svg=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px">`;
  vals.forEach((v,i)=>{
    const y=padT+i*(barH+gap);
    const bw=(v/maxV)*(W-padL-padR);
    svg+=`<rect x="${padL}" y="${y}" width="${bw}" height="${barH}" fill="${colors[i]}" rx="3"/>`;
    svg+=`<text x="${padL-6}" y="${y+barH/2+4}" text-anchor="end" font-size="11" fill="#4a3060">${names[i]}</text>`;
    svg+=`<text x="${padL+bw+6}" y="${y+barH/2+4}" font-size="11" font-weight="bold" fill="#3a1a5c">${v.toLocaleString()}</text>`;
  });
  svg+="</svg>";

  return `<div class="step-title">Step 1 — Explore the Data</div>
<div class="step-sub">Before building any model, always understand what you're working with. How many samples? What do the features look like? Are the classes balanced?</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Top 5 rows — scroll right to see all 36 features</div>
<div class="panel-body">${tableH}</div></div>
<div class="panel"><div class="panel-hdr">Key Stats</div><div class="panel-body" style="font-size:.9rem;line-height:2;color:#4a3060">
<strong>Samples:</strong> ${o.n_samples.toLocaleString()}<br>
<strong>Features:</strong> ${o.n_features} (<span class="tip" data-tip="Each of the 4 Landsat spectral bands (green, red, near-infrared, shortwave-infrared) is measured at all 9 pixel positions in the 3×3 neighborhood surrounding the center pixel. 4 bands × 9 pixels = 36 features.">4 spectral bands × 9 pixels in a 3×3 neighborhood</span>)<br>
<strong>Missing values:</strong> ${o.nan_count} — none!<br>
<strong>Categorical features:</strong> 0 — all features are continuous numeric measurements<br>
<strong>Classes:</strong> 6 (note: class 6 "Water" was removed from the <span class="tip" data-tip="The UCI Machine Learning Repository version of the Statlog benchmark, prepared for the machine learning community. The original full dataset included a 7th class — Water/Broads — but it was removed in the UCI version because it had very few samples and was easily distinguishable.">UCI version</span>)
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Class Distribution</div>
<div class="panel-body"><div class="barchart">${svg}</div>
<p style="font-size:.82rem;color:#8b6aaa;margin-top:8px">The dataset is <strong>moderately imbalanced</strong> — Cotton Crop (2) and Damp Grey Soil (4) have about half as many samples as Red Soil (1). Worth keeping in mind when evaluating accuracy.</p>
</div></div>
</div></div>`;
}

function step2(){
  const o=DATA.overview;
  let labelWarn="", nanWarn="", catWarn="";
  let labelCode="", nanCode="", catCode="";

  if(state.label_choice==="strip"){
    labelCode=`y = data.target.str.replace('.','').astype(int)\n<span class="code-cm"># Before: '1.', '2.', … → After: 1, 2, …</span>`;
  } else if(state.label_choice==="keep"){
    labelWarn=`<div class="warn-box">⚠️ <strong>Works, but watch out.</strong> Sklearn accepts string labels, but downstream code (confusion matrix labels, plots) will sort them alphabetically: '1.' '2.' '3.' '4.' '5.' '7.' — the missing '6.' may confuse students. Integer labels are cleaner.</div>`;
    labelCode=`y = data.target  <span class="code-cm"># strings: '1.', '2.', …</span>`;
  } else if(state.label_choice==="reencode"){
    labelCode=`label_map = {v:i for i,v in enumerate(sorted(data.target.unique()))}\ny = data.target.map(label_map).values  <span class="code-cm"># 0-5</span>`;
  }

  if(state.nan_choice==="drop"){
    nanCode=`df = df.dropna()  <span class="code-cm"># our dataset: no rows dropped (nan_count=0)</span>`;
  } else if(state.nan_choice==="mean"){
    nanCode=`<span class="code-kw">from</span> sklearn.impute <span class="code-kw">import</span> SimpleImputer\nimp = SimpleImputer(strategy=<span class="code-str">'mean'</span>)\nX = imp.fit_transform(X)`;
  } else if(state.nan_choice==="median"){
    nanCode=`<span class="code-kw">from</span> sklearn.impute <span class="code-kw">import</span> SimpleImputer\nimp = SimpleImputer(strategy=<span class="code-str">'median'</span>)\nX = imp.fit_transform(X)`;
  }

  if(state.cat_choice==="onehot"){
    catCode=`<span class="code-kw">from</span> sklearn.preprocessing <span class="code-kw">import</span> OneHotEncoder\n<span class="code-cm"># For our dataset: no categorical columns, so no action needed.</span>\n<span class="code-cm"># If we had one, e.g. 'soil_type' with values A/B/C:</span>\nenc = OneHotEncoder(sparse_output=False)\nX_enc = enc.fit_transform(X[['soil_type']])`;
  } else if(state.cat_choice==="label"){
    catCode=`<span class="code-kw">from</span> sklearn.preprocessing <span class="code-kw">import</span> LabelEncoder\n<span class="code-cm"># LabelEncoder assigns ordinal integers (A→0, B→1, C→2).</span>\n<span class="code-cm"># Only appropriate if the categories have a true order.</span>\n<span class="code-cm"># For our dataset: no categorical columns.</span>`;
  }

  return `<div class="step-title">Step 2 — Clean the Data</div>
<div class="step-sub">Three questions every data scientist asks before modeling: How do I handle the target labels? Are there missing values, and how do I handle them? Are there categorical features that need encoding?</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Question 1: The class labels are '1.', '2.', … — what do you do?</div><div class="panel-body">
<div class="choices">
${card("Strip the trailing dot → integers","<code>y = labels.str.replace('.','').astype(int)</code>","label_choice","strip")}
${card("Leave them as strings","They already work as labels in sklearn","label_choice","keep","bad")}
${card("Re-encode as 0–5","Map to contiguous zero-based integers","label_choice","reencode")}
</div>
${labelWarn}
${state.label_choice ? codeBlock(labelCode) : ""}
</div></div>

<div class="panel"><div class="panel-hdr">Question 2: How would you handle missing values?</div><div class="panel-body">
<p style="font-size:.85rem;color:#4a3060;margin-bottom:10px">Our dataset has <strong>0 missing values</strong>. But on a real dataset, you'd need a strategy.</p>
<div class="choices">
${card("Drop rows with any NaN","Simple, but can lose a lot of data if NaNs are widespread","nan_choice","drop")}
${card("Impute with column mean","Good for normally-distributed numeric features","nan_choice","mean")}
${card("Impute with column median","More robust than mean when outliers are present","nan_choice","median")}
</div>
${state.nan_choice ? codeBlock(nanCode) : ""}
</div></div>

<div class="panel"><div class="panel-hdr">Question 3: How would you handle categorical features?</div><div class="panel-body">
<p style="font-size:.85rem;color:#4a3060;margin-bottom:10px">Our dataset has <strong>0 categorical columns</strong>. But on a real dataset, you'd need to encode them.</p>
<div class="choices">
${card("One-hot encoding","Creates a binary column per category. No implied order. Use for nominal categories.","cat_choice","onehot")}
${card("Label encoding","Assigns integers 0, 1, 2… Only use if categories have a true ordinal ordering.","cat_choice","label")}
</div>
${state.cat_choice ? codeBlock(catCode) : ""}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Why does this matter?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
<strong>Labels:</strong> Bad encoding can cause silent bugs — sklearn may accept the wrong type and produce strange results without errors.<br><br>
<strong>Missing values:</strong> Dropping rows loses information. Imputing adds assumptions. The right choice depends on <em>why</em> values are missing (random vs. systematic).<br><br>
<strong>Categorical encoding:</strong> One-hot is almost always safer for nominal categories. Label encoding misleads tree-based models into thinking "C &gt; B &gt; A."
</div></div>
</div>
</div>`;
}

function step3(){
  let warn="";
  let code="";
  if(state.stratify==="no"){
    warn=`<div class="warn-box">⚠️ <strong>Risky with imbalanced classes.</strong> Without stratify, random chance could put most Cotton Crop (703 samples) or Damp Grey Soil (625 samples) into one split. Your training set would underrepresent those classes and your test set's class distribution wouldn't match training. For this dataset the effect is mild, but on more imbalanced data this can seriously bias your model.</div>`;
  }
  if(state.stratify){
    const sv = state.stratify==="yes" ? ", stratify=y" : "";
    code=`<span class="code-kw">from</span> sklearn.model_selection <span class="code-kw">import</span> train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.20, random_state=42${sv}\n)\n\n<span class="code-cm"># 80% train → ${Math.round(DATA.overview.n_samples*0.8)} samples</span>\n<span class="code-cm"># 20% test  → ${Math.round(DATA.overview.n_samples*0.2)} samples</span>`;
  }

  return `<div class="step-title">Step 3 — Split the Data</div>
<div class="step-sub">We'll use an 80/20 train/test split — standard for a dataset of this size. But there's a critical question: should we stratify the split?<br><br>
<strong>Data leakage warning:</strong> Notice that we split <em>first</em>, then scale. If you scale all the data before splitting, the scaler learns statistics from the test set — that's data leakage. Test-set performance becomes slightly optimistic, and in a real deployment the model would underperform. Always fit preprocessing on training data only.</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Should we stratify the split?</div><div class="panel-body">
<div class="choices">
${card("Yes — stratified split","Preserves class proportions in both train and test sets. Recommended when classes are imbalanced.","stratify","yes")}
${card("No — random split","Each sample is assigned to train/test completely at random.","stratify","no","bad")}
</div>
${warn}
${state.stratify ? codeBlock(code) : ""}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">What is stratified splitting?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
Stratified splitting ensures each split contains roughly the same proportion of each class as the full dataset.<br><br>
For example, if Cotton Crop is 10.9% of the data, it will be ~10.9% of both train and test — not 3% in one and 18% in the other by accident.<br><br>
sklearn's <code>train_test_split</code> handles this with <code>stratify=y</code>.
</div></div>
</div>
</div>`;
}

function step4(){
  let warn="";
  let code="";
  if(state.scaler==="none"){
    warn=`<div class="warn-box">⚠️ <strong>Will hurt k-NN and Logistic Regression significantly.</strong> k-NN measures Euclidean distances — a band in 0–255 will completely dominate one in 0–1. Logistic Regression depends on gradient convergence, which is slower without normalized features. Tree-based models and Naive Bayes are more robust to unscaled features.</div>`;
  }
  if(state.scaler){
    const sc = state.scaler==="standard" ? "StandardScaler()" : state.scaler==="minmax" ? "MinMaxScaler()" : "None  # no scaling";
    const body = state.scaler!=="none" ? `scaler = ${sc}\nX_train = scaler.fit_transform(X_train)  <span class="code-cm"># fit on training data only</span>\nX_test  = scaler.transform(X_test)       <span class="code-cm"># apply same transform to test</span>` : `<span class="code-cm"># No scaling applied. Features remain in their original range.</span>`;
    code=`<span class="code-kw">from</span> sklearn.preprocessing <span class="code-kw">import</span> ${state.scaler==="standard"?"StandardScaler":state.scaler==="minmax"?"MinMaxScaler":"# (nothing to import)"}\n\n${body}`;
  }
  return `<div class="step-title">Step 4 — Feature Scaling</div>
<div class="step-sub">Our 36 features are spectral band values — all measured on similar scales, but not identical. Should we scale them? If so, how?</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Choose your scaling strategy</div><div class="panel-body">
<div class="choices">
${card("StandardScaler — zero mean, unit variance","Subtracts the mean, divides by std dev. Best when features are approximately normally distributed and you care about outlier sensitivity.","scaler","standard")}
${card("MinMaxScaler — compress to [0, 1]","Shifts and scales to a fixed range. Preserves zero values; sensitive to outliers.","scaler","minmax")}
${card("No scaling","Use raw feature values as-is.","scaler","none","bad")}
</div>
${warn}
${state.scaler ? codeBlock(code) : ""}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Which models care about scaling?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
<strong>Sensitive to scale:</strong><br>
• k-NN — distances dominate<br>
• Logistic Regression — gradient convergence<br>
• Naive Bayes — variance estimates shift<br><br>
<strong>Scale-invariant:</strong><br>
• Decision Tree — only cares about split thresholds<br>
• Random Forest — same reason<br><br>
Both scalers here produce similar accuracy on this dataset. StandardScaler is the default choice when you don't know the distribution.
</div></div>
</div>
</div>`;
}

function step5(){
  const demo=DATA.regression_demo;
  let regContent="";
  if(state.prob_type==="regression"){
    const predsHtml=demo.sample_preds.map(([a,p])=>`<tr><td>${a}</td><td>${p}</td></tr>`).join("");
    regContent=`<div class="dead-box">
<strong>✗ This is a classification problem — regression gives nonsense results here.</strong>
Our target variable is a <em>category label</em> (1, 2, 3, 4, 5, or 7 — the land cover type). There is no meaningful numeric distance between "Red Soil" and "Cotton Crop."<br><br>
Here's what linear regression actually produces on this data:<br><br>
<strong>R² = ${demo.r2}</strong> (a good regression R² is close to 1.0; this is terrible)<br><br>
<table style="font-size:.82rem;margin:8px 0;border-collapse:collapse">
<tr><th style="padding:3px 10px;background:#f8d7da">Actual class</th><th style="padding:3px 10px;background:#f8d7da">Regression prediction</th></tr>
${predsHtml}
</table>
The model is predicting values like 3.7 or 1.2 — meaningless fractions between categories, with no way to interpret them. Go back and choose <strong>Classification</strong>.
</div>`;
  }

  return `<div class="step-title">Step 5 — What Type of Model?</div>
<div class="step-sub">Before choosing a specific algorithm, you need to decide what kind of prediction problem you're solving. This shapes everything downstream.</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">What are we predicting?</div><div class="panel-body">
<div class="choices">
${card("Classification","We're predicting a discrete category label: which of the 6 land cover types does this pixel belong to?","prob_type","classification")}
${card("Regression","We're predicting a continuous numeric value.","prob_type","regression","dead")}
</div>
${regContent}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Classification vs. Regression</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
<strong>Classification</strong> outputs a class label from a fixed set. Evaluated with accuracy, precision, recall, F1.<br><br>
<strong>Regression</strong> outputs a number on a continuous scale. Evaluated with MSE, RMSE, R².<br><br>
The target variable's type tells you which to use. If the target is a label, category, or group → classification. If it's a measured quantity (price, temperature, count) → regression.
</div></div>
</div>
</div>`;
}

// ── Metric helpers ────────────────────────────────────────────────────────────
function metricLabel(){
  return {accuracy:"Accuracy", macro_f1:"Macro F1", weighted_f1:"Weighted F1"}[state.metric] || "Accuracy";
}

function testSetScore(mk){
  const cm = getPrep().test_cms[mk];
  const n = cm.length;
  const support = cm.map(row=>row.reduce((a,b)=>a+b,0));
  const total = support.reduce((a,b)=>a+b,0);
  if(state.metric === "accuracy"){
    return {score: cm.reduce((s,row,i)=>s+row[i],0)/total};
  }
  const f1s = [];
  for(let i=0;i<n;i++){
    const colSum = cm.reduce((s,row)=>s+row[i],0);
    const p = colSum===0 ? 0 : cm[i][i]/colSum;
    const r = support[i]===0 ? 0 : cm[i][i]/support[i];
    f1s.push((p+r)===0 ? 0 : 2*p*r/(p+r));
  }
  if(state.metric === "macro_f1") return {score: f1s.reduce((a,b)=>a+b,0)/n};
  return {score: f1s.reduce((s,v,i)=>s+v*support[i],0)/total};
}

// ── Step 6 ────────────────────────────────────────────────────────────────────
function step6(){
  const imbalanceNote = `<div class="warn-box" style="margin-top:10px">⚠️ <strong>This dataset is moderately imbalanced.</strong> Cotton Crop and Damp Grey Soil each have about half the samples of Red Soil. Accuracy will be dominated by performance on majority classes — a model could score ~24% by always predicting the most common class. For this dataset, <strong>macro F1</strong> is more appropriate.</div>`;

  return `<div class="step-title">Step 6 — Choose Your Evaluation Metric</div>
<div class="step-sub">Before comparing models you need to decide what "better" means. Your metric choice determines which model wins — and on imbalanced data, different metrics give different rankings.</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Which metric will you use?</div><div class="panel-body">
<div class="choices">
${card("Accuracy","Correct predictions ÷ total predictions. Simple and intuitive, but misleading when classes are imbalanced.","metric","accuracy","bad")}
${card("Macro F1","F1 score averaged equally across all 6 classes — each class counts the same regardless of how many samples it has. Best when every class matters equally.","metric","macro_f1")}
${card("Weighted F1","F1 score averaged weighted by class size. Larger classes contribute more, but smaller classes still count proportionally. A reasonable compromise.","metric","weighted_f1")}
</div>
${state.metric === "accuracy" ? imbalanceNote : ""}
${state.metric ? `<div class="section-label">What is F1?</div>
<p style="font-size:.85rem;color:#4a3060;line-height:1.7">F1 is the harmonic mean of <strong>precision</strong> (of all the pixels you called class X, how many actually were?) and <strong>recall</strong> (of all the real class X pixels, how many did you catch?). F1 = 2·P·R / (P+R). The harmonic mean penalises imbalance between precision and recall — a model that gets 100% recall by predicting everything as one class will still have near-zero F1.</p>` : ""}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Accuracy vs. F1 on Imbalanced Data</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
<strong>Class distribution in this dataset:</strong><br>
Red Soil: 1,531 px &nbsp;·&nbsp; Very Damp Grey Soil: 1,508 px<br>
Grey Soil: 1,356 px &nbsp;·&nbsp; Cotton Crop: 703 px<br>
Damp Grey Soil: 625 px<br><br>
A model that <em>always predicts Red Soil</em> would achieve:<br>
• Accuracy: <strong>23.8%</strong><br>
• Macro F1: <strong>~4%</strong> (near zero — five classes completely missed)<br>
• Weighted F1: <strong>~5%</strong><br><br>
Accuracy lets a lazy model hide behind majority classes. Macro F1 forces it to perform across all six.
</div></div>
<div class="panel"><div class="panel-hdr">Which should you choose?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
<strong>Accuracy</strong> — if all classes are equally common and errors are equally costly. Neither is true here.<br><br>
<strong>Macro F1</strong> — if every class matters equally. This is usually right for multi-class land cover problems, where missing a rare class (Cotton Crop) is just as important as missing a common one.<br><br>
<strong>Weighted F1</strong> — if you care more about getting the common classes right. Reasonable if your downstream application is primarily about the majority surface types.
</div></div>
<div class="panel"><div class="panel-hdr">How F1 Is Calculated</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
For each class, define two quantities from the confusion matrix:<br><br>
<strong>Precision</strong> — of all the pixels you <em>predicted</em> as class X, what fraction actually were?<br>
<code style="background:#ede8f5;padding:2px 6px;border-radius:4px">Precision = TP / (TP + FP)</code><br><br>
<strong>Recall</strong> — of all the pixels that <em>really are</em> class X, what fraction did you catch?<br>
<code style="background:#ede8f5;padding:2px 6px;border-radius:4px">Recall = TP / (TP + FN)</code><br><br>
<strong>F1</strong> is their harmonic mean:<br>
<code style="background:#ede8f5;padding:2px 6px;border-radius:4px">F1 = 2 · Precision · Recall / (Precision + Recall)</code><br><br>
The harmonic mean is used instead of the arithmetic mean because it is dominated by whichever value is <em>lower</em>. A model that recalls every Cotton Crop pixel by labelling <em>everything</em> as Cotton Crop gets Recall = 1.0 but Precision ≈ 0.11 (703 out of 6430). Its F1 ≈ 0.20 — correctly penalised. The arithmetic mean would give a misleading 0.55.<br><br>
<strong>Macro F1</strong> averages the per-class F1 scores with equal weight — every class counts the same.<br>
<strong>Weighted F1</strong> weights each class's F1 by its share of the total samples.
</div></div>
<div class="panel"><div class="panel-hdr">Which Classes Are Costlier to Misclassify?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
Not all misclassifications are equally harmful. Landsat land cover maps are used for agricultural monitoring, irrigation planning, and flood risk assessment — so the cost depends on what the map is <em>for</em>.<br><br>
<strong>Cotton Crop — highest cost.</strong> This is the only agricultural class. Falsely labelling a cotton field as bare soil means missing farmland entirely — a critical error for crop yield estimation, subsidy allocation, and irrigation planning. It is also a minority class (703 samples), so models already receive less training signal for it.<br><br>
<strong>Very Damp Grey Soil — high cost.</strong> Very high moisture can indicate waterlogging or early flooding. Confusing this class with drier soil types understates water saturation — relevant for flood early warning and drainage decisions.<br><br>
<strong>Damp Grey Soil — medium cost.</strong> Confusing it with Very Damp Grey Soil is relatively benign (adjacent moisture levels). Confusing it with dry Grey Soil matters more for irrigation scheduling.<br><br>
<strong>Red Soil and Grey Soil — lower cost.</strong> Bare mineral soil classes matter less for most downstream applications, and as the majority classes, models already perform best on them.<br><br>
<em>This cost asymmetry is part of why Macro F1 is appropriate here: it refuses to let good performance on the easy majority classes mask poor performance on Cotton Crop and the moisture classes.</em>
</div></div>
</div>
</div>`;
}

function step7(){
  const cv_choices=[3,5,10];
  let chartHtml="", tableHtml="", notesHtml="";
  if(state.cv_folds){
    chartHtml=cvBarSvg(String(state.cv_folds));
    const d=getPrep().cv[String(state.cv_folds)];
    const sorted=["lr","knn","dt","nb","rf"].sort((a,b)=>d[b].mean-d[a].mean);
    tableHtml=`<table class="cm" style="margin-top:10px"><thead><tr>
<th>Model</th><th>${state.cv_folds}-fold CV Mean</th><th>Std Dev</th></tr></thead><tbody>`;
    sorted.forEach(mk=>{
      tableHtml+=`<tr><td style="font-weight:700;color:${MODEL_COLORS[mk]}">${MODEL_NAMES[mk]}</td>
<td>${d[mk].mean.toFixed(3)}</td><td>± ${d[mk].std.toFixed(3)}</td></tr>`;
    });
    tableHtml+=`<tr style="color:#8b6aaa;font-style:italic"><td>Dummy baseline</td><td>${DUMMY_BASELINE.accuracy.toFixed(3)}</td><td>—</td></tr>`;
    tableHtml+="</tbody></table>";
    notesHtml=`<p style="font-size:.82rem;color:#8b6aaa;margin-top:8px">Error bars show ± 1 std dev across folds. Tight bars = consistent model. Wide bars = results depended heavily on which fold was held out.</p>`;
  }

  const metricLbl = metricLabel();
  const usingCV = state.metric === "accuracy";
  let scoreHtml = "";
  if(state.cv_folds){
    if(usingCV){
      scoreHtml = `<div class="section-label">CV ${metricLbl} — All Models</div>${chartHtml}${tableHtml}${notesHtml}`;
    } else {
      const models = ["lr","knn","dt","nb","rf"];
      const scores = models.map(mk=>({mk, ...testSetScore(mk)}));
      scores.sort((a,b)=>b.score-a.score);
      const minV=0.0, maxV=1.0, W=460, padL=100, padR=60, barH=28, gap=10, padT=20;
      const scale=W-padL-padR;
      const H=padT+(barH+gap)*(models.length+1);
      let svg=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px">`;
      scores.forEach(({mk,score},i)=>{
        const y=padT+i*(barH+gap);
        const bw=Math.max(0,(score-minV)/(maxV-minV)*scale);
        svg+=`<rect x="${padL}" y="${y}" width="${bw}" height="${barH}" fill="${MODEL_COLORS[mk]}" rx="4"/>`;
        svg+=`<text x="${padL-6}" y="${y+barH/2+4}" text-anchor="end" font-size="12" fill="#4a3060">${MODEL_NAMES[mk]}</text>`;
        svg+=`<text x="${padL+bw+6}" y="${y+barH/2+4}" font-size="12" font-weight="bold" fill="#3a1a5c">${score.toFixed(3)}</text>`;
      });
      const f1BaseVal = state.metric==="macro_f1" ? DUMMY_BASELINE.macro_f1 : DUMMY_BASELINE.weighted_f1;
      const f1by=padT+models.length*(barH+gap);
      const f1bbw=Math.max(0,(f1BaseVal-minV)/(maxV-minV)*scale);
      svg+=`<rect x="${padL}" y="${f1by}" width="${f1bbw}" height="${barH}" fill="#c8c0d8" rx="4" stroke="#8b6aaa" stroke-width="1.5" stroke-dasharray="4 3"/>`;
      svg+=`<text x="${padL-6}" y="${f1by+barH/2+4}" text-anchor="end" font-size="11" fill="#8b6aaa">Dummy baseline</text>`;
      svg+=`<text x="${padL+f1bbw+6}" y="${f1by+barH/2+4}" font-size="12" font-weight="bold" fill="#8b6aaa">${f1BaseVal.toFixed(3)}</text>`;
      svg+=`<line x1="${padL}" x2="${padL}" y1="${padT-4}" y2="${H-4}" stroke="#c0a8e0" stroke-width="1"/></svg>`;
      const tbl = `<table class="cm" style="margin-top:10px"><thead><tr><th>Model</th><th>Test-set ${metricLbl}</th></tr></thead><tbody>`
        + scores.map(({mk,score})=>`<tr><td style="font-weight:700;color:${MODEL_COLORS[mk]}">${MODEL_NAMES[mk]}</td><td>${score.toFixed(3)}</td></tr>`).join("")
        + `<tr style="color:#8b6aaa;font-style:italic"><td>Dummy baseline</td><td>${f1BaseVal.toFixed(3)}</td></tr>`
        + "</tbody></table>";
      scoreHtml = `<div class="section-label">Test-set ${metricLbl} — All Models</div>
        <div class="barchart">${svg}</div>${tbl}
        <div class="warn-box" style="margin-top:10px">⚠️ We pre-computed cross-validation for <strong>accuracy only</strong>. For macro/weighted F1, these are test-set scores — no standard deviation is available, but the relative model ranking is still informative.</div>`;
    }
  }

  return `<div class="step-title">Step 7 — Cross-Validate All Models</div>
<div class="step-sub">Cross-validation gives a reliable estimate of how well each model will generalize. We train and evaluate each model ${state.cv_folds||"k"} times, each time holding out a different fold as the test set. Metric in use: <strong>${metricLbl}</strong>.<br><br>
Note: CV is run on the <strong>training set only</strong>. The held-out test set (20%) stays sealed until the very end.</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">How many folds?</div><div class="panel-body">
<div class="choices" style="flex-direction:row">
${[3,5,10].map(k=>`<div class="choice-card${state.cv_folds===k?" selected":""}" onclick="choose('cv_folds',${k})" style="flex:1;text-align:center">
<div class="choice-title">${k}-fold</div>
<div class="choice-body">${k===3?"Faster, less variance":""}${k===5?"Good balance (default)":""}${k===10?"More stable estimates, slower":""}</div>
</div>`).join("")}
</div>
${state.cv_folds ? scoreHtml : ""}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Bias vs. Variance in k-fold CV</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
<strong>More folds:</strong> each training set is larger (less bias in the estimate), but fold results are more correlated (more variance between runs). Also slower.<br><br>
<strong>Fewer folds:</strong> each training set is smaller (slightly biased toward underestimating generalization), but more independent fold evaluations.<br><br>
5-fold or 10-fold is the community standard. 3-fold is sometimes used when the dataset is large and runtime matters.
</div></div>
<div class="panel"><div class="panel-hdr">Why No SVM?</div><div class="panel-body" style="font-size:.88rem;line-height:1.8;color:#4a3060">
Support Vector Machines with an RBF kernel were excluded from this adventure for a concrete reason worth understanding:<br><br>
<strong>Memory:</strong> An RBF kernel SVM stores an <em>n × n</em> kernel matrix during training — one float per pair of training samples. On this 6,430-sample dataset that's 6,430² ≈ <strong>41 million floats, or ~330 MB per fit</strong>. With a 16-combination hyperparameter grid × 5 CV folds × 6 preprocessing combos, the original generator attempted ~480 SVM fits. Peak memory usage reached <strong>~40 GB</strong>, crashing the host machine.<br><br>
<strong>Runtime:</strong> SVM fitting is O(n²)–O(n³) in training samples. The full grid search took <strong>~13 of the original generator's 15 minutes</strong>. Without SVM, the generator runs in roughly <strong>2 minutes</strong>.<br><br>
<strong>Accuracy:</strong> SVM was not even the best model — Random Forest and Logistic Regression outperformed it on this dataset while being orders of magnitude cheaper to train.<br><br>
This is a real production lesson: <em>a technically correct algorithm can be practically unusable at scale</em>. SVMs are powerful on small, high-dimensional datasets (genomics, text classification). On larger tabular datasets, tree ensembles usually beat them at a fraction of the cost.
</div></div>
</div>
</div>`;
}

function step8(){
  if(!state.cv_folds || !state.scaler || !state.stratify){
    return `<div class="step-title">Step 8 — Select a Model to Tune</div>
<div class="step-sub" style="color:#dc2626">Please complete the previous steps before reaching this one.</div>`;
  }
  const metricLbl = metricLabel();
  const usingCV = state.metric === "accuracy";
  const d = getPrep().cv[String(state.cv_folds)];
  const cms = getPrep().test_cms;

  // Sort models by chosen metric
  const modelScore = mk => usingCV ? d[mk].mean : testSetScore(mk).score;
  const sorted = ["lr","knn","dt","nb","rf"].sort((a,b)=>modelScore(b)-modelScore(a));

  let cardsHtml = sorted.map(mk=>{
    const sel = state.model===mk?" selected":"";
    const scoreStr = usingCV
      ? `CV ${metricLbl}: <strong>${d[mk].mean.toFixed(3)} ± ${d[mk].std.toFixed(3)}</strong>`
      : `Test-set ${metricLbl}: <strong>${testSetScore(mk).score.toFixed(3)}</strong>`;
    return `<div class="choice-card${sel}" onclick="choose('model','${mk}')" style="border-left:4px solid ${MODEL_COLORS[mk]}">
<div class="choice-title">${MODEL_NAMES[mk]}</div>
<div class="choice-body">${scoreStr}</div>
</div>`;
  }).join("");

  let cmHtml="";
  if(state.model){
    cmHtml=`<div class="section-label">Test-Set Confusion Matrix — ${MODEL_NAMES[state.model]}</div>
${cmTable(cms[state.model], DATA.overview.class_names)}
<p style="font-size:.82rem;color:#8b6aaa;margin-top:6px">Rows = actual class. Columns = predicted class. Green diagonal = correct predictions. Purple off-diagonal = errors. Evaluated on the held-out 20% test set with default hyperparameters.</p>`;
  }

  return `<div class="step-title">Step 8 — Select a Model to Tune</div>
<div class="step-sub">Based on your cross-validation results, pick the model you want to tune. Models are ranked by <strong>${metricLbl}</strong> — your chosen metric. Consider both score and consistency (std dev where available). A slightly lower mean with much tighter std dev can be the better real-world choice.</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Choose your model</div><div class="panel-body">
<div class="choices">${cardsHtml}</div>
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Confusion Matrix Preview</div><div class="panel-body">
${state.model ? cmHtml : "<p style='color:#8b6aaa;font-size:.88rem'>Select a model to see its confusion matrix.</p>"}
</div></div>
</div>
</div>`;
}

function learningCurvePanel(){
  const lc = DATA.learning_curves && DATA.learning_curves[getKey()];
  if(!lc) return "";
  const mk = state.model;
  const d = lc[mk];
  if(!d) return "";
  const W=460, H=200, padL=44, padR=20, padT=16, padB=36, scale_x=W-padL-padR, scale_y=H-padT-padB;
  const xs = d.sizes;
  const minY=0.50, maxY=1.0;
  const px = v => padL + ((v - xs[0]) / (xs[xs.length-1] - xs[0])) * scale_x;
  const py = v => padT + (1 - (v - minY) / (maxY - minY)) * scale_y;
  const TRAIN_COL = MODEL_COLORS[mk];
  const VAL_COL   = "#8b6aaa";
  // build path strings
  const trainPath = d.train_mean.map((v,i)=>`${i===0?"M":"L"}${px(xs[i]).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
  const valPath   = d.val_mean.map((v,i)=>`${i===0?"M":"L"}${px(xs[i]).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
  // error band polygons
  const trainTop  = d.train_mean.map((v,i)=>  `${px(xs[i]).toFixed(1)},${py(v+d.train_std[i]).toFixed(1)}`).join(" ");
  const trainBot  = [...d.train_mean].reverse().map((v,i)=> `${px(xs[d.train_mean.length-1-i]).toFixed(1)},${py(v-d.train_std[d.train_mean.length-1-i]).toFixed(1)}`).join(" ");
  const valTop    = d.val_mean.map((v,i)=>    `${px(xs[i]).toFixed(1)},${py(v+d.val_std[i]).toFixed(1)}`).join(" ");
  const valBot    = [...d.val_mean].reverse().map((v,i)=>   `${px(xs[d.val_mean.length-1-i]).toFixed(1)},${py(v-d.val_std[d.val_mean.length-1-i]).toFixed(1)}`).join(" ");
  // y-axis ticks
  const yTicks = [0.5,0.6,0.7,0.8,0.9,1.0];
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px">`;
  yTicks.forEach(t=>{
    const y=py(t);
    if(y<padT||y>padT+scale_y) return;
    svg+=`<line x1="${padL}" x2="${padL+scale_x}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#e8e0f0" stroke-width="1"/>`;
    svg+=`<text x="${padL-4}" y="${(y+4).toFixed(1)}" text-anchor="end" font-size="10" fill="#8b6aaa">${t.toFixed(1)}</text>`;
  });
  // error bands
  svg+=`<polygon points="${trainTop} ${trainBot}" fill="${TRAIN_COL}" opacity="0.15"/>`;
  svg+=`<polygon points="${valTop} ${valBot}" fill="${VAL_COL}" opacity="0.15"/>`;
  // lines
  svg+=`<path d="${trainPath}" stroke="${TRAIN_COL}" stroke-width="2" fill="none"/>`;
  svg+=`<path d="${valPath}" stroke="${VAL_COL}" stroke-width="2" fill="none" stroke-dasharray="5 3"/>`;
  // x-axis labels
  xs.forEach(s=>{
    svg+=`<text x="${px(s).toFixed(1)}" y="${(padT+scale_y+14).toFixed(1)}" text-anchor="middle" font-size="10" fill="#8b6aaa">${s}</text>`;
  });
  svg+=`<line x1="${padL}" x2="${padL}" y1="${padT}" y2="${padT+scale_y}" stroke="#c0a8e0" stroke-width="1"/>`;
  svg+=`<line x1="${padL}" x2="${padL+scale_x}" y1="${(padT+scale_y).toFixed(1)}" y2="${(padT+scale_y).toFixed(1)}" stroke="#c0a8e0" stroke-width="1"/>`;
  svg+=`<text x="${(padL+scale_x/2).toFixed(1)}" y="${H}" text-anchor="middle" font-size="10" fill="#8b6aaa">Training set size</text>`;
  svg+=`</svg>`;
  const legend = `<div style="font-size:.8rem;display:flex;gap:16px;margin-bottom:4px">
    <span style="display:inline-flex;align-items:center;gap:4px"><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="${TRAIN_COL}" stroke-width="2"/></svg>Training</span>
    <span style="display:inline-flex;align-items:center;gap:4px"><svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5" stroke="${VAL_COL}" stroke-width="2" stroke-dasharray="5 3"/></svg>Validation (CV)</span>
  </div>`;
  // diagnosis
  const finalGap = d.train_mean[4] - d.val_mean[4];
  let diagnosis = "";
  if(finalGap > 0.10)
    diagnosis = `<strong>Overfitting detected</strong> — training accuracy (${d.train_mean[4].toFixed(3)}) is much higher than validation (${d.val_mean[4].toFixed(3)}). The model memorises training data instead of generalising. Try regularisation, reducing depth/complexity, or adding more data.`;
  else if(d.val_mean[4] < 0.75)
    diagnosis = `<strong>Possible underfitting</strong> — both training and validation accuracy are low. The model may not have enough capacity for this data. Try a more complex model or additional features.`;
  else if(d.val_mean[4] > d.val_mean[3] + 0.005)
    diagnosis = `<strong>Still learning</strong> — validation accuracy is still rising. More training data would likely continue to help.`;
  else
    diagnosis = `<strong>Well-converged</strong> — train and validation curves are close and flat. More data is unlikely to help much; focus on model complexity or feature engineering instead.`;
  return `<div class="panel"><div class="panel-hdr">Learning Curve — ${MODEL_NAMES[mk]}</div><div class="panel-body">
${legend}${svg}
<p style="font-size:.82rem;color:#8b6aaa;margin-top:8px">${diagnosis}</p>
</div></div>`;
}

function step9(){
  if(!state.model) return `<div class="step-title">Step 9 — Hyperparameter Tuning</div><div class="step-sub" style="color:#dc2626">Please select a model first.</div>`;
  const hp=getPrep().hp[state.model];
  const best=hp.results[hp.best_idx];
  const sel=state.hp_idx!==null ? hp.results[state.hp_idx] : null;
  const dispIdx=state.hp_idx!==null ? state.hp_idx : hp.best_idx;
  const disp=hp.results[dispIdx];

  const paramsStr=Object.entries(disp.params).map(([k,v])=>`${k}=${v==="null"?"None":v}`).join(", ");
  const bestStr=Object.entries(best.params).map(([k,v])=>`${k}=${v==="null"?"None":v}`).join(", ");

  const mkName=MODEL_NAMES[state.model];
  const mkKey=state.model;

  let codeStr="";
  if(mkKey==="lr") codeStr=`<span class="code-kw">from</span> sklearn.linear_model <span class="code-kw">import</span> LogisticRegression\nclf = LogisticRegression(${paramsStr}, max_iter=2000, solver=<span class="code-str">'saga'</span>)`;
  else if(mkKey==="knn") codeStr=`<span class="code-kw">from</span> sklearn.neighbors <span class="code-kw">import</span> KNeighborsClassifier\nclf = KNeighborsClassifier(${paramsStr})`;
  else if(mkKey==="dt") codeStr=`<span class="code-kw">from</span> sklearn.tree <span class="code-kw">import</span> DecisionTreeClassifier\nclf = DecisionTreeClassifier(${paramsStr}, random_state=42)`;
  else if(mkKey==="nb") codeStr=`<span class="code-kw">from</span> sklearn.naive_bayes <span class="code-kw">import</span> GaussianNB\nclf = GaussianNB(${paramsStr})`;
  else if(mkKey==="rf") codeStr=`<span class="code-kw">from</span> sklearn.ensemble <span class="code-kw">import</span> RandomForestClassifier\nclf = RandomForestClassifier(${paramsStr}, random_state=42)`;
  codeStr+=`\nclf.fit(X_train, y_train)\ny_pred = clf.predict(X_test)\n<span class="code-cm"># Accuracy: ${disp.score.toFixed(3)}</span>`;

  return `<div class="step-title">Step 9 — Hyperparameter Tuning (${mkName})</div>
<div class="step-sub">The default model uses arbitrary hyperparameter values. We've pre-evaluated every combination on the 20% test set. Click any cell in the heatmap to see that configuration's confusion matrix and scores. The ★ marks the best-performing combination.</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Accuracy Heatmap — Click to Explore</div><div class="panel-body">
<div style="overflow-x:auto">${hpHeatmap(state.model)}</div>
<p style="font-size:.78rem;color:#8b6aaa;margin-top:6px">★ = best accuracy. ${hp.hue ? `Rows = ${hp.hue}, Columns = ${hp.x}.` : `X-axis = ${hp.x}.`} Colors map low (dark purple) to high (light).</p>
<div class="section-label">Best combination found</div>
<div style="font-size:.88rem;color:#4a3060">${bestStr} — accuracy <strong>${best.score.toFixed(3)}</strong></div>
</div></div>
<div class="panel"><div class="panel-hdr">Selected: ${paramsStr} — Accuracy ${disp.score.toFixed(3)}</div>
<div class="panel-body">${codeBlock(codeStr)}</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Confusion Matrix — ${paramsStr}</div>
<div class="panel-body">${cmTable(disp.cm, DATA.overview.class_names)}</div></div>
<div class="panel"><div class="panel-hdr">Precision &amp; Recall per Class</div>
<div class="panel-body" style="max-height:380px;overflow-y:auto">${prBars(disp.precision, disp.recall, DATA.overview.class_names)}</div></div>
${learningCurvePanel()}
</div>
</div>`;
}

function step10(){
  if(!state.model || state.hp_idx===null) return `<div class="step-title">Step 10 — Error Analysis</div><div class="step-sub" style="color:#dc2626">Please complete the previous steps.</div>`;
  const hp=getPrep().hp[state.model];
  const res=hp.results[state.hp_idx];
  const params=Object.entries(res.params).map(([k,v])=>`${k}=${v==="null"?"None":v}`).join(", ");

  // Find which classes have worst recall
  const rcVals=res.recall.map((r,i)=>({name:DATA.overview.class_names[i],recall:r,i}));
  rcVals.sort((a,b)=>a.recall-b.recall);
  const worst=rcVals.slice(0,2);

  // Find biggest off-diagonal confusion
  const cm=res.cm; const n=cm.length;
  let topConfusions=[];
  for(let i=0;i<n;i++) for(let j=0;j<n;j++){
    if(i!==j && cm[i][j]>0) topConfusions.push({actual:DATA.overview.class_names[i],pred:DATA.overview.class_names[j],count:cm[i][j]});
  }
  topConfusions.sort((a,b)=>b.count-a.count);
  const top3=topConfusions.slice(0,3);

  return `<div class="step-title">Step 10 — Error Analysis</div>
<div class="step-sub">You've trained and tuned your final model: <strong>${MODEL_NAMES[state.model]}</strong> with <strong>${params}</strong>, achieving <strong>${res.score.toFixed(3)} test accuracy</strong>. Now: what mistakes does it make, and why?</div>
<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Final Confusion Matrix</div><div class="panel-body">
${cmTable(res.cm, DATA.overview.class_names)}
<p style="font-size:.82rem;color:#8b6aaa;margin-top:6px">Green diagonal = correct. Purple off-diagonal = errors. Darker purple = more frequent confusion.</p>
</div></div>
<div class="panel"><div class="panel-hdr">Top Confusions</div><div class="panel-body">
${top3.map((c,i)=>`<div style="margin-bottom:10px;padding:10px 12px;background:#fef3f2;border-radius:6px;font-size:.87rem;line-height:1.6">
<strong>${i+1}. Predicted "${c.pred}" for actual "${c.actual}"</strong> — ${c.count} times<br>
<span style="color:#6b4a8a">${confusionNote(c.actual, c.pred)}</span>
</div>`).join("")}
</div></div>
</div>
<div class="col-right">
<div class="panel"><div class="panel-hdr">Per-Class Precision &amp; Recall</div>
<div class="panel-body" style="max-height:400px;overflow-y:auto">${prBars(res.precision, res.recall, DATA.overview.class_names)}</div></div>
${featureImportancePanel()}
<div class="panel"><div class="panel-hdr">What Could You Try Next?</div><div class="panel-body" style="font-size:.88rem;line-height:1.9;color:#4a3060">
<p style="margin-bottom:10px">The grey soil moisture classes are where your model struggles most. In the next step you'll try three targeted interventions and see live before/after results:</p>
<div style="margin-bottom:8px">🔧 <strong>Feature engineering</strong> — add an NDMI moisture index (B3−B4)/(B3+B4) per pixel, giving the model a physics-informed shortcut instead of making it infer the ratio from raw band values.</div>
<div style="margin-bottom:8px">⚖️ <strong>Class weighting</strong> — retrain with <code>class_weight='balanced'</code> so errors on harder moisture classes cost more during training.</div>
<div style="margin-bottom:8px">🌲 <strong>Hierarchical classification</strong> — decompose into "grey vs. not-grey" then "which moisture level," so each classifier tackles a simpler problem.</div>
<div style="color:#5a2d82;font-weight:600;margin-top:8px">Click Continue to try them interactively →</div>
</div></div>
</div>
</div>`;
}

// ── Pixel neighbourhood visualizer ───────────────────────────────────────────
// feat: 36 raw band values [B1-Px1,B2-Px1,B3-Px1,B4-Px1, B1-Px2, …]
// Returns {fc, mo} SVG strings: false-colour and B4-moisture grayscale.
function pixelGrid(feat, cell=30, gap=3){
  const total = 3*cell + 2*gap;
  const b = [[],[],[],[]];
  for(let p=0;p<9;p++) for(let i=0;i<4;i++) b[i].push(feat[p*4+i]);
  const mn = b.map(a=>Math.min(...a));
  const mx = b.map(a=>Math.max(...a));
  const norm = (v,i)=> mx[i]===mn[i] ? 128 : Math.round((v-mn[i])/(mx[i]-mn[i])*255);

  let fc=`<svg width="${total}" height="${total}" style="border-radius:4px;display:block">`;
  let mo=`<svg width="${total}" height="${total}" style="border-radius:4px;display:block">`;
  for(let p=0;p<9;p++){
    const row=Math.floor(p/3), col=p%3;
    const x=col*(cell+gap), y=row*(cell+gap);
    const [n1,n2,n3,n4]=[0,1,2,3].map(i=>norm(feat[p*4+i],i));
    const ctr = p===4;
    // False colour: R=B3(NIR), G=B2(red), B=B1(green)
    fc+=`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="rgb(${n3},${n2},${n1})" rx="1"/>`;
    if(ctr) fc+=`<rect x="${x+cell/2-3}" y="${y+cell/2-3}" width="7" height="7" fill="none" stroke="white" stroke-width="2" rx="1"/>`;
    // B4 moisture: dark=low B4=wet, light=high B4=dry (grayscale)
    mo+=`<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="rgb(${n4},${n4},${n4})" rx="1"/>`;
    if(ctr) mo+=`<rect x="${x+cell/2-3}" y="${y+cell/2-3}" width="7" height="7" fill="none" stroke="white" stroke-width="2" rx="1"/>`;
  }
  fc+="</svg>"; mo+="</svg>";
  return {fc, mo};
}

// ── Feature importance panel (used in step 10) ────────────────────────────────
function featureImportancePanel(){
  const fi = DATA.feature_importances && DATA.feature_importances[getKey()];
  if(!fi) return "";
  const mk = (state.model==="rf"||state.model==="dt") ? state.model : "rf";
  const imps = fi[mk];
  const fnames = DATA.overview.feature_names;
  // Sort indices descending by importance
  const order = imps.map((_,i)=>i).sort((a,b)=>imps[b]-imps[a]);
  const BAND_COLORS = ["#3b82f6","#10b981","#f59e0b","#8b5cf6"];
  const W=460, barH=14, gap=3, padL=82, padR=55, padT=16;
  const maxImp = imps[order[0]];
  const scale = W-padL-padR;
  const H = padT + (barH+gap)*fnames.length;
  let svg=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px">`;
  order.forEach((fi_idx, row)=>{
    const y = padT + row*(barH+gap);
    const band = fi_idx % 4;  // B1=0,B2=1,B3=2,B4=3
    const bw = Math.max(1, (imps[fi_idx]/maxImp)*scale);
    svg+=`<rect x="${padL}" y="${y}" width="${bw}" height="${barH}" fill="${BAND_COLORS[band]}" rx="2" opacity="0.85"/>`;
    svg+=`<text x="${padL-4}" y="${y+barH/2+4}" text-anchor="end" font-size="10" fill="#4a3060">${fnames[fi_idx]}</text>`;
    svg+=`<text x="${padL+bw+4}" y="${y+barH/2+4}" font-size="10" fill="#3a1a5c">${imps[fi_idx].toFixed(3)}</text>`;
  });
  svg+=`<line x1="${padL}" x2="${padL}" y1="${padT-4}" y2="${H-4}" stroke="#c0a8e0" stroke-width="1"/>`;
  svg+=`</svg>`;
  const noteModel = (state.model!=="rf"&&state.model!=="dt")
    ? `<p style="font-size:.8rem;color:#8b6aaa;margin-top:6px">Your selected model (${MODEL_NAMES[state.model]}) doesn't expose feature importances directly. Showing Random Forest importances as a reference.</p>`
    : "";
  const legend = BAND_COLORS.map((c,i)=>
    `<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px"><svg width="12" height="12"><rect width="12" height="12" fill="${c}" rx="2"/></svg>B${i+1}</span>`
  ).join("");
  return `<div class="panel"><div class="panel-hdr">Feature Importances — ${mk==="rf"?"Random Forest":"Decision Tree"}</div><div class="panel-body">
<div style="font-size:.82rem;margin-bottom:6px">${legend}</div>
<div style="max-height:420px;overflow-y:auto">${svg}</div>
${noteModel}
<p style="font-size:.82rem;color:#8b6aaa;margin-top:8px"><strong>Px5 (centre pixel) dominates positionally</strong> — the labelled pixel always carries more information than its neighbours, though the surrounding pixels do add context. Across spectral bands, visible green (B1) and red (B2) contribute most in absolute terms, but <strong>B4 — shortwave-infrared (1.55–1.75 µm)</strong> packs the most moisture-specific signal per feature: water absorbs strongly at SWIR wavelengths, making B4 the key discriminator between dry, damp, and very damp grey soil classes. This is exactly why the NDMI feature engineering in step 11 (which uses B3 and B4 together) helps the model.</p>
</div></div>`;
}

// ── Step 10 ───────────────────────────────────────────────────────────────────
function step11(){
  const o = DATA.overview;
  const imp = DATA.improvements[getKey()];
  const mk  = state.model || "rf";
  const mkName = MODEL_NAMES[mk];
  const baseHp = getPrep().hp[mk];
  const baseRes = state.hp_idx !== null ? baseHp.results[state.hp_idx] : baseHp.results[baseHp.best_idx];

  // Grey class indices in CLASS_LABELS: 3→idx2, 4→idx3, 7→idx5
  const greyIdx = [2, 3, 5];
  const greyNames = ["Grey Soil", "Damp Grey Soil", "Very Damp Grey Soil"];

  // ── misclassified sample cards ─────────────────────────────────────────────
  const samples = (imp && imp.misclassified_grey) || [];
  const sampleCards = samples.map(s=>{
    const {fc, mo} = pixelGrid(s.features_raw);
    return `<div style="display:inline-block;vertical-align:top;margin-right:14px;margin-bottom:10px;padding:10px 12px;background:#fff;border-radius:10px;box-shadow:0 1px 8px rgba(0,0,0,.1);text-align:center">
      <div style="font-size:.75rem;font-weight:700;color:#5a2d82;margin-bottom:6px;max-width:110px">${s.actual}<br><span style="color:#dc2626">→ predicted ${s.pred}</span></div>
      <div style="display:flex;gap:5px;justify-content:center">
        <div><div style="font-size:.65rem;color:#8b6aaa;margin-bottom:2px">False Color</div>${fc}</div>
        <div><div style="font-size:.65rem;color:#8b6aaa;margin-bottom:2px">B4 Moisture</div>${mo}</div>
      </div>
      <div style="font-size:.62rem;color:#8b6aaa;margin-top:4px">□ = centre pixel</div>
    </div>`;
  }).join("");

  // ── results panel (appears after strategy chosen) ──────────────────────────
  let resultsHtml = "";
  if(state.improvement){
    const strat = state.improvement;

    if(strat === "hierarchical"){
      const h = imp.hierarchical;
      const compRows = greyNames.map((nm,i)=>{
        const bef = baseRes.recall[greyIdx[i]], aft = h.recall[greyIdx[i]];
        const delta = aft - bef;
        const col = delta > 0.005 ? "#16a34a" : delta < -0.005 ? "#dc2626" : "#6b7280";
        return `<tr>
          <td style="padding:4px 10px;font-size:.82rem;font-weight:600;color:#5a2d82">${nm}</td>
          <td style="padding:4px 10px;text-align:center;font-size:.82rem">${bef.toFixed(3)}</td>
          <td style="padding:4px 10px;text-align:center;font-size:.82rem;font-weight:700;color:${col}">${aft.toFixed(3)} (${delta>=0?"+":""}${delta.toFixed(3)})</td>
        </tr>`;
      }).join("");
      resultsHtml = `
      <div class="panel"><div class="panel-hdr">Results — Hierarchical Classifier (Random Forest)</div><div class="panel-body">
        <p style="font-size:.87rem;color:#4a3060;margin-bottom:10px">
          Stage 1 binary accuracy (grey vs. not-grey): <strong>${h.stage1_accuracy.toFixed(3)}</strong><br>
          Combined final accuracy: <strong>${h.score.toFixed(3)}</strong> vs. baseline <strong>${baseRes.score.toFixed(3)}</strong>
          <span class="info-tag ${h.score>baseRes.score?'tag-g':'tag-r'}">${h.score>baseRes.score?'▲':'▼'} ${Math.abs(h.score-baseRes.score).toFixed(3)}</span>
        </p>
        <div class="section-label">Grey Soil Recall — Before vs After</div>
        <table class="cm" style="margin-bottom:12px"><thead><tr>
          <th>Class</th><th>Before</th><th>After (Hierarchical)</th>
        </tr></thead><tbody>${compRows}</tbody></table>
        <div class="section-label">Confusion Matrix (After)</div>
        ${cmTable(h.cm, o.class_names)}
        <div class="section-label">Precision &amp; Recall per Class (After)</div>
        ${prBars(h.precision, h.recall, o.class_names)}
        <p style="font-size:.8rem;color:#8b6aaa;margin-top:8px"><strong>Key insight:</strong> Errors compound in a hierarchy — a non-grey pixel Stage 1 misclassifies as grey will be routed to the grey subclassifier and certainly misclassified. Stage 1 accuracy is the ceiling for combined performance.</p>
      </div></div>`;

    } else {
      let newRes = null, stratNote = "", unsupported = false;
      if(strat === "ndmi"){
        newRes = imp.ndmi[mk];
        stratNote = `Added 9 NDMI features — (B3−B4)/(B3+B4) per pixel — giving <strong>${mkName}</strong> 45 inputs instead of 36. These ratios directly encode how much shortwave-IR soil moisture suppresses relative to near-IR.`;
      } else if(strat === "smote"){
        if(!imp.smote){
          unsupported = true;
        } else {
          newRes = imp.smote[mk];
          stratNote = `Applied SMOTE before training <strong>${mkName}</strong>. Synthetic samples were generated by interpolating between existing minority-class examples in feature space, balancing the class distribution without discarding majority-class data. Unlike class weighting, SMOTE changes what the model sees rather than how it scores errors.`;
        }
      } else {
        if(imp.weighted[mk] && imp.weighted[mk].unsupported){
          unsupported = true;
        } else {
          newRes = imp.weighted[mk];
          stratNote = `Retrained <strong>${mkName}</strong> with <code>class_weight='balanced'</code>. The model now penalises errors on underrepresented classes more during training, shifting decision boundaries toward the harder moisture categories.`;
        }
      }

      if(unsupported){
        const warnMsg = strat==="smote"
          ? `⚠️ <strong>SMOTE results not yet computed.</strong> Run <code>python compute_missing.py</code> (after <code>pip install imbalanced-learn</code>), then regenerate the HTML.`
          : `⚠️ <strong>${state.model==="nb"?"Naive Bayes":"k-NN"} does not support <code>class_weight</code>.</strong> Go back to Step 7 and choose a different model, or try a different strategy here.`;
        resultsHtml = `<div class="warn-box">${warnMsg}</div>`;
      } else if(newRes){
        const compRows = greyNames.map((nm,i)=>{
          const bef=baseRes.recall[greyIdx[i]], aft=newRes.recall[greyIdx[i]];
          const delta=aft-bef, col=delta>0.005?"#16a34a":delta<-0.005?"#dc2626":"#6b7280";
          return `<tr>
            <td style="padding:4px 10px;font-size:.82rem;font-weight:600;color:#5a2d82">${nm}</td>
            <td style="padding:4px 10px;text-align:center;font-size:.82rem">${bef.toFixed(3)}</td>
            <td style="padding:4px 10px;text-align:center;font-size:.82rem;font-weight:700;color:${col}">${aft.toFixed(3)} (${delta>=0?"+":""}${delta.toFixed(3)})</td>
          </tr>`;
        }).join("");
        resultsHtml = `
        <div class="panel"><div class="panel-hdr">Results — ${{ndmi:"NDMI Feature Engineering",weighted:"Class Weighting",smote:"SMOTE Oversampling"}[strat]}</div><div class="panel-body">
          <p style="font-size:.87rem;color:#4a3060;margin-bottom:10px">${stratNote}</p>
          <p style="font-size:.87rem;color:#4a3060;margin-bottom:12px">
            New accuracy: <strong>${newRes.score.toFixed(3)}</strong> vs. baseline <strong>${baseRes.score.toFixed(3)}</strong>
            <span class="info-tag ${newRes.score>baseRes.score?'tag-g':'tag-r'}">${newRes.score>baseRes.score?'▲':'▼'} ${Math.abs(newRes.score-baseRes.score).toFixed(3)}</span>
          </p>
          <div class="section-label">Grey Soil Recall — Before vs After</div>
          <table class="cm" style="margin-bottom:12px"><thead><tr>
            <th>Class</th><th>Before</th><th>After</th>
          </tr></thead><tbody>${compRows}</tbody></table>
          <div class="section-label">Confusion Matrix (After)</div>
          ${cmTable(newRes.cm, o.class_names)}
          <div class="section-label">Precision &amp; Recall per Class (After)</div>
          ${prBars(newRes.precision, newRes.recall, o.class_names)}
        </div></div>`;
      }
    }
  }

  return `<div class="step-title">Step 11 — Try to Improve</div>
<div class="step-sub">You've diagnosed the problem: grey soil moisture levels are the hardest to separate. Now try one of four real interventions and see whether it moves the needle on exactly those classes.</div>

<div class="panel"><div class="panel-hdr">Misclassified Grey Soil Pixels — What Does the Model See?</div><div class="panel-body">
<p style="font-size:.85rem;color:#4a3060;margin-bottom:10px">Each card is a real pixel your model got wrong. <strong>Left:</strong> false-color composite (NIR→red channel, Red→green, Green→blue — healthy vegetation appears bright red). <strong>Right:</strong> Band 4 shortwave-infrared intensity, grayscale — <em>darker = more moisture absorbed = wetter soil</em>. The □ marks the centre pixel being classified.</p>
<div style="overflow-x:auto;white-space:nowrap;padding-bottom:8px">${sampleCards}</div>
<p style="font-size:.8rem;color:#8b6aaa;margin-top:4px">Notice how similar the B4 moisture views look between "Damp" and "Very Damp" pixels? The model is operating at the edge of what these 36 features can resolve. That's not a bug — it's physics.</p>
</div></div>

<div class="two-col">
<div class="col-left">
<div class="panel"><div class="panel-hdr">Choose a Strategy</div><div class="panel-body">
<div class="choices">
${card(`Feature Engineering — Add <span class="tip" data-tip="Normalized Difference Moisture Index. A spectral ratio (B3−B4)/(B3+B4) that isolates soil moisture signal: water molecules absorb shortwave-infrared (B4) strongly, so wet soil has lower B4 relative to near-infrared (B3), producing a lower NDMI. The normalization keeps the index in the range −1 to +1 regardless of overall brightness.">NDMI</span> Index`,
  "Compute (B3−B4)/(B3+B4) for each of the 9 pixels. This ratio directly encodes soil moisture: wet soil absorbs more B4, lowering the ratio. Gives the model a physics-informed shortcut instead of making it infer the ratio from raw band values.",
  "improvement","ndmi")}
${card("Class Weighting — Penalise Grey Soil Errors More",
  "Retrain with class_weight='balanced'. The loss function now weights errors on underrepresented classes proportionally, shifting decision boundaries toward harder moisture categories. Note: k-NN and Naive Bayes do not support this strategy.",
  "improvement","weighted")}
${card("Hierarchical Classification — Decompose the Problem",
  "Train two Random Forest classifiers: Stage 1 decides grey vs. not-grey; Stage 2 decides which moisture level for grey-predicted pixels. Each classifier tackles a simpler problem than the original 6-class task.",
  "improvement","hierarchical")}
${card("Oversampling — Generate Synthetic Minority-Class Samples",
  "Apply SMOTE (Synthetic Minority Oversampling Technique) before training. For each underrepresented class, SMOTE generates new synthetic samples by interpolating between existing examples in feature space, giving the model more balanced exposure to hard-to-classify moisture levels without throwing away majority-class data.",
  "improvement","smote")}
</div>
</div></div>
${resultsHtml}
</div>

<div class="col-right">
<div class="panel"><div class="panel-hdr">What About a 5×5 Neighbourhood?</div>
<div class="panel-body" style="font-size:.87rem;line-height:1.8;color:#4a3060">
A natural hypothesis: <em>would measuring a larger 5×5 neighbourhood help?</em><br><br>
<strong>For grey soil moisture: probably not.</strong> The confusion is <em>spectral</em> — how much shortwave-IR the centre pixel absorbs depends on its own moisture content, not on what surrounds it. More neighbouring pixels of the same damp material don't tell you more about <em>that pixel's</em> moisture level.<br><br>
A wider neighbourhood helps when confusion is <em>spatial/textural</em> — e.g., distinguishing a cotton field with regular row-planting texture from irregular scrub vegetation. For those classes, neighbours carry meaningful context.<br><br>
<strong>Also practical:</strong> the Statlog dataset only provides 3×3 neighbourhoods, so we can't test 5×5 here. But the physics suggests it wouldn't be the right lever for this specific problem.
</div></div>

${state.improvement ? `<div class="panel"><div class="panel-hdr">Reflection Questions</div>
<div class="panel-body" style="font-size:.87rem;line-height:1.9;color:#4a3060">
Try all four strategies. Ask yourself:<br><br>
• Does <em>overall accuracy</em> go up, down, or barely move?<br>
• Does grey soil <em>recall</em> improve even when accuracy drops slightly?<br>
• Which strategy helps <em>all three</em> grey classes, vs. just one or two?<br>
• Is the trade-off worth it — or does gaining grey soil recall cost precision on other classes?<br><br>
There's no universally right answer. The best strategy depends on what errors matter most in the real application. Flood risk mapping cares more about Very Damp Grey Soil recall than about Cotton Crop precision.
</div></div>` : ""}
</div>
</div>`;
}

function step12(){
  const hp     = getPrep().hp[state.model];
  const hpRes  = hp.results[state.hp_idx !== null ? state.hp_idx : hp.best_idx];
  const hpParams = Object.entries(hpRes.params).map(([k,v])=>`${k}=${v==="null"?"None":v}`).join(", ");
  const imp    = DATA.improvements[getKey()];
  const strat  = state.improvement;
  let impRes = null;
  if(strat==="hierarchical") impRes = imp.hierarchical;
  else if(strat==="ndmi")    impRes = imp.ndmi?.[state.model];
  else if(strat==="smote")   impRes = imp.smote?.[state.model];
  else                       impRes = imp.weighted?.[state.model]?.unsupported ? null : imp.weighted?.[state.model];
  const stratNames = {ndmi:"NDMI Feature Engineering",weighted:"Class Weighting",hierarchical:"Hierarchical RF",smote:"SMOTE Oversampling"};
  const scalerNote = {standard:"Normalises to zero mean, unit variance — best default; benefits distance-based and gradient models.",minmax:"Compresses to [0,1] — preserves zero values but sensitive to outliers.",none:"Raw values — fine for tree models, hurts k-NN and Logistic Regression."};
  const stratifyNote = {yes:"Class proportions preserved in both splits — reduces variance in evaluation, almost always the right choice.",no:"Random split — risks one split having more of a rare class than the other."};
  const metricNote = {accuracy:"Treats all classes equally by count — misleading on imbalanced data.",macro_f1:"Weights all classes equally regardless of size — best when every class matters.",weighted_f1:"Weights classes by frequency — majority classes dominate but minorities still count."};
  const cvNote = {3:"Faster; each training set is 66% of the data.",5:"Standard tradeoff — reliable estimates without excessive compute.",10:"Most stable estimates; each training set is 90% of the data."};
  const rows = [
    ["Feature Scaling",    {standard:"StandardScaler",minmax:"MinMaxScaler",none:"None"}[state.scaler],         scalerNote[state.scaler]],
    ["Stratified Split",   state.stratify==="yes"?"Yes":"No",                                                   stratifyNote[state.stratify]],
    ["Evaluation Metric",  {accuracy:"Accuracy",macro_f1:"Macro F1",weighted_f1:"Weighted F1"}[state.metric],   metricNote[state.metric]],
    ["CV Folds",           state.cv_folds+"‑fold",                                                              cvNote[state.cv_folds]],
    ["Model Selected",     MODEL_NAMES[state.model],                                                            `CV score: ${getPrep().cv[String(state.cv_folds)][state.model].mean.toFixed(3)}`],
    ["Hyperparameters",    hpParams,                                                                            `Test accuracy: ${hpRes.score.toFixed(3)}`],
    ["Improvement",        stratNames[strat],                                                                   impRes && !impRes.unsupported ? `Test accuracy: ${impRes.score.toFixed(3)} (baseline ${hpRes.score.toFixed(3)}, ${impRes.score>=hpRes.score?"▲":"▼"} ${Math.abs(impRes.score-hpRes.score).toFixed(3)})` : "Result: not applicable for this model"],
  ];
  const tableRows = rows.map(([label,choice,note])=>`
    <tr>
      <td style="padding:8px 12px;font-weight:700;color:#5a2d82;white-space:nowrap;vertical-align:top">${label}</td>
      <td style="padding:8px 12px;font-weight:600;color:#1e1030;vertical-align:top">${choice}</td>
      <td style="padding:8px 12px;font-size:.82rem;color:#4a3060;vertical-align:top">${note}</td>
    </tr>`).join("");
  return `
<div style="text-align:center;padding:36px 32px 24px;background:#ede8f5;border-radius:12px">
  <div style="font-size:1.5rem;font-weight:700;color:#5a2d82;margin-bottom:10px">🎉 Adventure Complete!</div>
  <div style="font-size:.93rem;color:#4a3060;line-height:1.8;max-width:640px;margin:0 auto">You made every decision a real data scientist makes: cleaning, splitting, scaling, choosing a problem type, cross-validating, selecting a model, tuning hyperparameters, diagnosing errors, and trying targeted improvements.<br><br>Try all four strategies — notice when overall accuracy moves versus when grey soil recall moves. They don't always agree, and that tension is real ML engineering.</div>
</div>

<div class="panel" style="margin-top:24px">
  <div class="panel-hdr">Your Playthrough Summary</div>
  <div class="panel-body" style="padding:0">
    <table style="width:100%;border-collapse:collapse">
      <thead><tr style="background:#f3eefb">
        <th style="padding:8px 12px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:#5a2d82">Decision</th>
        <th style="padding:8px 12px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:#5a2d82">Your Choice</th>
        <th style="padding:8px 12px;text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:#5a2d82">What It Means</th>
      </tr></thead>
      <tbody>${tableRows}</tbody>
    </table>
    <p style="font-size:.82rem;color:#8b6aaa;padding:12px 16px;border-top:1px solid #ede8f5">Different choices lead to different final scores — try resetting and taking a different path. The choices that look minor (stratification, metric) often matter more than the ones that look major (model selection).</p>
    <div style="padding:16px 16px 12px;border-top:1px solid #ede8f5;text-align:center">
      <button onclick="downloadNotebook()" style="background:#5a2d82;color:#fff;border:none;border-radius:8px;padding:10px 24px;font-size:.93rem;cursor:pointer;font-weight:600;letter-spacing:.02em">⬇ Download Notebook</button>
      <p style="font-size:.78rem;color:#8b6aaa;margin:8px 0 0">Downloads a runnable Jupyter notebook (.ipynb) with your choices as Python code and matplotlib charts — open in JupyterLab or VS Code to experiment further.</p>
    </div>
  </div>
</div>

<div class="panel" style="margin-top:24px">
  <div class="panel-hdr">What to Explore Next</div>
  <div class="panel-body" style="padding:16px 20px;display:grid;gap:20px">

    <div>
      <div style="font-weight:700;color:#5a2d82;font-size:.95rem;margin-bottom:6px">🔍 Investigate Other Confused Class Pairs</div>
      <p style="font-size:.86rem;color:#4a3060;line-height:1.75;margin:0">Grey soil got all the attention — but look again at your confusion matrix. <strong>Cotton Crop (2) vs. Soil with Vegetation (5)</strong> is often the next biggest source of errors. Both classes show plant-cover spectral signatures; what separates them is crop-row spatial regularity vs. irregular scrub, which the 3×3 neighbourhood only partially captures. Try: does NDMI help here too, or does it only help the moisture classes? Does class weighting shift errors from grey soil onto cotton/vegetation instead?</p>
    </div>

    <div style="border-top:1px solid #ede8f5;padding-top:18px">
      <div style="font-weight:700;color:#5a2d82;font-size:.95rem;margin-bottom:6px">🔗 Combine the Improvement Strategies</div>
      <p style="font-size:.86rem;color:#4a3060;line-height:1.75;margin:0">The four strategies aren't mutually exclusive — they address different root causes. <strong>NDMI + class_weight='balanced'</strong> is a natural first combination: better features for the spectral confusion, plus a loss function that penalises grey soil errors more. <strong>NDMI + SMOTE</strong> is another — synthetic samples in an augmented feature space. Worth checking: do the gains add up, or does one strategy already capture most of what the other would give you?</p>
    </div>

    <div style="border-top:1px solid #ede8f5;padding-top:18px">
      <div style="font-weight:700;color:#5a2d82;font-size:.95rem;margin-bottom:6px">🧠 Three Deeper Questions This Dataset Raises</div>
      <div style="font-size:.86rem;color:#4a3060;line-height:1.75">
        <p style="margin:0 0 10px"><strong>Ordinal structure.</strong> The three grey classes have a natural moisture order: Grey Soil → Damp Grey Soil → Very Damp Grey Soil. Standard classifiers treat these as unordered categories, so confusing "Grey" with "Very Damp" costs the same as confusing "Grey" with "Damp" — even though the first error is much worse in a flood-risk context. Look into <em>ordinal classification</em> (e.g. <code>mord</code> library) or a custom cost matrix in the loss function.</p>
        <p style="margin:0 0 10px"><strong>Decision threshold tuning.</strong> For models that output probabilities — Logistic Regression and Naive Bayes — you don't have to classify at the default 0.5 threshold. Lowering the threshold for "Very Damp Grey Soil" lets you catch more true positives at the cost of more false positives. Use <code>predict_proba</code> and a precision-recall curve to find the threshold that matches the real-world cost of each error type.</p>
        <p style="margin:0"><strong>Result stability.</strong> Everything you've seen was computed on one 80/20 random split. How much does the final test accuracy vary across different seeds? Try fitting your chosen model 20 times with different <code>random_state</code> values and plotting the distribution — you may find the apparent gains from your improvement strategy are smaller than the run-to-run variance.</p>
      </div>
    </div>

  </div>
</div>`;
}

// ── Restore state from localStorage on page load ─────────────────────────────
try {
  const saved = localStorage.getItem("landsat_state_v2");
  if(saved){ Object.assign(state, JSON.parse(saved)); }
} catch(e){}

// ── Tooltip (position:fixed, escapes overflow containers) ─────────────────────
const _tt = document.createElement("div");
_tt.style.cssText = [
  "position:fixed","background:#2e1a48","color:#e8d8f8",
  "padding:7px 11px","border-radius:7px","font-size:.76rem",
  "white-space:normal","max-width:280px","line-height:1.5",
  "text-align:left","pointer-events:none","z-index:9999",
  "box-shadow:0 3px 10px rgba(0,0,0,.35)","display:none"
].join(";");
document.body.appendChild(_tt);

function _posTip(e){
  const pad=14, tw=_tt.offsetWidth, th=_tt.offsetHeight;
  let x=e.clientX+pad, y=e.clientY-th-pad;
  if(x+tw>window.innerWidth-8) x=e.clientX-tw-pad;
  if(y<8) y=e.clientY+pad;
  _tt.style.left=x+"px"; _tt.style.top=y+"px";
}
document.addEventListener("mouseover", e=>{
  const el=e.target.closest("[data-tip]");
  if(el){ _tt.textContent=el.dataset.tip; _tt.style.display="block"; _posTip(e); }
  else   { _tt.style.display="none"; }
});
document.addEventListener("mousemove", e=>{
  if(_tt.style.display!=="none") _posTip(e);
});

// ── Notebook generator ────────────────────────────────────────────────────────
function _nbL(s){const ls=s.split('\n');return ls.map((l,i)=>i<ls.length-1?l+'\n':l);}
function _md(src){return{cell_type:"markdown",metadata:{},source:_nbL(src)};}
function _code(src){return{cell_type:"code",execution_count:null,metadata:{},outputs:[],source:_nbL(src)};}

function generateNotebook(){
  const hp    = getPrep().hp[state.model];
  const hpIdx = state.hp_idx!==null ? state.hp_idx : hp.best_idx;
  const p     = hp.results[hpIdx].params;
  const baseScore = hp.results[hpIdx].score;
  const fmt   = v=>(v==="null"||v===null||v===undefined)?"None":(typeof v==="string"?"'"+v+"'":String(v));
  const mc = {
    lr:  `LogisticRegression(max_iter=2000, solver='saga', C=${fmt(p.C)}, penalty=${fmt(p.penalty)})`,
    knn: `KNeighborsClassifier(n_neighbors=${fmt(p.n_neighbors)}, metric=${fmt(p.metric)})`,
    dt:  `DecisionTreeClassifier(random_state=42, max_depth=${fmt(p.max_depth)}, min_samples_leaf=${fmt(p.min_samples_leaf)})`,
    nb:  `GaussianNB(var_smoothing=${fmt(p.var_smoothing)})`,
    rf:  `RandomForestClassifier(random_state=42, n_estimators=${fmt(p.n_estimators)}, max_depth=${fmt(p.max_depth)})`,
  }[state.model];
  const mname   = {lr:"Logistic Regression",knn:"k-Nearest Neighbors",dt:"Decision Tree",nb:"Naive Bayes",rf:"Random Forest"}[state.model];
  const sname   = {standard:"StandardScaler",minmax:"MinMaxScaler",none:"None (raw values)"}[state.scaler];
  const mname_metric = {accuracy:"Accuracy",macro_f1:"Macro F1",weighted_f1:"Weighted F1"}[state.metric];
  const scoring = {accuracy:'"accuracy"',macro_f1:'"f1_macro"',weighted_f1:'"f1_weighted"'}[state.metric];
  const stratArg = state.stratify==="yes"?", stratify=y":"";
  const nf = state.cv_folds;
  const strat = state.improvement;
  const stratLabel = {ndmi:"NDMI Feature Engineering",weighted:"Class Weighting",hierarchical:"Hierarchical RF",smote:"SMOTE Oversampling"}[strat];
  const hpStr = Object.entries(p).map(([k,v])=>`${k}=${v==="null"?"None":v}`).join(", ");

  const scalerSetup = state.scaler==="standard" ? `scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)` : state.scaler==="minmax" ? `scaler = MinMaxScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)` : `X_tr = X_train.copy()
X_te = X_test.copy()`;

  const scalerNote = {
    standard:"Normalises features to zero mean / unit variance — the safe default for distance-based and gradient models.",
    minmax:  "Compresses all features to [0, 1] — preserves zero values but can amplify outliers.",
    none:    "No scaling applied — fine for tree models, but hurts k-NN and Logistic Regression."
  }[state.scaler];

  const cells = [];

  // ── Title ─────────────────────────────────────────────────────────────────
  cells.push(_md(
`# Landsat Satellite Classification — Your Adventure Choices

Generated from the **Landsat Adventure** interactive walkthrough.
This notebook reproduces your decisions as runnable Python code with matplotlib visualizations.

| Decision | Your Choice |
|---|---|
| Feature Scaling | ${sname} |
| Stratified Split | ${state.stratify==="yes"?"Yes":"No"} |
| Evaluation Metric | ${mname_metric} |
| CV Folds | ${nf}-fold |
| Model | ${mname} |
| Hyperparameters | ${hpStr} |
| Improvement Strategy | ${stratLabel} |`
  ));

  // ── Install dependencies ──────────────────────────────────────────────────
  cells.push(_md("## Setup — Install Dependencies\n\nRun this cell first. `%pip` installs into the active kernel; safe to re-run."));
  cells.push(_code(
`%pip install -q "numpy>=1.26" "matplotlib>=3.10" "scikit-learn>=1.5"${strat==="smote" ? ' "imbalanced-learn>=0.14"' : ""}`
  ));

  // ── Imports ───────────────────────────────────────────────────────────────
  cells.push(_md("## Step 0 — Imports"));
  cells.push(_code(
`import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              precision_recall_fscore_support, f1_score)

CLASS_LABELS = [1, 2, 3, 4, 5, 7]
CLASS_NAMES  = ["Red Soil", "Cotton Crop", "Grey Soil",
                "Damp Grey Soil", "Soil w/ Vegetation", "Very Damp Grey Soil"]`
  ));

  // ── Load data ─────────────────────────────────────────────────────────────
  cells.push(_md(
`## Step 1 — Load the Data

The Statlog (Landsat Satellite) dataset: 6,430 pixels × 36 spectral features
(4 multispectral bands × 9 neighbourhood pixels), 6 land-cover classes.
First run downloads ~1 MB from OpenML.`
  ));
  cells.push(_code(
`print("Loading dataset …")
ds = fetch_openml(name="satimage", version=1, as_frame=True)
X  = ds.data.values.astype(float)
y  = np.array([int(lbl.replace(".", "")) for lbl in ds.target.values])
print(f"X: {X.shape},  classes: {np.unique(y)}")

counts = [int((y == lbl).sum()) for lbl in CLASS_LABELS]
fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.bar(CLASS_NAMES, counts, color="#8b5cf6", edgecolor="white")
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_title("Class Distribution (n = 6,430 pixels)", fontsize=13, pad=10)
ax.set_ylabel("Count")
ax.set_ylim(0, max(counts) * 1.15)
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.tight_layout()
plt.show()`
  ));

  // ── Split & Scale ─────────────────────────────────────────────────────────
  cells.push(_md(
`## Steps 2–4 — Split & Scale

**Your choices:** ${state.stratify==="yes"?"Stratified":"Random"} 80/20 train/test split, ${sname}.

*${scalerNote}*`
  ));
  cells.push(_code(
`X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42${stratArg})

${scalerSetup}

print(f"Train: {X_train.shape},  Test: {X_test.shape}")`
  ));

  // ── Cross-validate ────────────────────────────────────────────────────────
  cells.push(_md(
`## Step 7 — Cross-Validate All Models

**Your choices:** ${nf}-fold stratified CV, scored by ${mname_metric}.

Each bar = mean CV score ± 1 std. The red dashed line is the dummy "always predict majority class" baseline.`
  ));
  cells.push(_code(
`models = {
    "Logistic Reg.": LogisticRegression(max_iter=2000, solver="saga"),
    "k-NN":          KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes":   GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
}
colors = {
    "Logistic Reg.": "#3b82f6", "k-NN": "#10b981", "Decision Tree": "#f59e0b",
    "Naive Bayes":   "#06b6d4", "Random Forest": "#8b5cf6",
}
cv = StratifiedKFold(n_splits=${nf}, shuffle=True, random_state=42)

cv_results = {}
for name, clf in models.items():
    scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring=${scoring})
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name:18s}  {scores.mean():.4f} ± {scores.std():.4f}")

dummy = max(np.bincount(y_train)[1:]) / len(y_train)
print(f"{'Dummy baseline':18s}  {dummy:.4f}")

names_r = list(cv_results.keys())
means   = [cv_results[n][0] for n in names_r]
stds    = [cv_results[n][1] for n in names_r]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(names_r, means, yerr=stds, capsize=5,
              color=[colors[n] for n in names_r], edgecolor="white")
ax.axhline(dummy, color="#ef4444", linestyle="--", linewidth=1.3,
           label=f"Dummy baseline ({dummy:.3f})")
ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=4, fontsize=9)
ax.set_title("${nf}-fold CV — ${mname_metric} (training set)", fontsize=13)
ax.set_ylabel(${scoring})
ax.set_ylim(0, 1.08)
ax.legend(fontsize=9)
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.show()`
  ));

  // ── Model + HPs ───────────────────────────────────────────────────────────
  cells.push(_md(
`## Steps 8–9 — Fit Your Model with Tuned Hyperparameters

**Your choice:** ${mname} — ${hpStr}

Test-set accuracy from the adventure: **${baseScore.toFixed(3)}**`
  ));
  cells.push(_code(
`clf = ${mc}
clf.fit(X_tr, y_train)
y_pred = clf.predict(X_te)

acc = float((y_pred == y_test).mean())
f1m = f1_score(y_test, y_pred, labels=CLASS_LABELS, average="macro",     zero_division=0)
f1w = f1_score(y_test, y_pred, labels=CLASS_LABELS, average="weighted",  zero_division=0)
print(f"Test accuracy:   {acc:.4f}")
print(f"Macro F1:        {f1m:.4f}")
print(f"Weighted F1:     {f1w:.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=CLASS_LABELS,
    display_labels=[n.replace(" ", "\\n") for n in CLASS_NAMES],
    cmap="Purples", ax=ax, colorbar=False)
ax.set_title("${mname} — Confusion Matrix (test set)", fontsize=11)
plt.tight_layout()
plt.show()`
  ));

  // ── Precision / Recall ────────────────────────────────────────────────────
  cells.push(_md("## Step 10 — Per-Class Precision & Recall"));
  cells.push(_code(
`pr, rc, _, _ = precision_recall_fscore_support(y_test, y_pred, labels=CLASS_LABELS, zero_division=0)

x = np.arange(len(CLASS_LABELS))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, pr, w, label="Precision", color="#8b5cf6", edgecolor="white")
ax.bar(x + w/2, rc, w, label="Recall",    color="#06b6d4", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\\n") for n in CLASS_NAMES], fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_title("Per-Class Precision & Recall — ${mname}", fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()

for i, lbl in enumerate(CLASS_LABELS):
    print(f"  Class {lbl}  {CLASS_NAMES[i]:26s}  prec={pr[i]:.3f}  recall={rc[i]:.3f}")`
  ));

  // ── Learning curve ────────────────────────────────────────────────────────
  cells.push(_md(
`## Learning Curve

Train vs. validation performance as training set size grows.
A large gap → overfitting. Both curves flat and low → underfitting.`
  ));
  cells.push(_code(
`lc_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
sizes, tr_sc, val_sc = learning_curve(
    ${mc}, X_tr, y_train,
    train_sizes=[0.10, 0.25, 0.50, 0.75, 1.00],
    cv=lc_cv, scoring=${scoring}, n_jobs=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sizes, tr_sc.mean(axis=1),  "o-", color="#8b5cf6", label="Train")
ax.fill_between(sizes,
    tr_sc.mean(axis=1) - tr_sc.std(axis=1),
    tr_sc.mean(axis=1) + tr_sc.std(axis=1), alpha=.15, color="#8b5cf6")
ax.plot(sizes, val_sc.mean(axis=1), "o-", color="#06b6d4", label="Validation")
ax.fill_between(sizes,
    val_sc.mean(axis=1) - val_sc.std(axis=1),
    val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=.15, color="#06b6d4")
ax.set_xlabel("Training set size")
ax.set_ylabel(${scoring})
ax.set_ylim(0, 1.05)
ax.set_title("Learning Curve — ${mname}", fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()`
  ));

  // ── Feature importances (RF / DT only) ────────────────────────────────────
  if(state.model==="rf"||state.model==="dt"){
    cells.push(_md(
`## Feature Importances

${mname} provides feature importances (mean decrease in impurity) for free.
Feature layout: B1-Px1, B2-Px1, B3-Px1, B4-Px1, B1-Px2, …
Band 4 (shortwave-infrared, moisture indicator) dominates — confirming the grey soil story.`
    ));
    cells.push(_code(
`importances = clf.feature_importances_
feat_names  = [f"B{(i % 4) + 1}-Px{i // 4 + 1}" for i in range(36)]
idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(36), importances[idx], color="#8b5cf6", edgecolor="white", linewidth=.4)
ax.set_xticks(range(36))
ax.set_xticklabels([feat_names[i] for i in idx], rotation=70, ha="right", fontsize=7)
ax.set_title("${mname} — Feature Importances (all 36)", fontsize=13)
ax.set_ylabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.show()

print("Top 5 features:")
for rank, fi in enumerate(idx[:5]):
    print(f"  {rank+1}. {feat_names[fi]:12s}  importance={importances[fi]:.4f}")`
    ));
  }

  // ── Improvement strategy ──────────────────────────────────────────────────
  cells.push(_md(`## Step 11 — Improvement: ${stratLabel}`));

  const beforeAfterPlot = (title) => `pr_b, rc_b, _, _ = precision_recall_fscore_support(y_test, y_pred,  labels=CLASS_LABELS, zero_division=0)
pr_a, rc_a, _, _ = precision_recall_fscore_support(y_test, y_new,  labels=CLASS_LABELS, zero_division=0)

x = np.arange(len(CLASS_LABELS)); w = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.bar(x - w/2, pr_b, w, label="Baseline", color="#a78bfa", edgecolor="white")
ax1.bar(x + w/2, pr_a, w, label="${title}",  color="#8b5cf6", edgecolor="white")
ax1.set_xticks(x); ax1.set_xticklabels([n.replace(" ", "\\n") for n in CLASS_NAMES], fontsize=8)
ax1.set_ylim(0, 1.1); ax1.set_title("Precision"); ax1.legend(fontsize=9)
ax2.bar(x - w/2, rc_b, w, label="Baseline", color="#67e8f9", edgecolor="white")
ax2.bar(x + w/2, rc_a, w, label="${title}",  color="#06b6d4", edgecolor="white")
ax2.set_xticks(x); ax2.set_xticklabels([n.replace(" ", "\\n") for n in CLASS_NAMES], fontsize=8)
ax2.set_ylim(0, 1.1); ax2.set_title("Recall"); ax2.legend(fontsize=9)
plt.suptitle("${title} — Precision & Recall Before vs After", fontsize=13)
plt.tight_layout(); plt.show()`;

  if(strat==="ndmi"){
    cells.push(_code(
`def add_ndmi(X_raw):
    cols = []
    for px in range(9):
        b3 = X_raw[:, px*4+2].astype(float)
        b4 = X_raw[:, px*4+3].astype(float)
        cols.append(((b3 - b4) / (b3 + b4 + 1e-8)).reshape(-1, 1))
    return np.hstack([X_raw, np.hstack(cols)])

X_tr_aug = add_ndmi(X_train)
X_te_aug = add_ndmi(X_test)
${state.scaler!=="none"?`X_tr_aug = scaler.fit_transform(X_tr_aug)
X_te_aug = scaler.transform(X_te_aug)`:`# no scaler — raw values`}

clf_aug = ${mc}
clf_aug.fit(X_tr_aug, y_train)
y_new = clf_aug.predict(X_te_aug)

print(f"Baseline accuracy:  {acc:.4f}")
print(f"+ NDMI accuracy:    {float((y_new == y_test).mean()):.4f}")

${beforeAfterPlot("+ NDMI")}`
    ));
  } else if(strat==="weighted"){
    const wmc = state.model==="knn"||state.model==="nb"
      ? null
      : state.model==="lr"
        ? `LogisticRegression(max_iter=2000, solver='saga', C=${fmt(p.C)}, penalty=${fmt(p.penalty)}, class_weight='balanced')`
        : state.model==="dt"
          ? `DecisionTreeClassifier(random_state=42, max_depth=${fmt(p.max_depth)}, min_samples_leaf=${fmt(p.min_samples_leaf)}, class_weight='balanced')`
          : `RandomForestClassifier(random_state=42, n_estimators=${fmt(p.n_estimators)}, max_depth=${fmt(p.max_depth)}, class_weight='balanced')`;
    if(!wmc){
      cells.push(_code(
`# ${mname} does not support class_weight='balanced'.
# Try NDMI or SMOTE with this model, or switch to Logistic Regression / Decision Tree / Random Forest.
print("class_weight is not supported by ${mname}.")`
      ));
    } else {
      cells.push(_code(
`# class_weight='balanced' upweights minority classes during training
clf_w = ${wmc}
clf_w.fit(X_tr, y_train)
y_new = clf_w.predict(X_te)

print(f"Baseline accuracy:   {acc:.4f}")
print(f"Weighted accuracy:   {float((y_new == y_test).mean()):.4f}")

${beforeAfterPlot("Weighted")}`
      ));
    }
  } else if(strat==="smote"){
    cells.push(_code(
`from imblearn.over_sampling import SMOTE

print("Class counts before SMOTE:", {lbl: int((y_train == lbl).sum()) for lbl in CLASS_LABELS})
sm = SMOTE(random_state=42, k_neighbors=5)
X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_train)
print("Class counts after SMOTE: ", {lbl: int((y_tr_res == lbl).sum()) for lbl in CLASS_LABELS})
print(f"{len(y_train)} -> {len(y_tr_res)} training samples")

clf_sm = ${mc}
clf_sm.fit(X_tr_res, y_tr_res)
y_new = clf_sm.predict(X_te)

print(f"Baseline accuracy:  {acc:.4f}")
print(f"SMOTE accuracy:     {float((y_new == y_test).mean()):.4f}")

${beforeAfterPlot("SMOTE")}`
    ));
  } else {
    cells.push(_code(
`# Stage 1: binary classifier — grey soil vs everything else
GREY_LABELS = {3, 4, 7}
y_bin_tr = np.array([1 if lbl in GREY_LABELS else 0 for lbl in y_train])
y_bin_te = np.array([1 if lbl in GREY_LABELS else 0 for lbl in y_test])

rf1 = RandomForestClassifier(random_state=42, n_estimators=100)
rf1.fit(X_tr, y_bin_tr)

# Stage 2: among grey pixels, classify moisture level
grey_mask_tr = y_bin_tr == 1
rf2 = RandomForestClassifier(random_state=42, n_estimators=100)
rf2.fit(X_tr[grey_mask_tr], y_train[grey_mask_tr])

# Combine predictions
y_bin_pred = rf1.predict(X_te)
y_new = y_pred.copy()
grey_test_idx = np.where(y_bin_pred == 1)[0]
if len(grey_test_idx):
    y_new[grey_test_idx] = rf2.predict(X_te[grey_test_idx])

print(f"Baseline accuracy:     {acc:.4f}")
print(f"Hierarchical accuracy: {float((y_new == y_test).mean()):.4f}")

${beforeAfterPlot("Hierarchical")}`
    ));
  }

  // ── Suggestions ───────────────────────────────────────────────────────────
  cells.push(_md(
`## Where to Go From Here

### Easy customisations
- Change hyperparameters (C, n_neighbors, max_depth, etc.) to explore the performance surface
- Try a different improvement strategy in the last section
- Switch the scoring metric to see how model rankings shift

### Medium customisations
- Combine improvements: apply NDMI + class_weight='balanced' together
- Experiment with other band-ratio features (B1/B2, B2/B3) as additional engineered features
- Tune the number of SMOTE neighbours (k_neighbors) or the SMOTE sampling strategy

### Advanced customisations
- Calibrate predicted probabilities with CalibratedClassifierCV
- Stack or vote across multiple models (VotingClassifier / StackingClassifier)
- Add a third stage to the hierarchical model to distinguish Cotton Crop from Red Soil`
  ));

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata:{
      kernelspec:{display_name:"Python 3",language:"python",name:"python3"},
      language_info:{name:"python",version:"3.9.0"}
    },
    cells: cells
  };
}

function downloadNotebook(){
  const nb   = generateNotebook();
  const blob = new Blob([JSON.stringify(nb, null, 2)], {type:"application/json"});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "landsat_" + (state.model||"model") + ".ipynb";
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

render();
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

CACHE = "landsat_data.json"
OUT   = "landsat_adventure.html"

def main():
    X = y = None

    def load_data():
        nonlocal X, y
        if X is None:
            print("Loading dataset …")
            data = fetch_openml(name="satimage", version=1, as_frame=True)
            X = data.data.values.astype(float)
            y = np.array([int(lbl.replace(".", "")) for lbl in data.target.values])

    if os.path.exists(CACHE):
        print(f"Loading cached data from {CACHE} …")
        with open(CACHE) as f:
            blob = json.load(f)
        if "improvements" not in blob:
            load_data()
            print("Computing improvement strategies (~10 min) …")
            blob["improvements"] = compute_improvements(X, y)
            with open(CACHE, "w") as f:
                json.dump(blob, f)
            print(f"\nUpdated {CACHE}")
        if "feature_importances" not in blob:
            load_data()
            print("Computing feature importances (~1 min) …")
            blob["feature_importances"] = compute_feature_importances(X, y)
            with open(CACHE, "w") as f:
                json.dump(blob, f)
            print(f"\nUpdated {CACHE}")
        lc = blob.get("learning_curves", {})
        all_pks = [f"{s}_{t}" for s in ["standard","minmax","none"] for t in ["yes","no"]]
        all_mks = ["lr","knn","dt","nb","rf"]
        lc_complete = all(mk in lc.get(pk,{}) for pk in all_pks for mk in all_mks)
        if not lc_complete:
            load_data()
            done = sum(mk in lc.get(pk,{}) for pk in all_pks for mk in all_mks)
            print(f"Computing learning curves ({done}/24 done) …")
            blob["learning_curves"] = compute_learning_curves(X, y, cache_path=CACHE, cache_blob=blob)
            print(f"\nUpdated {CACHE}")
    else:
        load_data()
        print("Computing all combinations (this takes ~15 min) …")
        blob = compute_all(X, y)
        print("\nComputing improvement strategies (~10 min) …")
        blob["improvements"] = compute_improvements(X, y)
        blob["feature_importances"] = compute_feature_importances(X, y)
        blob["learning_curves"] = compute_learning_curves(X, y, cache_path=CACHE, cache_blob=blob)
        with open(CACHE, "w") as f:
            json.dump(blob, f)
        print(f"\nSaved to {CACHE}")

    html = HTML_TEMPLATE.replace("<<<DATA_JSON>>>", json.dumps(blob))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Written to {OUT}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()

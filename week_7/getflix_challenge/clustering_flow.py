"""
Getflix Adventure — Clustering Pipeline

Trains a user-clustering model with the choices you made in the walkthrough
and saves it to flask_app/models/clustering.pkl for the Flask app to serve.

Usage:
    python clustering_flow.py run --fill zero --model_key kmeans_k5

Parameters:
    --fill        zero | user_mean
    --model_key   kmeans_k3 | kmeans_k5 | kmeans_k8 |
                  agg_ward_k3 | agg_ward_k5 | agg_complete_k5 |
                  dbscan_e05_m5 | dbscan_e10_m5 | dbscan_e15_m5

Run history:
    python clustering_flow.py show
"""
import os
import pickle

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, step
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "flask_app", "models")

_GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
_ITEM_COLS = (
    ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    + _GENRE_NAMES
)


class ClusteringFlow(FlowSpec):

    fill = Parameter(
        "fill",
        help="Missing-rating fill strategy: 'zero' or 'user_mean'",
        default="zero",
    )
    model_key = Parameter(
        "model_key",
        help="Algorithm key, e.g. kmeans_k5 or agg_ward_k5",
        default="kmeans_k5",
    )

    @step
    def start(self):
        valid_fills = {"zero", "user_mean"}
        valid_models = {
            "kmeans_k3", "kmeans_k5", "kmeans_k8",
            "agg_ward_k3", "agg_ward_k5", "agg_complete_k5",
            "dbscan_e05_m5", "dbscan_e10_m5", "dbscan_e15_m5",
        }
        assert self.fill in valid_fills, f"--fill must be one of {valid_fills}"
        assert self.model_key in valid_models, f"--model_key must be one of {valid_models}"
        print(f"Pipeline: fill={self.fill}, model={self.model_key}")
        self.next(self.load_data)

    @step
    def load_data(self):
        self.ratings = pd.read_csv(
            os.path.join(DATA_DIR, "u.data"),
            sep="\t", names=["user_id", "movie_id", "rating", "timestamp"],
        )
        items = pd.read_csv(
            os.path.join(DATA_DIR, "u.item"),
            sep="|", names=_ITEM_COLS, encoding="latin-1",
        )
        items["movie_id"] = items["movie_id"].astype(int)
        self.items = items[["movie_id"] + _GENRE_NAMES]
        print(f"Loaded {len(self.ratings):,} ratings")
        self.next(self.build_matrix)

    @step
    def build_matrix(self):
        pivot = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )
        if self.fill == "zero":
            filled = pivot.fillna(0)
        else:
            filled = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)

        scaler = StandardScaler()
        self.scaled = scaler.fit_transform(filled)
        self.scaler = scaler
        self.user_ids = list(filled.index)
        print(f"Matrix: {len(self.user_ids)} users × {filled.shape[1]} movies")
        self.next(self.fit_model)

    @step
    def fit_model(self):
        key = self.model_key
        scaled = self.scaled

        if key.startswith("kmeans"):
            k = int(key.split("_k")[1])
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        elif key.startswith("agg"):
            parts = key.split("_")
            k = int(parts[-1][1:])
            linkage = parts[1]
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        elif key.startswith("dbscan"):
            eps_str = key.split("_e")[1].split("_")[0]
            eps = float(eps_str) / 10
            model = DBSCAN(eps=eps, min_samples=5, n_jobs=1)

        self.labels = model.fit_predict(scaled).tolist()
        self.model = model

        n_clusters = len(set(self.labels) - {-1})
        noise = self.labels.count(-1)
        print(f"Clusters: {n_clusters}  Noise points: {noise}")
        self.next(self.evaluate)

    @step
    def evaluate(self):
        labels = np.array(self.labels)
        unique = [l for l in set(self.labels) if l != -1]
        if len(unique) >= 2:
            mask = labels != -1
            self.silhouette = round(float(
                silhouette_score(self.scaled[mask], labels[mask])
            ), 4)
        else:
            self.silhouette = None

        # Genre lift per cluster
        merged = self.ratings.merge(self.items, on="movie_id")
        global_props = merged[_GENRE_NAMES].sum() / merged[_GENRE_NAMES].sum().sum()
        lift = {}
        for cluster in sorted(unique):
            users = {uid for uid, lbl in zip(self.user_ids, self.labels) if lbl == cluster}
            c_merged = self.ratings[self.ratings["user_id"].isin(users)].merge(
                self.items, on="movie_id"
            )
            c_props = c_merged[_GENRE_NAMES].sum() / c_merged[_GENRE_NAMES].sum().sum()
            top = (c_props / global_props).nlargest(5)
            lift[str(cluster)] = {g: round(float(v), 3) for g, v in top.items()}
        self.genre_lift = lift
        print(f"Silhouette: {self.silhouette}")
        self.next(self.save_model)

    @step
    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "fill_strategy": self.fill,
            "user_ids": self.user_ids,
            "is_pca": False,
        }
        out = os.path.join(MODELS_DIR, "clustering.pkl")
        with open(out, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Saved → {out}")
        print(f"Silhouette: {self.silhouette}")
        print(f"Top genre lift: {self.genre_lift}")
        self.next(self.end)

    @step
    def end(self):
        print("Done. Run  python flask_app/app.py  to serve the model.")


if __name__ == "__main__":
    ClusteringFlow()

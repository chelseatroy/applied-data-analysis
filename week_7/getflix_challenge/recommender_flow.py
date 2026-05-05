"""
Getflix Adventure — Recommender Pipeline

Trains a collaborative-filtering model with the choices you made in the
walkthrough and saves it to flask_app/models/recommender.pkl.

Usage:
    python recommender_flow.py run --model_key svd_f50

Parameters:
    --model_key   svd_f20 | svd_f50 | svd_f100 |
                  knn_basic_k20 | knn_basic_k40 | knn_basic_k80 |
                  knn_means_k20 | knn_means_k40 | knn_means_k80

Run history:
    python recommender_flow.py show
"""
import os
import pickle

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, step
from surprise import KNNBasic, KNNWithMeans, SVD, Dataset, Reader
from surprise.model_selection import cross_validate

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "flask_app", "models")

_ITEM_COLS = (
    ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    + [f"genre_{i}" for i in range(19)]
)
_SIM_OPTS = {"name": "cosine", "user_based": True}

VALID_KEYS = {
    "svd_f20", "svd_f50", "svd_f100",
    "knn_basic_k20", "knn_basic_k40", "knn_basic_k80",
    "knn_means_k20", "knn_means_k40", "knn_means_k80",
}


def algo_from_key(key):
    if key.startswith("svd"):
        n = int(key.split("f")[1])
        return SVD(n_factors=n, random_state=42)
    k = int(key.split("k")[-1])
    if key.startswith("knn_basic"):
        return KNNBasic(k=k, sim_options=_SIM_OPTS, verbose=False)
    return KNNWithMeans(k=k, sim_options=_SIM_OPTS, verbose=False)


class RecommenderFlow(FlowSpec):

    model_key = Parameter(
        "model_key",
        help="Algorithm key, e.g. svd_f50 or knn_means_k40",
        default="svd_f50",
    )

    @step
    def start(self):
        assert self.model_key in VALID_KEYS, \
            f"--model_key must be one of {sorted(VALID_KEYS)}"
        print(f"Pipeline: model={self.model_key}")
        self.next(self.load_data)

    @step
    def load_data(self):
        path = os.path.join(DATA_DIR, "u.data")
        self.ratings_df = pd.read_csv(
            path, sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(
            self.ratings_df[["user_id", "movie_id", "rating"]], reader
        )
        movies = pd.read_csv(
            os.path.join(DATA_DIR, "u.item"),
            sep="|", names=_ITEM_COLS, encoding="latin-1",
        )
        movies["movie_id"] = movies["movie_id"].astype(int)
        self.movies_df = movies[["movie_id", "title"]]
        print(f"Loaded {len(self.ratings_df):,} ratings")
        self.next(self.cross_validate)

    @step
    def cross_validate(self):
        algo = algo_from_key(self.model_key)
        cv = cross_validate(algo, self.data, measures=["rmse", "mae"], cv=5, verbose=False)
        self.rmse = round(float(np.mean(cv["test_rmse"])), 4)
        self.mae  = round(float(np.mean(cv["test_mae"])),  4)
        print(f"5-fold CV  RMSE={self.rmse}  MAE={self.mae}")
        self.next(self.fit_full)

    @step
    def fit_full(self):
        trainset = self.data.build_full_trainset()
        self.algo = algo_from_key(self.model_key)
        self.algo.fit(trainset)
        print("Fitted on full dataset")
        self.next(self.save_model)

    @step
    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        out = os.path.join(MODELS_DIR, "recommender.pkl")
        with open(out, "wb") as f:
            pickle.dump({"algo": self.algo}, f)
        print(f"Saved → {out}")
        print(f"RMSE: {self.rmse}  MAE: {self.mae}")
        self.next(self.end)

    @step
    def end(self):
        print("Done. Run  python flask_app/app.py  to serve the model.")


if __name__ == "__main__":
    RecommenderFlow()

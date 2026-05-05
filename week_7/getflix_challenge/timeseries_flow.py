"""
Getflix Adventure — Time Series Pipeline

Fits an ARIMA model for all five genres with the choices you made in the
walkthrough and saves forecasts to flask_app/models/timeseries.pkl.

Uses Metaflow's foreach to fit each genre in a separate parallel step —
the same fan-out / fan-in pattern used in production ML pipelines.

Usage:
    python timeseries_flow.py run --transform raw --p 1 --d 1 --q 1

Parameters:
    --transform   raw | log
    --p           AR order (0–3)
    --d           differencing order (0–2)
    --q           MA order (0–3)

Run history:
    python timeseries_flow.py show
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, step
from statsmodels.tsa.arima.model import ARIMA

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "flask_app", "models")
FORECAST_WEEKS = 4

GENRES = ["Drama", "Comedy", "Action", "Thriller", "Romance"]

_GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
_ITEM_COLS = (
    ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    + _GENRE_NAMES
)


class TimeseriesFlow(FlowSpec):

    transform = Parameter(
        "transform",
        help="Preprocessing: 'raw' or 'log'",
        default="raw",
    )
    p = Parameter("p", help="AR order (0–3)",           default=1, type=int)
    d = Parameter("d", help="Differencing order (0–2)", default=1, type=int)
    q = Parameter("q", help="MA order (0–3)",           default=1, type=int)

    @step
    def start(self):
        assert self.transform in {"raw", "log"}, "--transform must be 'raw' or 'log'"
        assert 0 <= self.p <= 3, "--p must be 0–3"
        assert 0 <= self.d <= 2, "--d must be 0–2"
        assert 0 <= self.q <= 3, "--q must be 0–3"
        print(f"Pipeline: transform={self.transform}  ARIMA({self.p},{self.d},{self.q})")
        self.next(self.load_data)

    @step
    def load_data(self):
        ratings = pd.read_csv(
            os.path.join(DATA_DIR, "u.data"),
            sep="\t", names=["user_id", "movie_id", "rating", "timestamp"],
        )
        ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
        items = pd.read_csv(
            os.path.join(DATA_DIR, "u.item"),
            sep="|", names=_ITEM_COLS, encoding="latin-1",
        )
        items["movie_id"] = items["movie_id"].astype(int)
        merged = ratings.merge(items[["movie_id"] + _GENRE_NAMES], on="movie_id")

        series = {}
        for genre in GENRES:
            g = merged[merged[genre] == 1].copy()
            weekly = g.resample("W", on="date")["rating"].mean().dropna()
            counts  = g.resample("W", on="date")["rating"].count()
            s = weekly[counts >= 10]
            if self.transform == "log":
                s = np.log(s)
            series[genre] = s

        self.series_by_genre = series
        self.genres_to_fit = GENRES
        print(f"Loaded weekly series for {len(GENRES)} genres")
        self.next(self.fit_genre, foreach="genres_to_fit")

    # -------------------------------------------------------------------
    # Fan-out: one branch per genre, all run in parallel
    # -------------------------------------------------------------------

    @step
    def fit_genre(self):
        genre = self.input
        series = self.series_by_genre[genre]
        order = (self.p, self.d, self.q)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ARIMA(series, order=order).fit(
                method_kwargs={"warn_convergence": False}
            )

        fc = result.get_forecast(steps=FORECAST_WEEKS)
        fc_ci = fc.conf_int(alpha=0.2)
        last = series.index[-1]
        fc_dates = pd.date_range(
            start=last + pd.Timedelta(weeks=1), periods=FORECAST_WEEKS, freq="W"
        )

        self.genre = genre
        self.aic   = round(float(result.aic), 2)
        self.forecast_entry = {
            "forecast":         fc.predicted_mean.round(4).tolist(),
            "conf_int_lower":   fc_ci.iloc[:, 0].round(4).tolist(),
            "conf_int_upper":   fc_ci.iloc[:, 1].round(4).tolist(),
            "periods":          [d.strftime("%Y-%m-%d") for d in fc_dates],
        }
        print(f"  {genre:12s}  AIC={self.aic}")
        self.next(self.join)

    # -------------------------------------------------------------------
    # Fan-in: collect all genre results
    # -------------------------------------------------------------------

    @step
    def join(self, inputs):
        self.forecasts = {inp.genre: inp.forecast_entry for inp in inputs}
        self.aic_by_genre = {inp.genre: inp.aic for inp in inputs}
        self.next(self.save_model)

    @step
    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        out = os.path.join(MODELS_DIR, "timeseries.pkl")
        with open(out, "wb") as f:
            pickle.dump({"forecasts": self.forecasts}, f)
        print(f"Saved → {out}")
        for genre, aic in sorted(self.aic_by_genre.items()):
            print(f"  {genre:12s}  AIC={aic}")
        self.next(self.end)

    @step
    def end(self):
        print("Done. Run  python flask_app/app.py  to serve the model.")


if __name__ == "__main__":
    TimeseriesFlow()

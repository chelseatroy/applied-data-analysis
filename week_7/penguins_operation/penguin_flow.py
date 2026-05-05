"""
Penguin species classifier — Metaflow training pipeline.

Usage:
    python penguin_flow.py run --max_depth 2
    python penguin_flow.py run --max_depth 5
    python penguin_flow.py show
"""
from metaflow import FlowSpec, step, Parameter


class PenguinFlow(FlowSpec):

    max_depth = Parameter(
        "max_depth",
        help="Maximum depth of the decision tree (try 2, 3, 5)",
        default=3,
        type=int,
    )
    test_size = Parameter(
        "test_size",
        help="Fraction of data held out for evaluation (0.0–1.0)",
        default=0.2,
        type=float,
    )

    # Encoding maps shared between the flow and the Flask app.
    # The Flask app imports these directly so training and inference
    # always use the same transformation.
    ISLAND_MAP = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
    SEX_MAP    = {"female": 0, "male": 1}
    SPECIES    = ["Adelie", "Chinstrap", "Gentoo"]
    FEATURES   = [
        "bill_length_mm", "bill_depth_mm",
        "flipper_length_mm", "body_mass_g",
        "island_enc", "sex_enc",
    ]

    @step
    def start(self):
        assert 1 <= self.max_depth <= 20, "--max_depth must be between 1 and 20"
        assert 0 < self.test_size < 1,    "--test_size must be between 0 and 1"
        print(f"Training decision tree  max_depth={self.max_depth}  test_size={self.test_size}")
        self.next(self.load_data)

    @step
    def load_data(self):
        import seaborn as sns
        self.df = sns.load_dataset("penguins")
        print(f"Loaded {len(self.df)} rows  ({self.df['species'].value_counts().to_dict()})")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        df = self.df.dropna().copy()
        n_dropped = len(self.df) - len(df)
        print(f"Dropped {n_dropped} rows with missing values  ({len(df)} remain)")

        df["island_enc"] = df["island"].map(self.ISLAND_MAP)
        df["sex_enc"]    = df["sex"].map(self.SEX_MAP)
        df["species_enc"] = df["species"].map(
            {s: i for i, s in enumerate(self.SPECIES)}
        )

        from sklearn.model_selection import train_test_split
        X = df[self.FEATURES].values
        y = df["species_enc"].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=42,
            stratify=y,
        )
        self.n_dropped = n_dropped
        print(f"Train: {len(self.X_train)} rows   Test: {len(self.X_test)} rows")
        self.next(self.train)

    @step
    def train(self):
        from sklearn.tree import DecisionTreeClassifier, export_text
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42,
        )
        self.model.fit(self.X_train, self.y_train)

        # Print the tree so students can read the rules
        rules = export_text(self.model, feature_names=self.FEATURES)
        print("\n── Decision tree rules ──────────────────────────────")
        print(rules)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        from sklearn.metrics import accuracy_score, classification_report
        preds = self.model.predict(self.X_test)
        self.accuracy = float(accuracy_score(self.y_test, preds))
        report = classification_report(
            self.y_test, preds, target_names=self.SPECIES
        )
        print(f"\n── Evaluation (max_depth={self.max_depth}) ─────────────")
        print(f"Test accuracy: {self.accuracy:.3f}")
        print(report)
        self.next(self.save_model)

    @step
    def save_model(self):
        import os, pickle
        from datetime import datetime, timezone

        out_dir = os.path.join(os.path.dirname(__file__), "flask_app", "models")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "penguin_model.pkl")

        payload = {
            "model":        self.model,
            "feature_names": self.FEATURES,
            "species":      self.SPECIES,
            "island_map":   self.ISLAND_MAP,
            "sex_map":      self.SEX_MAP,
            "max_depth":    self.max_depth,
            "accuracy":     self.accuracy,
            "n_train":      len(self.X_train),
            "n_test":       len(self.X_test),
            "n_dropped":    self.n_dropped,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

        print(f"\nSaved → {out_path}")
        print(f"Accuracy: {self.accuracy:.3f}   max_depth: {self.max_depth}")
        self.next(self.end)

    @step
    def end(self):
        print("\nDone.")
        print("Start the server:   python flask_app/app.py")
        print("Or reload a running server:  GET http://localhost:5051/reload")


if __name__ == "__main__":
    PenguinFlow()

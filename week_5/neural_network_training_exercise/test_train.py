import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from train import standardize, build_model, train_with_early_stopping, compare_regularization


# Shared data fixture so we don't reload the dataset for every test
@pytest.fixture(scope="module")
def breast_cancer_split():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def standardized_split(breast_cancer_split):
    X_train, X_test, y_train, y_test = breast_cancer_split
    X_tr_std, X_te_std = standardize(X_train, X_test)
    return X_tr_std, X_te_std, y_train, y_test


# =============================================================================
# Task 1: standardize()
# =============================================================================

class TestStandardize:

    def test_train_mean_near_zero(self, breast_cancer_split):
        X_train, X_test, _, _ = breast_cancer_split
        X_tr, _ = standardize(X_train, X_test)
        assert np.allclose(X_tr.mean(axis=0), 0, atol=1e-10), (
            "Each feature in X_train should have mean ≈ 0 after standardization. "
            "Make sure you are fitting StandardScaler on X_train."
        )

    def test_train_std_near_one(self, breast_cancer_split):
        X_train, X_test, _, _ = breast_cancer_split
        X_tr, _ = standardize(X_train, X_test)
        assert np.allclose(X_tr.std(axis=0), 1, atol=1e-10), (
            "Each feature in X_train should have std ≈ 1 after standardization."
        )

    def test_scaler_not_fitted_on_test(self, breast_cancer_split):
        X_train, X_test, _, _ = breast_cancer_split
        _, X_te = standardize(X_train, X_test)
        # If the scaler was (incorrectly) fit on test data, X_test mean would be ~0.
        # Fit on train only: test mean will differ from 0 because the distributions differ.
        test_means_near_zero = np.allclose(X_te.mean(axis=0), 0, atol=0.05)
        assert not test_means_near_zero, (
            "The scaler appears to have been fit on X_test (its column means are ~0). "
            "Fit the scaler on X_train only, then use transform() on both arrays."
        )

    def test_shapes_preserved(self, breast_cancer_split):
        X_train, X_test, _, _ = breast_cancer_split
        X_tr, X_te = standardize(X_train, X_test)
        assert X_tr.shape == X_train.shape, "X_train shape should not change after standardization."
        assert X_te.shape == X_test.shape, "X_test shape should not change after standardization."

    def test_returns_tuple_of_two_arrays(self, breast_cancer_split):
        X_train, X_test, _, _ = breast_cancer_split
        result = standardize(X_train, X_test)
        assert len(result) == 2, "standardize() should return exactly two arrays: (X_train, X_test)."


# =============================================================================
# Task 2: train_with_early_stopping()
# =============================================================================

class TestEarlyStopping:

    def test_returns_integer(self, standardized_split):
        X_tr, _, y_tr, _ = standardized_split
        model = build_model()
        result = train_with_early_stopping(model, X_tr, y_tr, max_epochs=50, patience=5)
        assert isinstance(result, int), (
            "train_with_early_stopping() should return an int: the number of epochs trained."
        )

    def test_model_is_fitted_after_call(self, standardized_split):
        X_tr, _, y_tr, _ = standardized_split
        model = build_model()
        train_with_early_stopping(model, X_tr, y_tr, max_epochs=50, patience=5)
        assert hasattr(model, "coefs_"), (
            "The model should be fitted (have coefs_) after train_with_early_stopping() returns."
        )

    def test_runs_at_least_patience_epochs(self, standardized_split):
        X_tr, _, y_tr, _ = standardized_split
        patience = 15
        model = build_model()
        epochs = train_with_early_stopping(model, X_tr, y_tr, max_epochs=500, patience=patience)
        assert epochs >= patience, (
            f"The model should train for at least `patience` ({patience}) epochs before stopping. "
            f"Got {epochs}."
        )

    def test_does_not_exceed_max_epochs(self, standardized_split):
        X_tr, _, y_tr, _ = standardized_split
        max_epochs = 30
        model = build_model()
        epochs = train_with_early_stopping(model, X_tr, y_tr, max_epochs=max_epochs, patience=5)
        assert epochs <= max_epochs, (
            f"Training should never exceed max_epochs ({max_epochs}). Got {epochs}."
        )

    def test_stops_early_when_no_improvement_possible(self):
        # With random labels the model cannot improve validation accuracy,
        # so early stopping should trigger well before max_epochs.
        rng = np.random.RandomState(99)
        X_random = rng.randn(120, 20)
        y_random = rng.randint(0, 2, 120)
        model = build_model()
        epochs = train_with_early_stopping(
            model, X_random, y_random, max_epochs=500, patience=10
        )
        assert epochs < 200, (
            f"With random labels the model cannot improve, so early stopping should trigger "
            f"long before 500 epochs. Got {epochs} epochs — check that your patience counter "
            f"resets only when validation accuracy strictly improves."
        )


# =============================================================================
# Task 3: compare_regularization()
# =============================================================================

class TestRegularization:

    ALPHAS = [0, 0.001, 0.01, 0.1]

    def test_returns_correct_number_of_results(self):
        results = compare_regularization(self.ALPHAS)
        assert len(results) == len(self.ALPHAS), (
            f"compare_regularization() should return one (train_acc, test_acc) tuple "
            f"per alpha. Expected {len(self.ALPHAS)}, got {len(results)}."
        )

    def test_accuracies_are_valid_probabilities(self):
        results = compare_regularization(self.ALPHAS)
        for alpha, (train_acc, test_acc) in zip(self.ALPHAS, results):
            assert 0.0 <= train_acc <= 1.0, (
                f"Train accuracy for alpha={alpha} should be in [0, 1]. Got {train_acc}."
            )
            assert 0.0 <= test_acc <= 1.0, (
                f"Test accuracy for alpha={alpha} should be in [0, 1]. Got {test_acc}."
            )

    def test_high_regularization_reduces_overfit_gap(self):
        # With alpha=0, train accuracy should exceed test accuracy (overfitting gap).
        # With strong regularization (alpha=1), that gap should shrink.
        results = compare_regularization([0, 1.0])
        gap_none = results[0][0] - results[0][1]
        gap_high = results[1][0] - results[1][1]
        assert gap_high <= gap_none + 0.03, (
            f"Strong regularization (alpha=1.0) should reduce the train/test accuracy gap "
            f"compared to no regularization (alpha=0). "
            f"Gap with alpha=0: {gap_none:.4f}, gap with alpha=1.0: {gap_high:.4f}."
        )

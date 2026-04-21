import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------
# Task 1: Standardize inputs
#
# Neural network weights are sensitive to feature scale. A feature
# ranging 0–1000 will receive very different weight magnitudes than
# one ranging 0–1, making training unstable.
#
# Implement this function so that it:
#   - Fits a StandardScaler on X_train only
#   - Transforms both X_train and X_test using that fitted scaler
#   - Returns the transformed (X_train, X_test) as a tuple
#
# Common mistake to avoid: do NOT fit the scaler on X_test.
# Fitting on test data leaks information about the test distribution
# into your preprocessing, which inflates your test accuracy estimate.
# -----------------------------------------------------------------
def standardize(X_train, X_test):
    raise NotImplementedError


def build_model(alpha=0.0):
    return MLPClassifier(
        hidden_layer_sizes=(20, 10),
        activation='relu',
        solver='sgd',
        learning_rate_init=0.01,
        alpha=alpha,
        warm_start=True,
        max_iter=1,
        random_state=42,
        tol=0,
        n_iter_no_change=10000,
    )


# -----------------------------------------------------------------
# Task 2: Early stopping
#
# The model currently trains for a fixed number of epochs. That
# means it either stops too early (underfitting) or too late
# (overfitting) depending on an arbitrary choice of max_iter.
#
# A better approach: monitor validation accuracy after each epoch
# and stop when it has not improved for `patience` consecutive epochs.
#
# Implement this function so that it:
#   - Splits X_train/y_train into 80% sub-train and 20% validation
#   - Trains the model one epoch at a time by calling model.fit()
#     (warm_start=True means each call continues from the last)
#   - After each epoch, evaluates accuracy on the validation set
#   - Stops if validation accuracy has not improved for `patience`
#     consecutive epochs
#   - Returns the total number of epochs trained as an integer
#
# Hint: keep track of the best validation accuracy seen so far and
# a counter of how many epochs have passed without improvement.
# -----------------------------------------------------------------
def train_with_early_stopping(model, X_train, y_train, max_epochs=500, patience=20):
    raise NotImplementedError


# -----------------------------------------------------------------
# Task 3: L2 weight decay (regularization)
#
# Without regularization (alpha=0), the model can overfit: training
# accuracy climbs while test accuracy plateaus or falls. Adding an
# L2 penalty on the weights — controlled by the `alpha` parameter —
# discourages the model from fitting noise in the training data.
#
# Implement this function so that it:
#   - Loads and standardizes the data (reuse load_data and standardize)
#   - For each value in `alphas`, trains a model using
#     train_with_early_stopping and records (train_accuracy, test_accuracy)
#   - Prints a table showing alpha, train accuracy, and test accuracy
#     for each run
#   - Returns a list of (train_accuracy, test_accuracy) tuples,
#     one per alpha value, in the same order as `alphas`
#
# After running this, look at the results: which alpha gives the
# smallest gap between train and test accuracy? Is there a point
# where too much regularization hurts test accuracy?
# -----------------------------------------------------------------
def compare_regularization(alphas):
    raise NotImplementedError


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Uncomment after completing Task 1:
    # X_train, X_test = standardize(X_train, X_test)

    model = build_model()

    # Replace with train_with_early_stopping() after completing Task 2:
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # Uncomment after completing Task 3:
    # print("\nRegularization comparison:")
    # compare_regularization([0, 0.0001, 0.001, 0.01, 0.1, 1.0])


if __name__ == "__main__":
    main()

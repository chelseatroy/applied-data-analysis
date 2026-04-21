# Neural Network Training Exercise

This is a working but incomplete training script (`train.py`) that builds a neural
network to classify breast cancer tumors as malignant or benign. Your job is to improve it
in three concrete ways that are covered in the reading.

## Setup

```bash
pip install scikit-learn numpy pytest
```

## Running the script

```bash
python train.py
```

## Running the tests

```bash
pytest test_train.py -v
```

All tests will fail until you complete the corresponding task. Work through the tasks in
order — each one builds on the last.

---

## Task 1 — Standardize inputs (`standardize`)

Implement the `standardize(X_train, X_test)` function in `train.py`.

Neural network weights are sensitive to feature scale. The breast cancer dataset has
features ranging from 0–1 (e.g. smoothness) and others in the hundreds (e.g. area).
Without standardization, the network assigns wildly different weight magnitudes to
compensate for scale, which destabilizes training.

Your function should:
1. Fit a `StandardScaler` on `X_train`
2. Transform both `X_train` and `X_test` using that fitted scaler
3. Return `(X_train_scaled, X_test_scaled)`

**Do not fit the scaler on `X_test`.** The test set represents unseen data; fitting the
scaler on it leaks information about the test distribution into preprocessing and inflates
your accuracy estimate.

Once implemented, uncomment the `standardize` call in `main()`.

---

## Task 2 — Early stopping (`train_with_early_stopping`)

Implement the `train_with_early_stopping(model, X_train, y_train, max_epochs, patience)`
function in `train.py`.

The current script trains for a fixed number of epochs, which means it either stops too
early (underfitting) or too late (overfitting) based on an arbitrary choice. Early stopping
lets the data decide when to stop: we monitor validation accuracy and halt when it has not
improved for `patience` consecutive epochs.

Your function should:
1. Split `X_train`/`y_train` into 80% sub-train and 20% validation
2. Train the model one epoch at a time by calling `model.fit()` (with `warm_start=True`
   already set in `build_model()`, each call continues from where the last left off)
3. After each epoch, evaluate accuracy on the validation set
4. Stop if validation accuracy has not improved for `patience` consecutive epochs
5. Return the total number of epochs trained as an `int`

Once implemented, replace the `model.fit()` call in `main()` with
`train_with_early_stopping(model, X_train, y_train)`.

---

## Task 3 — Regularization (`compare_regularization`)

Implement the `compare_regularization(alphas)` function in `train.py`.

Without regularization (`alpha=0`), neural networks tend to overfit: training accuracy
climbs while test accuracy plateaus or falls. L2 weight decay adds a penalty proportional
to the squared magnitude of the weights, discouraging the network from fitting noise.
The `alpha` parameter in `build_model()` controls its strength.

Your function should:
1. Load and standardize the data using `load_data()` and `standardize()`
2. For each value in `alphas`, train a model using `train_with_early_stopping()` and
   record `(train_accuracy, test_accuracy)`
3. Print a table showing `alpha`, train accuracy, and test accuracy for each run
4. Return a list of `(train_accuracy, test_accuracy)` tuples in the same order as `alphas`

After running this, answer in a comment at the bottom of `train.py`:
- Which alpha value gives the best test accuracy?
- Is there a point where too much regularization hurts performance? Why?

Once implemented, uncomment the `compare_regularization` call in `main()`.

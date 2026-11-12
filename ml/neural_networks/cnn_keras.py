# cnn_keras.py
# The same 1D CNN as cnn_torch.py, built with TensorFlow/Keras.
# Same data loading, preprocessing, cross-validation and output as ml/model.py
# (see its header for why the original version failed and how interpolation is used),
# with logistic regression swapped for a CNN so the two can be compared.
#
# Keras vs PyTorch: in cnn_torch.py we wrote the training loop ourselves
# (zero_grad -> loss -> backward -> step). Keras hides that loop behind
# model.compile() and model.fit().
#
# Run from the project root:  python ml/neural_networks/cnn_keras.py

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's startup info messages

import time
import numpy as np
import pandas as pd
from tensorflow import keras
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Settings ----------------
GRID_MIN, GRID_MAX, GRID_STEP = 750.0, 3990.0, 8.0   # cm-1, inside the image's 717-4003 range
SG_WINDOW, SG_POLY = 11, 2                            # Savitzky-Golay smoothing/derivative
MIN_MATCH_CORR = 0.70                                 # below this, pixel = "no match" (background/noise)

EPOCHS = 15            # passes over the training data
BATCH_SIZE = 128       # spectra per training step
LEARNING_RATE = 1e-3

keras.utils.set_random_seed(0)   # same starting weights every run -> repeatable results

grid = np.arange(GRID_MIN, GRID_MAX, GRID_STEP)


def preprocess(spectra, wavenumbers):
    """Resample spectra (n, bands) onto `grid`, take 1st derivative, then SNV-normalize each row."""
    spectra = np.asarray(spectra, dtype=float)
    order = np.argsort(wavenumbers)                   # np.interp needs increasing x
    wavenumbers, spectra = wavenumbers[order], spectra[:, order]
    resampled = np.array([np.interp(grid, wavenumbers, s) for s in spectra])
    deriv = savgol_filter(resampled, SG_WINDOW, SG_POLY, deriv=1, axis=1)
    mean = deriv.mean(axis=1, keepdims=True)
    std = deriv.std(axis=1, keepdims=True) + 1e-12
    return (deriv - mean) / std


# ---------------- The CNN ----------------
def build_model(n_points):
    """Same architecture as SpectrumCNN in cnn_torch.py. Output = P(plastic)."""
    model = keras.Sequential([
        keras.Input(shape=(n_points, 1)),                                    # Keras puts channels last: (405, 1)
        keras.layers.Conv1D(16, 7, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(2),                                        # 405 -> 202
        keras.layers.Conv1D(32, 7, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(2),                                        # 202 -> 101
        keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(2),                                        # 101 -> 50
        keras.layers.Flatten(),                                              # keeps position information
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),                                           # reduce overfitting
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train(X, y):
    """Train a fresh CNN on spectra X (n, 405) with labels y (n,)."""
    model = build_model(X.shape[1])
    # Balance the classes, like class_weight='balanced' in logistic regression
    n = len(y)
    class_weight = {0: n / (2 * (y == 0).sum()), 1: n / (2 * (y == 1).sum())}
    model.fit(X[..., np.newaxis], y,                 # add channel dim -> (n, 405, 1)
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weight, verbose=2)  # one line per epoch
    return model


def predict_proba(model, X):
    """P(plastic) for each spectrum in X."""
    return model.predict(X[..., np.newaxis], batch_size=4096, verbose=0).ravel()


# ---------------- 1-2. Load library, keep FTIR, crop to shared range ----------------
t0 = time.time()
header = pd.read_csv("data/raw.csv", nrows=0).columns
spectral_columns = [c for c in header if c.replace('.', '').isdigit()]
lib_wavenumbers = np.array([float(c) for c in spectral_columns])

meta_columns = ['spectrum_type', 'plastic_or_not', 'material_class', 'sample_name']
library = pd.read_csv("data/raw.csv", usecols=meta_columns + spectral_columns, low_memory=False)
library = library[library['spectrum_type'] == 'ftir'].reset_index(drop=True)

in_range = (lib_wavenumbers >= GRID_MIN - 10) & (lib_wavenumbers <= GRID_MAX + 10)
cols_in_range = [c for c, keep in zip(spectral_columns, in_range) if keep]
lib_X = library[cols_in_range].apply(pd.to_numeric, errors='coerce').to_numpy()
complete = np.isfinite(lib_X).all(axis=1)
lib_X = lib_X[complete]
library = library[complete].reset_index(drop=True)
print("FTIR training spectra: %d" % len(library))

y = (library['plastic_or_not'] == 'plastic').astype(int).to_numpy()   # 1 = plastic, 0 = not plastic
groups = library['sample_name'].fillna('unknown').astype(str).to_numpy()

# ---------------- 3-4. Interpolate + preprocess ----------------
lib_P = preprocess(lib_X, lib_wavenumbers[in_range])

# ---------------- 5. Cross-validation (same grouped folds as model.py) ----------------
print("Cross-validating (5 folds, grouped by sample)...")
cv_pred = np.empty_like(y)
for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(lib_P, y, groups), start=1):
    print("Fold %d" % fold)
    model = train(lib_P[train_idx], y[train_idx])
    cv_pred[test_idx] = (predict_proba(model, lib_P[test_idx]) >= 0.5).astype(int)
print(confusion_matrix(y, cv_pred))
print(classification_report(y, cv_pred, target_names=['not plastic', 'plastic']))

print("Training final model on all data...")
model = train(lib_P, y)

# ---------------- 6. Predict every pixel ----------------
cube = np.load("output/hyperspectral_data.npy")      # (rows, cols, bands)
img_wavenumbers = np.load("output/wavelengths.npy").astype(float)
rows, cols, bands = cube.shape
pixels = cube.reshape(-1, bands)
pix_P = preprocess(pixels, img_wavenumbers)

prob_plastic = predict_proba(model, pix_P)

# Best library match by Pearson correlation. After SNV, correlation = dot product / n_points.
n_points = len(grid)
best_corr = np.empty(len(pix_P))
best_idx = np.empty(len(pix_P), dtype=int)
for start in range(0, len(pix_P), 2000):
    corr = pix_P[start:start + 2000] @ lib_P.T / n_points
    best_idx[start:start + 2000] = corr.argmax(axis=1)
    best_corr[start:start + 2000] = corr.max(axis=1)

# ---------------- 7. Final call: "no match" for background ----------------
has_signal = best_corr >= MIN_MATCH_CORR
prediction = np.where(~has_signal, 'no match',
                      np.where(prob_plastic >= 0.5, 'plastic', 'not plastic'))

# ---------------- 8. Save results ----------------
row_idx, col_idx = np.divmod(np.arange(len(pixels)), cols)
results = pd.DataFrame({
    'row': row_idx,
    'column': col_idx,
    'prob_plastic': prob_plastic.round(4),
    'best_match_corr': best_corr.round(4),
    'best_match_class': library['material_class'].to_numpy()[best_idx],
    'best_match_label': library['plastic_or_not'].to_numpy()[best_idx],
    'prediction': prediction,
})
results.to_csv('output/pixel_predictions_keras.csv', index=False)

print("Pixels: %d" % len(results))
print(results['prediction'].value_counts())
print("\nSaved output/pixel_predictions_keras.csv (%.1f s total)" % (time.time() - t0))
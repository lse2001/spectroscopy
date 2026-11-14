# model.py
# Classifies pixels of an FTIR hyperspectral image as plastic / not plastic,
# using the Open Specy reference library (raw.csv) as training data.
#
# ---------------------------------------------------------------------------
# WHY THE OLD VERSION FAILED (predicted "not plastic" for every pixel)
# ---------------------------------------------------------------------------
# The model (logistic regression) was fine; the data going into it wasn't.
#   - Mixed instruments: raw.csv holds FTIR, Raman and NIR spectra. We trained
#     on all of them, but our image is FTIR, and Raman values are on a totally
#     different scale (counts in the 1000s vs absorbance ~0-1).
#   - Fake data from extrapolation: raw.csv spans 102-11,994 cm-1, but each
#     spectrum only fills its own instrument's range (rest = NaN). We filled
#     NaNs with column means and np.interp'd pixels onto the full range.
#     Our machine only measures 717-4,003, so outside that np.interp just
#     repeats the edge value: ~80% of every pixel was flat, made-up numbers.
#   - No preprocessing: baseline/intensity differences swamped peak shape.
#   - Background pixels (no particle, just noise) were forced into a class.
#   - Accuracy was checked on the training data itself, which proves nothing.
#
# ---------------------------------------------------------------------------
# INTERPOLATION, IN PLAIN ENGLISH
# ---------------------------------------------------------------------------
# The library measures every 6 cm-1 (1002, 1008...); our machine every ~7.71
# (1002.85, 1010.56...). To compare spectra point by point, both need values
# at the same spots. Interpolation "connects the dots" to estimate values in
# between: if it's 0.40 at 1002 and 0.46 at 1008, it's ~0.43 at 1005.
# We put both library and pixels on one shared ruler: 750-3,990 every 8 cm-1.
# Key rule: interpolation only works BETWEEN measured points, never past the
# edges. So the ruler stays inside the range both sources actually measured.
#   Spacing mismatch -> interpolate.   Range mismatch -> crop.
#
# ---------------------------------------------------------------------------
# PIPELINE (in execution order)
# ---------------------------------------------------------------------------
#   1. Load raw.csv, keep only FTIR spectra
#   2. Crop to 750-3,990 cm-1; drop library spectra with gaps there (no filling)
#   3. Interpolate library and pixels onto the shared 8 cm-1 ruler
#   4. Preprocess both the same way: Savitzky-Golay 1st derivative (slope,
#      removes baseline) + SNV (scale to mean 0 / std 1, so only shape matters)
#   5. Train; test with 5-fold cross-validation grouped by sample, so the
#      model is always scored on samples it never saw
#   6. Predict every pixel in hyperspectral_data.npy: plastic probability +
#      closest library match and its correlation score
#   7. Pixels with match < MIN_MATCH_CORR -> "no match" (background/noise)
#   8. Save pixel_predictions.csv

import time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Settings ----------------
GRID_MIN, GRID_MAX, GRID_STEP = 750.0, 3990.0, 8.0   # cm-1, inside the image's 717-4003 range
SG_WINDOW, SG_POLY = 11, 2                            # Savitzky-Golay smoothing/derivative
MIN_MATCH_CORR = 0.70                                 # below this, pixel = "no match" (background/noise)
C_REG = 0.1                                           # logistic regression regularization (smaller = stronger)

grid = np.arange(GRID_MIN, GRID_MAX, GRID_STEP)


def preprocess(spectra, wavenumbers):
    """Steps 3-4: resample spectra (n, bands) onto `grid`, take 1st derivative, then SNV-normalize each row."""
    spectra = np.asarray(spectra, dtype=float)
    order = np.argsort(wavenumbers)                   # np.interp needs increasing x
    wavenumbers, spectra = wavenumbers[order], spectra[:, order]
    resampled = np.array([np.interp(grid, wavenumbers, s) for s in spectra])
    deriv = savgol_filter(resampled, SG_WINDOW, SG_POLY, deriv=1, axis=1)
    mean = deriv.mean(axis=1, keepdims=True)
    std = deriv.std(axis=1, keepdims=True) + 1e-12
    return (deriv - mean) / std


# ---------------- 1. Load library, keep FTIR only ----------------
t0 = time.time()
header = pd.read_csv("data/raw.csv", nrows=0).columns
spectral_columns = [c for c in header if c.replace('.', '').isdigit()]
lib_wavenumbers = np.array([float(c) for c in spectral_columns])

meta_columns = ['spectrum_type', 'plastic_or_not', 'material_class', 'sample_name']
library = pd.read_csv("data/raw.csv", usecols=meta_columns + spectral_columns, low_memory=False)
print("Loaded library: %d spectra (%.1f s)" % (len(library), time.time() - t0))
print(library['spectrum_type'].value_counts(), "\n")

library = library[library['spectrum_type'] == 'ftir'].reset_index(drop=True)

# ---------------- 2. Crop to shared range, drop spectra with gaps ----------------
in_range = (lib_wavenumbers >= GRID_MIN - 10) & (lib_wavenumbers <= GRID_MAX + 10)
cols_in_range = [c for c, keep in zip(spectral_columns, in_range) if keep]
lib_X = library[cols_in_range].apply(pd.to_numeric, errors='coerce').to_numpy()
complete = np.isfinite(lib_X).all(axis=1)
lib_X = lib_X[complete]
library = library[complete].reset_index(drop=True)
print("FTIR spectra fully covering %.0f-%.0f cm-1: %d" % (GRID_MIN, GRID_MAX, len(library)))

y = (library['plastic_or_not'] == 'plastic').astype(int).to_numpy()   # 1 = plastic, 0 = not plastic
print("Label counts: plastic=%d, not plastic=%d\n" % (y.sum(), len(y) - y.sum()))

# ---------------- 3-4. Interpolate + preprocess library ----------------
lib_P = preprocess(lib_X, lib_wavenumbers[in_range])

# ---------------- 5. Train + honest evaluation ----------------
model = LogisticRegression(max_iter=3000, C=C_REG, class_weight='balanced')

# Group by sample so near-duplicate spectra of one sample can't be in both train and test
groups = library['sample_name'].fillna('unknown').astype(str).to_numpy()
print("Cross-validating (5 folds, grouped by sample)...")
cv_pred = cross_val_predict(model, lib_P, y, cv=GroupKFold(n_splits=5), groups=groups)
print(confusion_matrix(y, cv_pred))
print(classification_report(y, cv_pred, target_names=['not plastic', 'plastic']))

model.fit(lib_P, y)

# ---------------- 6. Predict every pixel ----------------
cube = np.load("output/hyperspectral_data.npy")            # (rows, cols, bands)
img_wavenumbers = np.load("output/wavelengths.npy").astype(float)
rows, cols, bands = cube.shape
pixels = cube.reshape(-1, bands)
pix_P = preprocess(pixels, img_wavenumbers)

prob_plastic = model.predict_proba(pix_P)[:, 1]

# Best library match by Pearson correlation. After SNV, correlation = dot product / n_points.
n_points = len(grid)
best_corr = np.empty(len(pix_P))
best_idx = np.empty(len(pix_P), dtype=int)
for start in range(0, len(pix_P), 2000):            # chunks to keep memory reasonable
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
results.to_csv('output/pixel_predictions.csv', index=False)

print("Pixels: %d" % len(results))
print(results['prediction'].value_counts())
print("\nMost common best-match classes among matched pixels:")
print(results.loc[has_signal, 'best_match_class'].value_counts().head(10))
print("\nSaved pixel_predictions.csv (%.1f s total)" % (time.time() - t0))
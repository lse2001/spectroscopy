# cnn_torch.py
# 1D convolutional neural network (CNN) classifier in PyTorch.
# Same data loading, preprocessing, cross-validation and output as ml/model.py
# (see its header for why the original version failed and how interpolation is used),
# with logistic regression swapped for a CNN so the two can be compared.
#
# Logistic regression vs CNN: logistic regression learns one weight per wavenumber
# and adds them up. A CNN slides small windows along the spectrum and learns to
# recognise local shapes (peaks, shoulders), then decides plastic / not plastic
# from which shapes appear where.
#
# Run from the project root:  python ml/neural_networks/cnn_torch.py

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

torch.manual_seed(0)   # same starting weights every run -> repeatable results
np.random.seed(0)

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
class SpectrumCNN(nn.Module):
    """Input: (batch, 1, 405) spectra. Output: (batch,) raw scores; sigmoid(score) = P(plastic)."""

    def __init__(self, n_points):
        super().__init__()
        # Each block: convolution (learn local shape detectors) -> ReLU -> max pool (halve the length)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),    # 405 -> 202
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),   # 202 -> 101
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),   # 101 -> 50
        )
        # Flatten keeps position information: a peak at 2915 cm-1 means something different than one at 1700
        n_flat = 64 * (n_points // 2 // 2 // 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 64), nn.ReLU(),
            nn.Dropout(0.3),                  # randomly drop 30% of units while training to reduce overfitting
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def train(X, y):
    """Train a fresh CNN on spectra X (n, 405) with labels y (n,)."""
    model = SpectrumCNN(X.shape[1])
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1),   # add channel dim -> (n, 1, 405)
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # pos_weight balances the classes, like class_weight='balanced' in logistic regression
    pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()            # clear gradients from the previous step
            loss = loss_fn(model(xb), yb)    # how wrong are we on this batch?
            loss.backward()                  # compute gradients
            optimizer.step()                 # nudge weights to be less wrong
            total_loss += loss.item() * len(xb)
        print("  epoch %2d/%d  loss %.4f" % (epoch + 1, EPOCHS, total_loss / len(dataset)))
    return model


def predict_proba(model, X):
    """P(plastic) for each spectrum in X."""
    model.eval()                             # turns off dropout
    probs = []
    with torch.no_grad():                    # no gradients needed for prediction
        for start in range(0, len(X), 4096):
            xb = torch.tensor(X[start:start + 4096], dtype=torch.float32).unsqueeze(1)
            probs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(probs)


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
results.to_csv('output/pixel_predictions_torch.csv', index=False)

print("Pixels: %d" % len(results))
print(results['prediction'].value_counts())
print("\nSaved output/pixel_predictions_torch.csv (%.1f s total)" % (time.time() - t0))
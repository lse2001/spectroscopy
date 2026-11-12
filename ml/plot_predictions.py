# plot_predictions.py
# Visual check of model.py's output. Shows three maps side by side:
#   1. The image: mean absorbance across all 427 wavenumbers ("where is there stuff")
#   2. The prediction for each pixel: plastic / not plastic / no match
#   3. The best library match score per pixel, with the MIN_MATCH_CORR cutoff marked
# Plus a histogram of match scores, to help pick MIN_MATCH_CORR.
# Run model.py first (it creates pixel_predictions.csv).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

MIN_MATCH_CORR = 0.70   # keep in sync with model.py

cube = np.load("hyperspectral_data.npy")               # (rows, cols, bands)
rows, cols, bands = cube.shape
results = pd.read_csv("pixel_predictions.csv")

# Panel 1: average all wavenumber slices into one picture
mean_image = cube.mean(axis=2)

# Panel 2: turn text labels into numbers so they can be drawn as colours
label_codes = {'no match': 0, 'not plastic': 1, 'plastic': 2}
pred_map = results['prediction'].map(label_codes).to_numpy().reshape(rows, cols)
pred_cmap = ListedColormap(['#e8e8e8', '#4c78a8', '#e45756'])   # grey, blue, red

# Panel 3: match score per pixel
corr_map = results['best_match_corr'].to_numpy().reshape(rows, cols)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

low, high = np.percentile(mean_image, [1, 99])     # ignore the extreme 1% at each end
im0 = axes[0].imshow(mean_image, cmap='gray', vmin=low, vmax=high)
axes[0].set_title("Image (mean absorbance, all wavenumbers)")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

axes[1].imshow(pred_map, cmap=pred_cmap, vmin=0, vmax=2, interpolation='nearest')
axes[1].set_title("Prediction")
axes[1].legend(handles=[Patch(color='#e45756', label='plastic'),
                        Patch(color='#4c78a8', label='not plastic'),
                        Patch(color='#e8e8e8', label='no match')],
               loc='lower right', fontsize=8)

im2 = axes[2].imshow(corr_map, cmap='viridis', vmin=0, vmax=1)
axes[2].contour(corr_map >= MIN_MATCH_CORR, levels=[0.5], colors='white', linewidths=0.6)
axes[2].set_title("Best library match (white line = %.2f cutoff)" % MIN_MATCH_CORR)
fig.colorbar(im2, ax=axes[2], fraction=0.046)

for ax in axes:
    ax.set_xlabel("column")
    ax.set_ylabel("row")
plt.tight_layout()

# Histogram: how are match scores distributed? Real particles should form a separate bump.
plt.figure(figsize=(7, 4))
plt.hist(results['best_match_corr'], bins=100, color='#4c78a8')
plt.axvline(MIN_MATCH_CORR, color='#e45756', linestyle='--', label='cutoff %.2f' % MIN_MATCH_CORR)
plt.xlabel("Best library match (correlation)")
plt.ylabel("Number of pixels")
plt.title("Distribution of match scores")
plt.legend()
plt.tight_layout()

plt.show()
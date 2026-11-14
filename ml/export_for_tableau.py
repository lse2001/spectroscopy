# export_for_tableau.py
# Turns the models' prediction CSVs into tidy tables for a Tableau dashboard.
# Python does the processing; Tableau does the reporting.
#
# Writes to output/tableau/:
#   pixels.csv         - one row per pixel per model (for the particle map)
#   particles.csv      - one row per detected plastic particle per model (for the particle inventory)
#   model_summary.csv  - one row per model: pixel counts, shares, agreement (for model comparison)
#   library.csv        - one row per training spectrum: type, label, class (for the training-data overview)
#
# Run a model first (ml/model.py and/or the CNNs), then from the project root:
#   python ml/export_for_tableau.py

import os
import numpy as np
import pandas as pd
from scipy import ndimage

# Prediction files written by each model; any that don't exist yet are skipped
MODEL_FILES = {
    "Logistic regression": "output/pixel_predictions.csv",
    "PyTorch CNN": "output/pixel_predictions_torch.csv",
    "Keras CNN": "output/pixel_predictions_keras.csv",
}
LIBRARY_PATH = "data/raw.csv"
OUT_DIR = "output/tableau"
PIXEL_SIZE_UM = 25.0   # from the .hdr: pixel size = 0.000025 m = 25 micrometres

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- 1. Pixels: stack every model's predictions into one long table ----------------
frames = []
for model_name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print("Skipping %s (no %s yet)" % (model_name, path))
        continue
    df = pd.read_csv(path)
    df.insert(0, "model", model_name)
    frames.append(df)

if not frames:
    raise SystemExit("No prediction files found in output/. Run ml/model.py first.")

pixels = pd.concat(frames, ignore_index=True)
pixels["pixel_id"] = pixels["row"].astype(str) + "_" + pixels["column"].astype(str)
pixels["is_plastic"] = (pixels["prediction"] == "plastic").astype(int)
pixels.to_csv(os.path.join(OUT_DIR, "pixels.csv"), index=False)
print("pixels.csv: %d rows (%d models)" % (len(pixels), pixels["model"].nunique()))


# ---------------- 2. Particles: group neighbouring plastic pixels into particles ----------------
particle_rows = []
for model_name, df in pixels.groupby("model"):
    rows, cols = df["row"].max() + 1, df["column"].max() + 1
    df = df.sort_values(["row", "column"])

    plastic_mask = (df["prediction"] == "plastic").to_numpy().reshape(rows, cols)
    # Label connected regions: touching pixels (including diagonals) belong to the same particle
    labels, n_particles = ndimage.label(plastic_mask, structure=np.ones((3, 3)))

    flat_labels = labels.ravel()
    for particle_id in range(1, n_particles + 1):
        members = df.iloc[np.flatnonzero(flat_labels == particle_id)]
        n_pixels = len(members)
        particle_rows.append({
            "model": model_name,
            "particle_id": particle_id,
            "n_pixels": n_pixels,
            "area_um2": n_pixels * PIXEL_SIZE_UM ** 2,
            "equiv_diameter_um": round(2 * np.sqrt(n_pixels / np.pi) * PIXEL_SIZE_UM, 1),
            "centroid_row": round(members["row"].mean(), 1),
            "centroid_column": round(members["column"].mean(), 1),
            "material_class": members["best_match_class"].mode().iloc[0],   # most common match in the particle
            "mean_prob_plastic": round(members["prob_plastic"].mean(), 4),
            "mean_match_corr": round(members["best_match_corr"].mean(), 4),
            "is_single_pixel": int(n_pixels == 1),                          # likely noise
        })

particles = pd.DataFrame(particle_rows)
particles.to_csv(os.path.join(OUT_DIR, "particles.csv"), index=False)
print("particles.csv: %d rows" % len(particles))


# ---------------- 3. Model summary: one row per model ----------------
summary_rows = []
for model_name, df in pixels.groupby("model"):
    matched = df[df["prediction"] != "no match"]
    model_particles = particles[particles["model"] == model_name]
    summary_rows.append({
        "model": model_name,
        "total_pixels": len(df),
        "plastic_pixels": int((df["prediction"] == "plastic").sum()),
        "not_plastic_pixels": int((df["prediction"] == "not plastic").sum()),
        "no_match_pixels": int((df["prediction"] == "no match").sum()),
        "plastic_share_of_matched": round((matched["prediction"] == "plastic").mean(), 4),
        # How often the model's call matches the label of each pixel's closest library spectrum
        "agrees_with_closest_match": round(((matched["prediction"] == "plastic") ==
                                            (matched["best_match_label"] == "plastic")).mean(), 4),
        "particles": len(model_particles),
        "single_pixel_particles": int(model_particles["is_single_pixel"].sum()) if len(model_particles) else 0,
    })

model_summary = pd.DataFrame(summary_rows)
model_summary.to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)
print("model_summary.csv: %d rows" % len(model_summary))


# ---------------- 4. Library: training-data overview (metadata only, no spectra) ----------------
library = pd.read_csv(LIBRARY_PATH, usecols=["spectrum_type", "plastic_or_not", "material_class"],
                      low_memory=False)
library["used_for_training"] = (library["spectrum_type"] == "ftir").astype(int)
library.to_csv(os.path.join(OUT_DIR, "library.csv"), index=False)
print("library.csv: %d rows" % len(library))

print("\nSaved Tableau tables to %s/" % OUT_DIR)
print(model_summary.to_string(index=False))
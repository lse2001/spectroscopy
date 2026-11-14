# Spectroscopy-Based Materials Classification (FTIR)

**Interactive dashboard:** [FTIR Microplastic Classification: ML Model Comparison](https://public.tableau.com/app/profile/lucas.s4097/viz/FTIRMicroplasticClassificationMLModelComparison/Dashboard1) (Tableau Public)

Pixel-level classification of microplastics in FTIR hyperspectral images, developed with the Moore Institute for Plastic Pollution Research (MIPPR). Each pixel is classified as plastic, not plastic, or background using models trained on the Open Specy reference library. A logistic regression baseline is compared against 1D CNNs in PyTorch and TensorFlow/Keras, and the results are published to an interactive Tableau dashboard.

## Project Structure

```
spectroscopy/
├── data/          # raw inputs (not tracked): ENVI .hdr/.dat files, raw.csv (Open Specy library)
├── output/        # generated files (not tracked): .npy arrays, prediction CSVs, Tableau tables
├── imaging/       # loading and exploring the hyperspectral image
└── ml/            # classification models, plots, Tableau export
    └── neural_networks/
```

## Setup and Running

```
python3 -m venv venv314
source venv314/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib spectral torch tensorflow
```

Run from the project root, in this order:

```
python imaging/main.py
python ml/model.py
python ml/neural_networks/cnn_torch.py
python ml/neural_networks/cnn_keras.py
python ml/plot_predictions.py
python ml/export_for_tableau.py
```

When using new data samples, the `.hdr` file must contain `byte order = 0` and `data file = dat_file_name.dat`.

## Scripts

**`imaging/`**
- `main.py`: loads the ENVI image and saves the image and wavelengths as `.npy` files in `output/`. Set `hdr_file` and `dat_file` to choose the sample.
- `sample.py`: displays the image at 2915 cm-1, a C-H stretch where common plastics absorb strongly.
- `histogram.py`: plots the distribution of mean absorbance across pixels.
- `functions.py`: shared spectrum-analysis utilities.
- `selection.py`: selects high-absorbance pixels (legacy; no longer used by the models).

**`ml/`**
- `model.py`: logistic regression. Keeps only FTIR spectra from `data/raw.csv`, crops library and image spectra to their shared wavenumber range, interpolates both onto one grid, and preprocesses both identically (Savitzky-Golay first derivative + SNV) so the model compares peak shape rather than intensity. Evaluated with 5-fold cross-validation grouped by sample. Pixels that don't closely match any library spectrum are labeled "no match."
- `neural_networks/cnn_torch.py`: same pipeline with a 1D CNN in PyTorch, which learns local spectral shapes such as peaks.
- `neural_networks/cnn_keras.py`: the same CNN in TensorFlow/Keras.
- `plot_predictions.py`: shows the image, prediction map, match-score map, and match-score histogram.
- `export_for_tableau.py`: combines all three models' predictions into tables for the dashboard (`output/tableau/`), including plastic particles grouped from touching pixels.

## Findings

Results for the `dat2` sample (39,592 pixels), trained on 15,333 FTIR library spectra. All models share the same "no match" step: 35,061 pixels (88.6%) are background. Plastic predictions mostly match polyolefins (polyethylene/polypropylene).

| Metric | Logistic Regression | PyTorch CNN | Keras CNN |
|---|---|---|---|
| Library accuracy (cross-validated) | ~94% | ~99% | ~99% |
| Plastic pixels | 3,607 (9.1%) | 2,924 (7.4%) | 2,515 (6.4%) |
| Not plastic pixels | 924 (2.3%) | 1,607 (4.1%) | 2,016 (5.1%) |
| Plastic particles detected | 527 | 572 | 752 |
| Single-pixel particles (likely noise) | 351 | 393 | 519 |
| Agrees with closest library match | 81% | 71% | 59% |

The CNNs score higher on clean library spectra but appear less reliable on the noisier image: they label more particles "not plastic," produce more single-pixel noise, and agree less with the closest library match. This suggests overfitting to the library and domain shift on real sensor data. The image has no ground-truth labels, so agreement with the closest library match is a proxy, not true accuracy. CNN results vary slightly between runs.

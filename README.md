# Spectroscopy-Based Materials Classification (FTIR)

**Interactive dashboard:** [FTIR Microplastic Classification: ML Model Comparison](https://public.tableau.com/app/profile/lucas.s4097/viz/FTIRMicroplasticClassificationMLModelComparison/Dashboard1) (Tableau Public). Explore the per-pixel predictions and switch between the logistic regression, PyTorch CNN and Keras CNN models.

This project classifies materials in Fourier Transform Infrared (FTIR) hyperspectral images at the pixel level, with a focus on detecting microplastics. Developed as part of ongoing work with the Moore Institute for Plastic Pollution Research (MIPPR), it preprocesses absorbance data from ENVI-format files, classifies each pixel as plastic, not plastic, or background using models trained on the Open Specy reference library, compares a logistic regression baseline against 1D convolutional neural networks in PyTorch and TensorFlow/Keras, and publishes the results to an interactive Tableau dashboard.

## Project Structure

```
spectroscopy/
├── data/                      # raw inputs (not tracked): ENVI .hdr/.dat files, raw.csv (Open Specy library)
├── output/                    # generated files (not tracked): .npy arrays, prediction CSVs, Tableau tables
├── imaging/                   # loading and exploring the hyperspectral image
│   ├── main.py
│   ├── sample.py
│   ├── histogram.py
│   ├── functions.py
│   └── selection.py
└── ml/                        # classification models and outputs
    ├── model.py
    ├── plot_predictions.py
    ├── export_for_tableau.py
    └── neural_networks/
        ├── cnn_torch.py
        └── cnn_keras.py
```

## Setup and Running

The project runs on Python 3 in a virtual environment (`venv314`):

```
python3 -m venv venv314
source venv314/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib spectral torch tensorflow
```

All scripts are run from the project root, in this order:

```
python imaging/main.py                      # ENVI image -> output/*.npy
python ml/model.py                          # logistic regression
python ml/neural_networks/cnn_torch.py      # PyTorch CNN
python ml/neural_networks/cnn_keras.py      # Keras CNN
python ml/plot_predictions.py               # visual check
python ml/export_for_tableau.py             # tables for the Tableau dashboard
```

## Scripts

The script `imaging/main.py` loads the ENVI-format hyperspectral image and extracts wavelength metadata, then saves the image and metadata to NumPy `.npy` files in `output/` for use by other scripts. You may edit the `multiplier` parameter when filtered_means is called, as well as the vars `hdr_file` and `dat_file` to your desired specifications.
Need to manually specify the following in the hdr files when using new data samples:
byte order = 0
data file = dat_file_name.dat

The script `imaging/sample.py` displays a single wavenumber slice of the hyperspectral image at 2915 cm-1, a C-H stretch where common plastics absorb strongly. The colour scale is set from the 1st-99th percentile of pixel values so extreme outliers don't wash out the image.

The script `imaging/histogram.py` creates visualizations of the average absorbance across all pixels. It helps assess the general signal quality of the full image before filtering.

The script `imaging/functions.py` contains reusable utilities for spectrum analysis, including mean absorbance calculations, pixel filtering logic, and band indexing functions.

The script `imaging/selection.py` applies statistical filtering to identify high-quality pixels based on mean absorbance across wavelengths and stores them with their spectra in a NumPy `.npy` file. It is no longer used by the models, which now classify every pixel and use a library-match score to filter out background.
To change the bounds of acceptable good pixels you may edit:
lower_threshold = overall_mean_absorbance + 2 * std_dev_absorbance
upper_threshold = overall_mean_absorbance + 3 * std_dev_absorbance

The script `ml/model.py` loads labeled training data from `data/raw.csv` (the Open Specy reference library) and keeps only FTIR spectra, since the library also contains Raman and NIR data. It crops the training data to the wavenumber range both the library and the instrument measured, then interpolates the library and image spectra onto one shared wavenumber grid. Both are preprocessed the same way (Savitzky-Golay first derivative and SNV normalization) so the model compares peak shape rather than intensity. A logistic regression model is trained using scikit-learn and evaluated with 5-fold cross-validation grouped by sample. Every pixel is then classified as plastic or non-plastic, and pixels that don't closely match any library spectrum are labeled "no match" as background. Results are exported to `output/pixel_predictions.csv`.

The script `ml/neural_networks/cnn_torch.py` uses the same data loading, preprocessing, cross-validation and "no match" step as `model.py`, but replaces logistic regression with a 1D convolutional neural network (CNN) built in PyTorch. Instead of learning one weight per wavenumber, the CNN slides small windows along the spectrum to learn local shapes such as peaks, then classifies each pixel from which shapes appear where. Results are exported to `output/pixel_predictions_torch.csv`.

The script `ml/neural_networks/cnn_keras.py` builds the same 1D CNN in TensorFlow/Keras, using Keras's built-in training loop (`compile` and `fit`) instead of the hand-written PyTorch loop. Results are exported to `output/pixel_predictions_keras.csv`.

The script `ml/plot_predictions.py` visualizes the model output. It shows the mean absorbance image, the per-pixel prediction map, and the library match score map side by side, along with a histogram of match scores to help choose the `MIN_MATCH_CORR` cutoff.

The script `ml/export_for_tableau.py` combines the prediction files from all three models into tidy tables for the Tableau dashboard, saved to `output/tableau/`. It writes one row per pixel per model (`pixels.csv`), groups neighbouring plastic pixels into particles with their size, material class and confidence (`particles.csv`), summarizes each model's results (`model_summary.csv`), and exports the training library's metadata (`library.csv`).

## Tableau Dashboard

The [Tableau Public dashboard](https://public.tableau.com/app/profile/lucas.s4097/viz/FTIRMicroplasticClassificationMLModelComparison/Dashboard1) is built from the tables in `output/tableau/`. It shows an interactive particle map of the sample, where each pixel is coloured as plastic, not plastic, or no match, with hover details for the plastic probability, library match score and closest material. A model selector switches the map between logistic regression and the two CNNs, so the differences between models can be compared directly on the image.

## Findings

Results are for the `dat2` sample (39,592 pixels), trained on 15,333 FTIR library spectra. All three models share the same "no match" step, so they agree that 35,061 pixels (88.6%) are background, leaving 4,531 pixels (11.4%) to classify. Plastic predictions mostly match polyolefins (polyethylene/polypropylene) and line up with particle-shaped regions in the image.

| Metric | Logistic Regression | PyTorch CNN | Keras CNN |
|---|---|---|---|
| Library accuracy (cross-validated) | ~94% | ~99% | ~99% |
| Plastic pixels | 3,607 (9.1%) | 2,924 (7.4%) | 2,515 (6.4%) |
| Not plastic pixels | 924 (2.3%) | 1,607 (4.1%) | 2,016 (5.1%) |
| Plastic particles detected | 527 | 572 | 752 |
| Single-pixel particles (likely noise) | 351 | 393 | 519 |
| Agrees with closest library match | 81% | 71% | 59% |

Neural network results vary slightly between runs and machines, so CNN numbers may differ a little each time the models are retrained.

The CNNs score higher on the library but appear less reliable on the image: they classify more particles as "not plastic," produce more isolated single-pixel detections, and agree less often with each pixel's closest library match. We suspect this is because cross-validation tests the models on held-out library spectra, which are clean lab measurements similar to their training data, while image pixels are noisier and contain detector-stripe artifacts. The CNNs are flexible enough to learn fine details of the library spectra that help on the library but don't survive the noise in real image pixels (overfitting and domain shift). Logistic regression is a simpler model that relies on broader features, so it holds up better on the image. For now, logistic regression appears to be the most reliable model on real data.

The image has no ground-truth labels, so agreement with the closest library match is only a proxy for accuracy.

## Next Steps

- Hand-label a sample of image pixels to measure real-world accuracy for each model
- Train the CNNs with added noise and small wavenumber shifts (data augmentation) so library spectra look more like image pixels
- Handle the vertical detector-stripe artifacts visible in the raw image
- Remove isolated single-pixel detections or tune the `MIN_MATCH_CORR` cutoff
- Add particle inventory (counts by polymer type and size) and training-library views to the dashboard

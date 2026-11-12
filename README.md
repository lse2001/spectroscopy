# Spectroscopy-Based Materials Classification (FTIR)

This project focuses on classifying materials based on Fourier Transform Infrared (FTIR) spectroscopy data using hyperspectral image analysis. Developed as part of ongoing work with the Moore Institute for Plastic Pollution Research (MIPPR), the goal is to preprocess, visualize, and classify materials at the pixel level using absorbance data from ENVI-format files. The project runs on Python 3 using a virtual environment (`venv314`).

The script `main.py` loads the ENVI-format hyperspectral image and extracts wavelength metadata. It also saves the hyperspectral data and metadata to NumPy `.npy` files for use by other scripts.
In `main.py` you may edit the `multiplier` parameter when filtered_means is called, as well as the vars `hdr_file` and `dat_file` to your desired specifications.
Need to manually specify the following in the hdr files when using new data samples:
byte order = 0
data file = dat_file_name.dat

The script `sample.py` displays a single wavenumber slice of the hyperspectral image at 2915 cm-1, a C-H stretch where common plastics absorb strongly, using `matplotlib.pyplot`. The colour scale is set from the 1st-99th percentile of pixel values so extreme outliers don't wash out the image.

The script `histogram.py` creates visualizations of the average absorbance across all pixels. It helps assess the general signal quality and verify absorbance characteristics of the full image before filtering.

The script `functions.py` contains reusable utilities for spectrum analysis, including mean absorbance calculations, pixel filtering logic, and band indexing functions used in other parts of the project.

The script `selection.py` applies statistical filtering to identify high-quality pixels based on mean absorbance across wavelengths. These selected "good pixels" are stored with their corresponding absorbance spectra in a NumPy `.npy` file. This script is no longer used by `model.py`, which now classifies every pixel and uses a library-match score to filter out background.
To change the bounds of acceptable good pixels you may edit:
lower_threshold = overall_mean_absorbance + 2 * std_dev_absorbance
upper_threshold = overall_mean_absorbance + 3 * std_dev_absorbance

The script `model.py` loads labeled training data from `raw.csv` (the Open Specy reference library) and keeps only FTIR spectra, since the library also contains Raman and NIR data. It crops the training data to the wavenumber range both the library and the instrument measured, then interpolates the library and image spectra onto one shared wavenumber grid. Both are preprocessed the same way (Savitzky-Golay first derivative and SNV normalization) so the model compares peak shape rather than intensity. A logistic regression model is trained using scikit-learn and evaluated with cross-validation grouped by sample. Every pixel is then classified as plastic or non-plastic, and pixels that don't closely match any library spectrum are labeled "no match" as background. Results are exported to `pixel_predictions.csv`.

The script `neural_networks/cnn_torch.py` uses the same data loading, preprocessing, cross-validation and "no match" step as `model.py`, but replaces logistic regression with a 1D convolutional neural network (CNN) built in PyTorch. Instead of learning one weight per wavenumber, the CNN slides small windows along the spectrum to learn local shapes such as peaks, then classifies each pixel from which shapes appear where. Results are exported to `pixel_predictions_torch.csv`.

The script `neural_networks/cnn_keras.py` builds the same 1D CNN in TensorFlow/Keras, using Keras's built-in training loop (`compile` and `fit`) instead of the hand-written PyTorch loop. Results are exported to `pixel_predictions_keras.csv`.

The script `plot_predictions.py` visualizes the output of `model.py`. It shows the mean absorbance image, the per-pixel prediction map, and the library match score map side by side, along with a histogram of match scores to help choose the `MIN_MATCH_CORR` cutoff.

## Findings

Results are for the `dat2` sample (39,592 pixels), trained on 15,333 FTIR library spectra. All three models share the same "no match" step, so they agree that 35,061 pixels (88.6%) are background, leaving 4,531 pixels (11.4%) to classify.

Logistic regression reached about 95% cross-validated accuracy on the library. On the image it classified 3,607 pixels (9.1%) as plastic and 924 (2.3%) as not plastic. Its plastic predictions line up with particle-shaped regions in the image, mostly matching polyolefins (polyethylene/polypropylene), and agree with each pixel's closest library match 81% of the time.

The PyTorch and Keras CNNs both reached about 99% cross-validated accuracy, but on the image they classified far fewer pixels as plastic: 2,200 (5.6%) for PyTorch and 1,169 (3.0%) for Keras. They agree with each pixel's closest library match only 51% and 38% of the time.

We suspect the CNNs' accuracy is higher because cross-validation tests them on held-out library spectra, which are clean lab measurements similar to their training data, while image pixels are noisier and contain detector-stripe artifacts. The CNNs are flexible enough to learn fine details of the library spectra that help on the library but don't survive the noise in real image pixels (overfitting and domain shift). Logistic regression is a simpler model that relies on broader features, so it holds up better on the image. The image has no ground-truth labels, so agreement with the closest library match is only a proxy for accuracy. For now, logistic regression appears to be the most reliable model on real data. Training the CNNs with added noise and small wavenumber shifts (data augmentation) could help close the gap.
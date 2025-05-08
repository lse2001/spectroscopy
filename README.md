# Spectroscopy-Based Materials Classification (FTIR)

This project focuses on classifying materials based on Fourier Transform Infrared (FTIR) spectroscopy data using hyperspectral image analysis. Developed as part of ongoing work with the Moore Institute for Plastic Pollution Research (MIPPR), the goal is to preprocess, visualize, and classify materials at the pixel level using reflectance data from ENVI-format files.

The script `main.py` loads the ENVI-format hyperspectral image and extracts wavelength metadata. It also saves the hyperspectral data and metadata to NumPy `.npy` files for use by other scripts.
In `main.py` you may edit the `multiplier` parameter when filtered_means is called, as well as the vars `hdr_file` and `dat_file` to your desired specifications.
Need to manually specify the following in the hdr files when using new data samples:
byte order = 0
data file = dat_file_name.dat

The script `sample.py` displays the hyperspectral data using both the `spectral` library and `matplotlib.pyplot` to show how each method renders pixel values. It demonstrates two visualization techniques using bands of the 3D hyperspectral image array.

The script `histogram.py` creates visualizations of the average absorbance across all pixels. It helps assess the general signal quality and verify reflectance characteristics of the full image before filtering.

The script `functions.py` contains reusable utilities for spectrum analysis, including mean absorbance calculations, pixel filtering logic, and band indexing functions used in other parts of the project.

The script `selection.py` applies statistical filtering to identify high-quality pixels based on mean absorbance across wavelengths. These selected “good pixels” are stored with their corresponding absorbance spectra in a NumPy `.npy` file for downstream classification.
To change the bounds of acceptable good pixels you may edit:
lower_threshold = overall_mean_absorbance + std_dev_absorbance
upper_threshold = overall_mean_absorbance + 2 * std_dev_absorbance

The script `model.py` loads labeled FTIR training data from `raw.csv`, cleans the data by handling NaN and infinite values, and trains a logistic regression model using scikit-learn. It then interpolates the absorbance spectrum of each good pixel to align with the training wavelength grid and predicts whether each pixel corresponds to plastic or non-plastic material. Results are exported to `pixel_predictions.csv`.

import numpy as np
from functions import *
import random

"""
Note:
Even after IQR filtering, a pixel's mean absorbance value might not always represent a meaningful material signal.
It is possible that a pixel with a "normal" mean simply averaged out random noise,
rather than containing a true absorbance spectrum from the sample.
Therefore, when selecting pixels for plotting spectra, we prefer pixels with higher-than-average mean absorbance,
as these are more likely to correspond to real material interaction rather than background noise.
"""


# Load hyperspectral data
data = np.load('hyperspectral_data.npy', allow_pickle=True)
print("Loaded hyperspectral data with shape:", data.shape)

# Load wavelengths
wavelengths = np.load('wavelengths.npy')
wavelengths = wavelengths.astype(float)
print("Loaded wavelengths array with shape:", wavelengths.shape)


# Find mean absorbance per pixel (mean across all wavenumbers for that pixel)
mean_absorbances_per_pixel = np.mean(data, axis=2)  # axis=2 means collapse across wavenumbers

# Calculate standard deviation of mean absorbances
std_dev_absorbance = np.std(mean_absorbances_per_pixel)

# Find overall mean absorbance
overall_mean_absorbance = np.mean(mean_absorbances_per_pixel)
print("Overall mean absorbance across all pixels:", overall_mean_absorbance)

# Define lower and upper thresholds for filtering
lower_threshold = overall_mean_absorbance + std_dev_absorbance
upper_threshold = overall_mean_absorbance + 2 * std_dev_absorbance

# Create a boolean mask for pixels between 1 and 2 standard deviations above the mean
good_pixels_mask = np.logical_and(
    mean_absorbances_per_pixel >= lower_threshold,
    mean_absorbances_per_pixel <= upper_threshold
)

# Count good pixels
num_good_pixels = np.sum(good_pixels_mask)
total_pixels = mean_absorbances_per_pixel.size
percentage_good_pixels = (num_good_pixels / float(total_pixels)) * 100

print("Number of selected good pixels:", num_good_pixels)
print("Total number of pixels:", total_pixels)



# Build list of all good pixels
good_pixels_list = []

rows, cols = mean_absorbances_per_pixel.shape

for y in range(rows):
    for x in range(cols):
        if good_pixels_mask[y, x]:
            absorbance_array = data[y, x, :]  # <-- NEW: get absorbance spectrum
            good_pixels_list.append((y, x, absorbance_array))  # <-- NEW: save (row, col, absorbance)


print("Total number of good pixels found:", len(good_pixels_list))
print("Percentage of good pixels: %.2f%%" % percentage_good_pixels)

# SAVE the good pixels here
np.save('good_pixels.npy', np.array(good_pixels_list, dtype=object))
print("Saved good_pixels.npy with", len(good_pixels_list), "good pixels.")

# Randomly pick 5 good pixels
chosen_pixels = random.sample(good_pixels_list, 5)

print("Chosen pixels (y, x):", [(y, x) for (y, x, _) in chosen_pixels])

"""
# Plot each chosen pixel
for y, x in chosen_pixels:
    absorbance_spectrum = data[y, x, :]
    plot_pixel_spectrum(wavelengths, absorbance_spectrum, y, x)

plt.show()
"""

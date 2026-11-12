from functions import *
import matplotlib.pyplot as plt
import numpy as np
import os

# Define filenames
means_filename = "mean_absorbance_values.npy"
filtered_filename = "filtered_mean_absorbance_values.npy"

# Check if the required .npy files exist
if not os.path.exists(means_filename) or not os.path.exists(filtered_filename):
    raise IOError("Required data files not found. Please run main.py first to prepare mean_absorbance_values.npy and filtered_mean_absorbance_values.npy.")

# Load original mean absorbance values
means = np.load(means_filename, allow_pickle=True)
means = [tuple(item) for item in means]
print("Loaded original mean absorbance values from %s." % means_filename)

# Load filtered mean absorbance values
filtered_means = np.load(filtered_filename, allow_pickle=True)
filtered_means = [tuple(item) for item in filtered_means]
print("Loaded filtered mean absorbance values from %s." % filtered_filename)

# Calculate and print percent of data removed by filtering
original_length = len(means)
filtered_length = len(filtered_means)
fraction_remaining = filtered_length / float(original_length)
percent_removed = (1 - fraction_remaining) * 100

print("Length of original means:", original_length)
print("Length of filtered means:", filtered_length)
print("Percent of data removed after filtering: %.2f%%" % percent_removed)

"""
Note:
We are not plotting a sampling distribution here.
Instead, we are plotting the true distribution of mean absorbance values,
where each value corresponds to the mean absorbance measured at a single pixel
across all wavelengths.

A sampling distribution would require taking repeated random samples of pixels
and calculating the distribution of their sample means. That is not what we are doing.
We are working directly with the full set of measured pixel means (the population data).
"""

# Plot original distribution
plt.figure(1)
plot_mean_absorbance(means)
plt.title("Distribution of Mean Absorbance per Pixel (Original)")

# Plot filtered distribution
plt.figure(2)
plot_mean_absorbance(filtered_means)
plt.title("Distribution of Mean Absorbance per Pixel (After IQR Filtering)")

# Show all figures
plt.show()

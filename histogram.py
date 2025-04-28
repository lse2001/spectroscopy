from main import data, metadata
from functions import *
import matplotlib.pyplot as plt
import numpy as np
import os


filename = "mean_absorbance_values.npy"

# Step 1: Calculate/load the means
if os.path.exists(filename):
    means = np.load(filename, allow_pickle=True)
    # NumPy's save/load preserves the data values but not the exact object types (e.g., tuples become lists).
    # To restore the original list of tuples after loading, we manually convert each list back into a tuple.
    means = [tuple(item) for item in means]
    print("Loaded mean absorbance values from %s." % filename)
else:
    means = get_mean_absorbance_list(data)
    means_array = np.array(means, dtype=object)
    np.save(filename, means_array)
    print("Calculated and saved mean absorbance values to %s." % filename)
    print(means[:5])



means = [tuple(item) for item in means]


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


# Step 2: Plot original distribution
plt.figure(1)
plot_mean_absorbance(means)
plt.title("Distribution of Mean Absorbance per Pixel (Original)")

# Step 3: Filter the data using IQR
filtered_means = filter_by_iqr(means, multiplier=.8)

# Step 4: Plot filtered distribution
plt.figure(2)
plot_mean_absorbance(filtered_means)
plt.title("Distribution of Mean Absorbance per Pixel (After IQR Filtering)")

# Step 5: Calculate and print percent of data removed
original_length = len(means)
filtered_length = len(filtered_means)

fraction_remaining = filtered_length / float(original_length)
percent_removed = (1 - fraction_remaining) * 100

print("Length of original means:", len(means))
print("Length of filtered means:", len(filtered_means))

print("Percent of data removed after filtering: %.2f%%" % percent_removed)

# Step 6: Show all figures
plt.show()

# Step 7: Save the filtered mean absorbance values for later selection
filtered_array = np.array(filtered_means, dtype=object)
np.save("filtered_mean_absorbance_values.npy", filtered_array)
print("Filtered mean absorbance values saved successfully.")

from idlelib.Bindings import menudefs

import spectral
import matplotlib.pyplot as plt
import numpy as np


def get_mean_absorbance_list(data):
    """
    Calculates the mean absorbance for every pixel in a hyperspectral image.

    Args:
        data: 3D array-like (rows, cols, bands) - each pixel contains a spectrum of absorbance values.

    Returns:
        A flat list of floats: one mean absorbance value per pixel.
        Length of list = rows * cols.
    """
    rows, cols, bands = data.shape
    mean_values = []
    for y in range(rows):
        for x in range(cols):
            # data[y, x, :] returns a 3D slice like (1, 1, bands); we flatten it to (bands,)
            # so we get a 1D list of absorbance values we can loop over
            spectrum = np.array(data[y, x, :]).flatten()
            total = 0
            for value in spectrum:
                total += value
            mean_absorbance = total / len(spectrum)
            mean_values.append(mean_absorbance)
    return mean_values


def plot_mean_absorbance_histogram(means):
    """
    Plots a histogram of mean absorbance values.

    Args:
        means: List or array of mean absorbance values.

    Returns:
        None
    """
    plt.hist(means, bins=50, edgecolor='black')
    plt.xlabel('Mean Absorbance')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram of Mean Absorbance Values')
    plt.grid(True)


def calculate_iqr_bounds(means):
    """
    Calculates the 25th percentile (Q1) and 75th percentile (Q3) bounds
    for a list of mean absorbance values, without altering the original order.

    We do not sort the list manually because the order of values corresponds
    to specific pixel positions in the hyperspectral image. Sorting would break
    the mapping between each mean absorbance and its (x, y) location.

    Args:
        means: List or array of mean absorbance values.

    Returns:
        (q1, q3): Tuple of floats representing Q1 and Q3.
    """
    means_array = np.array(means)
    q1 = np.percentile(means_array, 25)
    q3 = np.percentile(means_array, 75)
    return q1, q3


def filter_by_iqr(means, q1, q3, multiplier=1.0):
    """
    Filters mean absorbance values using the IQR method.

    Args:
        means: List of mean absorbance values.
        q1: First quartile (25th percentile).
        q3: Third quartile (75th percentile).
        multiplier: Multiplier for the IQR to control filtering strictness (default 1.0).

    Returns:
        A filtered list containing only values within the specified IQR bounds.
        Original list order is preserved.

    Notes:
        - The IQR (Interquartile Range) covers the middle 50 percent of the data (between Q1 and Q3).
        - In a normally distributed dataset, 1 IQR is approximately equal to 1.35 standard deviations (sigma).
        - Using 1.0 times IQR removes data more than roughly plus or minus 1.35 standard deviations from the mean (keeps about 70 to 75 percent of the data).
        - Using 1.5 times IQR removes data more than roughly plus or minus 2.0 standard deviations from the mean (keeps about 95 percent of the data).
        - Lower multipliers create stricter filters (cut closer to the mean), and higher multipliers create looser filters.
    """
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered = []
    for value in means:
        if lower_bound <= value <= upper_bound:
            filtered.append(value)

    return filtered


envi_file = "MIPPR_IDC_ARA_PB_01.hdr"
image = spectral.open_image(envi_file)  # Manually add the parameter "byte order" to the hdr file since it was not specified. 0 corresponds to little-endian and 1 corresponds to big-endian.
# image is BipFile type, similar to SpyFile

data = image.load()
# data is ImageArray type (similar to NumPy array)

# Access ALL metadata, is stored as dict
metadata = image.metadata

print(data.shape)

"""
means = get_mean_absorbance_list(data)
np.save("mean_absorbance_values.npy", np.array(means))
"""

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


# Step 1: Load or calculate the original means
means = np.load("mean_absorbance_values.npy")  # Original data
print("Length of original means:", len(means))

# Step 2: Plot original distribution
plt.figure(1)
plot_mean_absorbance_histogram(means)
plt.title("Distribution of Mean Absorbance per Pixel (Original)")

# Step 3: Calculate IQR bounds on the original means
q1, q3 = calculate_iqr_bounds(means)

# Step 4: Filter but do NOT overwrite 'means'
filtered_means = filter_by_iqr(means, q1, q3, multiplier=1)

# Step 5: Plot filtered distribution
plt.figure(2)
plot_mean_absorbance_histogram(filtered_means)
plt.title("Distribution of Mean Absorbance per Pixel (After IQR Filtering)")

print("Length after filtering:", len(filtered_means))

fraction = len(filtered_means) / float(len(means))
print("Percent of data removed from filtering: %.2f%%" % ((1-fraction)*100))

# Step 6: Show all figures
plt.show()

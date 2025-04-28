# main.py
# Loads hyperspectral data, calculates mean absorbance values, filters outliers using the IQR method,
# and saves both the raw and filtered results as .npy files.
# Also prints useful statistics about how much data was filtered.

import os
import datetime
from dotenv import load_dotenv
import spectral
import numpy as np
from functions import get_mean_absorbance_list, filter_by_iqr

load_dotenv()  # Load environment variables from .env file

envi_file = os.getenv("ENVI_FILE")

if envi_file is None:
    raise ValueError("Environment variable ENVI_FILE not set. Please define it before running.")

image = spectral.open_image(envi_file)  # Manually add the parameter "byte order" to the hdr file since it was not specified. 0 corresponds to little-endian and 1 corresponds to big-endian.
# image is BipFile type, similar to SpyFile

data = image.load()
# data is ImageArray type (similar to NumPy array)

# Access ALL metadata, is stored as dict
metadata = image.metadata

# Calculate mean absorbance values
means = get_mean_absorbance_list(data)

# Save raw mean absorbance values
means_array = np.array(means, dtype=object)
np.save("mean_absorbance_values.npy", means_array)
print("Saved original mean absorbance values to mean_absorbance_values.npy.")
print("Timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Filter mean absorbance values using IQR method
filtered_means = filter_by_iqr(means, multiplier=1.0)

# Save filtered mean absorbance values
filtered_array = np.array(filtered_means, dtype=object)
np.save("filtered_mean_absorbance_values.npy", filtered_array)
print("Saved filtered mean absorbance values to filtered_mean_absorbance_values.npy.")
print("Timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Calculate and print % of data filtered
original_length = len(means)
filtered_length = len(filtered_means)
fraction_remaining = filtered_length / float(original_length)
percent_removed = (1 - fraction_remaining) * 100
print("Length of original means:", len(means))
print("Length of filtered means:", len(filtered_means))
print("Percent of data removed after filtering: %.2f%%" % percent_removed)

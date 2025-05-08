# main.py
# Loads hyperspectral data, calculates mean absorbance values, filters outliers using the IQR method,
# and saves both the raw and filtered results as .npy files.
# Also prints useful statistics about how much data was filtered.

import os
import datetime
from spectral import envi
import numpy as np
from functions import get_mean_absorbance_list, filter_by_iqr


# Only calculate/save if running main.py directly
if __name__ == "__main__":
    # Load environment variables

    data_dir = "data"
    hdr_file = os.path.join(data_dir, "hdr2.hdr")
    dat_file = os.path.join(data_dir, "dat2.dat")

    image = envi.open(hdr_file, image=dat_file)
    data = image.load()
    metadata = image.metadata

    # image = BipFile (a type of SpyFile)
    # data = ImageArray (similar to a 3D NumPy array)
    # After calling .load(), image -> data becomes full (rows, cols, bands) array
    # BipFile contains metadata (like wavelengths) in image.metadata
    # .hdr files might require manually specifying byte order if missing

    # Save full hyperspectral data
    np.save("hyperspectral_data.npy", data)
    print("Saved hyperspectral image array to hyperspectral_data.npy.")
    print("Timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Save wavelengths separately
    wavelengths = np.array(image.metadata['wavelength'])  # or 'wavenumber' if available
    np.save("wavelengths.npy", wavelengths)
    print("Saved wavelengths array to wavelengths.npy.")
    print("Timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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
    print("Length of original means:", original_length)
    print("Length of filtered means:", filtered_length)
    print("Percent of data removed after filtering: %.2f%%" % percent_removed)

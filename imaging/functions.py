import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def get_mean_absorbance_list(data):
    """
    Calculates the mean absorbance for every pixel in a hyperspectral image.

    Args:
        data: 3D array-like (rows, cols, bands) - each pixel contains a spectrum of absorbance values.

    Returns:
        A flat list of (row, col, mean_absorbance) tuples: one tuple per pixel.
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
            # Store (row, col, mean_absorbance) instead of just mean_absorbance
            mean_values.append((y, x, mean_absorbance))
    return mean_values


def calculate_iqr_bounds(mean_values):
    """
    Calculates Q1, Q3, and IQR bounds for mean absorbance values.

    Args:
        mean_values: List of (row, col, mean_absorbance) tuples.

    Returns:
        q1, q3, iqr
    """
    absorbances = []
    for value in mean_values:
        absorbances.append(value[2])
    absorbances.sort()

    n = len(absorbances)
    mid = n // 2

    if n % 2 == 0:
        lower_half = absorbances[:mid]
        upper_half = absorbances[mid:]
    else:
        lower_half = absorbances[:mid]
        upper_half = absorbances[mid+1:]

    def median(values):
        m = len(values)
        mid_idx = m // 2
        if m % 2 == 0:
            return (values[mid_idx - 1] + values[mid_idx]) / 2.0
        else:
            return values[mid_idx]

    q1 = median(lower_half)
    q3 = median(upper_half)
    iqr = q3 - q1

    return q1, q3, iqr


def filter_by_iqr(mean_values, multiplier=1.0):
    """
    Filters mean absorbance values using the IQR method and Tukey's Rule for identifying outliers.

    Args:
        mean_values: List of (row, col, mean_absorbance) tuples.
        multiplier: Multiplier for the IQR to control filtering strictness (default 1.0).

    Returns:
        A filtered list containing only (row, col, mean_absorbance) tuples within the specified IQR bounds.
        Original list order is preserved.

    Notes:
        - IQR covers the middle 50 percent of the data.
        - 1.0 times IQR removes points roughly plus or minus 1.35 standard deviations from the mean (keeps about 70 to 75 percent).
        - 1.5 times IQR removes points roughly plus or minus 2.0 standard deviations from the mean (keeps about 95 percent).
        - Lower multipliers create stricter filters; higher multipliers create looser filters.
        - For compact datasets (small IQR), larger multipliers have less filtering impact.
    """
    q1, q3, iqr = calculate_iqr_bounds(mean_values)

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = (q3 + multiplier * iqr)

    filtered = []
    for value in mean_values:
        absorbance = value[2]
        if lower_bound <= absorbance <= upper_bound:
            filtered.append(value)

    return filtered


def plot_mean_absorbance(mean_values):
    """
    Plots a histogram of mean absorbance values.

    Args:
        mean_values: List of (row, col, mean_absorbance) tuples.
    """
    # Extract only the mean absorbance values (z component) for plotting
    absorbances = []
    for value in mean_values:
        absorbances.append(value[2])

    plt.hist(absorbances, bins=50, color='orange', edgecolor='black')
    plt.xlabel('Mean Absorbance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mean Absorbance Values')
    plt.grid(True)


def plot_pixel_spectrum(wavelengths, absorbance_spectrum, y, x):
    """
    Plot wavelength (cm-1) vs absorbance for a selected pixel.

    Args:
        wavelengths (np.ndarray): 1D array of wavenumbers in cm-1.
        absorbance_spectrum (np.ndarray): 1D array of absorbance values at the selected pixel.
        y (int): Row index of the selected pixel.
        x (int): Column index of the selected pixel.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))  # Make the figure wider
    plt.plot(wavelengths, absorbance_spectrum)
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Absorbance')
    plt.title("Absorbance Spectrum at Chosen Pixel\nRow: {} Column: {}".format(y, x))

    # Set custom x-ticks: 10 evenly spaced ticks across the wavenumber range
    ax = plt.gca()
    min_wavenumber = np.min(wavelengths)
    max_wavenumber = np.max(wavelengths)
    ticks = np.linspace(min_wavenumber, max_wavenumber, num=10)
    ticks = np.round(ticks).astype(int)
    ax.set_xticks(ticks)

    plt.tight_layout()  # Prevent label cutoff

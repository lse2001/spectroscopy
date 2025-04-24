import spectral
import matplotlib.pyplot as plt
import numpy as np

"""
Function
A function is something you call on its own, and you pass in the object as a parameter.
Example:
len("hello")         # len is a function
np.mean([1, 2, 3])   # mean is a NumPy function

Method
A method is a function that is attached to an object. You call it on the object using dot syntax.
Example:
"hello".upper()           # upper is a method of a string
some_array.flatten()      # flatten is a method of a NumPy array
"""


def get_mean_absorbance_list(data):
    """
    Calculates the mean absorbance for every pixel in a hyperspectral image.

    Args:
        data: 3D array-like (rows, cols, bands) - each pixel contains a spectrum of absorbance values.

    Returns:
        A flat list of floats: one mean absorbance value per pixel.
        Length of list = rows * cols.
    """
    import numpy as np
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

means = np.load("mean_absorbance_values.npy")
print(means[:10])  # show first 10 values

"""
You're able to use .squeeze() directly because data is a spectral.ImageArray, which is a subclass of NumPy's ndarray under the hood.
So even though squeeze() is technically a NumPy method, it's available as a method of the array object itself: array.squeeze() is the same as np.squeeze(array)
"""









"""
spectrum = data[0, 0, :].squeeze()  # Creates list of absorbances for each wavenumber at pixel (0,0)
# Find mean absorbance at (0,0) using loop
total = 0
for value in spectrum:
    total += value

mean_absorbance = total / len(spectrum)
print("Mean absorbance at pixel (0, 0):", mean_absorbance)
"""
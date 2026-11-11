import numpy as np
import matplotlib.pyplot as plt


"""
The wavelength list represents the number of spectral bands - i.e., how many different wavelengths were measured for each pixel.
The number of pixels = lines * samples
lines = number of rows
samples = number of columns
So if lines = 583 and samples = 571, then you have 332,993 pixels
Each pixel has one reflectance value per wavelength, so:
wavelengths = 427 means each pixel has a 427-element spectrum

Each pixel at position [row, col] has a 1D spectrum (427 values in your case - one for each wavelength).
"""

"""
Hyperspectral data structure:
The data is a 3D array with shape (rows, cols, bands).
Each pixel (x, y) has a full spectrum of reflectance values across wavelengths.
Accessing data[y, x, band] gives the reflectance at:
    - spatial position (x, y)
    - specific wavelength corresponding to 'band' index
In other words:
Reflectance = f(x, y, wavelength)
Example:
data[200, 100, 42] = reflectance at pixel (x=100, y=200) and 43rd wavelength.
"""

data = np.load("hyperspectral_data.npy", allow_pickle=True)

fig2 = plt.figure()  # For plt.imshow

plt.figure(fig2.number)  # Activate fig2
wavelengths = np.load("wavelengths.npy").astype(float)
band = np.argmin(np.abs(wavelengths - 2915))        # slice closest to 2915 cm-1 (C-H stretch)
band_image = data[:, :, band]
low, high = np.percentile(band_image, [1, 99])      # ignore extreme outliers
plt.imshow(band_image, cmap='gray', vmin=low, vmax=high)
plt.title("Band %d (%.0f cm-1)" % (band, wavelengths[band]))

plt.suptitle("Hyperspectral Data - plt.imshow()")
plt.gca().set_xlabel("X Position")
plt.gca().set_ylabel("Y Position")

plt.show(block=True)

"""
spectral.imshow()
Creates an ImageView object (from the spectral package) in a new figure
plt.imshow()
Creates an AxesImage object (from Matplotlib) in the current axes

spectral.imshow(data, bands=(0,0,0))  # Creates NEW figure
plt.imshow(data[:,:,[0,0,0]])         # Overwrites CURRENT axes
"""
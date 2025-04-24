import spectral
import matplotlib.pyplot as plt

envi_file = "MIPPR_IDC_ARA_PB_01.hdr"
image = spectral.open_image(envi_file)  # Manually add the parameter "byte order" to the hdr file since it was not specified. 0 corresponds to little-endian and 1 corresponds to big-endian.
# image is BipFile type, similar to SpyFile

data = image.load()
# data is ImageArray type (similar to NumPy array)


# Access ALL metadata, is stored as dict
metadata = image.metadata
# Print metadata, u-prefix means unicode object
for k, v in metadata.items():
    print(k, v)

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



fig1 = plt.figure()  # For spectral.imshow
fig2 = plt.figure()  # For plt.imshow

# Plot to specific figures
plt.figure(fig1.number)  # Activate fig1
spectral.imshow(data, bands=(0,0,0), fignum=fig1.number)  # spectral.imshow() Creates NEW figure, so we have to explicitly state which figure to populate
plt.suptitle("Hyperspectral Data - spectral.imshow()")
plt.gca().set_xlabel("X Position")
plt.gca().set_ylabel("Y Position")

plt.figure(fig2.number)  # Activate fig2
plt.imshow(data[:,:,[0,0,0]])  # plt.imshow() overwrites CURRENT figure?
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

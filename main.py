import spectral
import matplotlib.pyplot as plt

envi_file = "MIPPR_IDC_ARA_PB_01.hdr"
image = spectral.open_image(envi_file)  # Manually add the parameter "byte order" to the hdr file since it was not specified. 0 corresponds to little-endian and 1 corresponds to big-endian.
# image is BipFile type, similar to SpyFile

data = image.load()
# data is ImageArray type (similar to NumPy array)


plt.imshow(data[:,:,[0,0,0]])  # AxesImage type
spectral.imshow(data, bands=(0,0,0))  # ImageView type
# spectral.imshow expects numPy array-like object or SpyFile object, that is why we have to load the data from BipFile first to be ImageArray, which is similar to NumPy array

plt.show(block="True")

"""something about the imshow() method of spectral package vs imshow() method of plt package is changing the objects that are being displayed by plt.show()
I think this issues comes from calling plt.imshow() after spectral.imshow() has already created objects. plt.imshow() deletes the previously created objects"""


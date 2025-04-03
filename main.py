import spectral
import matplotlib.pyplot as plt

envi_file = "MIPPR_IDC_ARA_PB_01.hdr"
image = spectral.open_image(envi_file)  # Manually add the parameter "byte order" to the hdr file since it was not specified. 0 corresponds to little-endian and 1 corresponds to big-endian.
# image is BipFile type, similar to SpyFile

data = image.load()
# data is ImageArray type (similar to NumPy array)

view = spectral.imshow(data, (29, 19, 9))
# imshow expects np.ndarray or SpyFile object, that is why we have to load the data from BipFile first
# view is ImageView type

plt.show()

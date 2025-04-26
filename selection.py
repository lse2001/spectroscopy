import spectral
import numpy as np

"""
Note:
Even after IQR filtering, a pixel's mean absorbance value might not always represent a meaningful material signal.
It is possible that a pixel with a "normal" mean simply averaged out random noise,
rather than containing a true absorbance spectrum from the sample.
Therefore, when selecting pixels for plotting spectra, we prefer pixels with higher-than-average mean absorbance,
as these are more likely to correspond to real material interaction rather than background noise.
"""


def get_mean_absorbance_tuple(data):
    mean_absorbance_per_pixel = []
    rows, cols, bands = data.shape
    for y in range(rows):
        for x in range(cols):
            spectrum = data[y,x,:].flatten()
            mean_absorbance = np.mean(spectrum)
            mean_absorbance_per_pixel.append((y, x, mean_absorbance))
    return mean_absorbance_per_pixel



envi_file = "MIPPR_IDC_ARA_PB_01.hdr"
image = spectral.open_image(envi_file)  # Manually add the parameter "byte order" to the hdr file since it was not specified. 0 corresponds to little-endian and 1 corresponds to big-endian.
# image is BipFile type, similar to SpyFile

data = image.load()
# data is ImageArray type (similar to NumPy array)

print(data.shape)
# (number of rows vertically, number of columns horizontally, number of bands spectrally)

mean_absorbance_per_pixel = get_mean_absorbance_tuple(data)
print(len(mean_absorbance_per_pixel))

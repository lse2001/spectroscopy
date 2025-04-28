from main import data, metadata
from functions import *

"""
Note:
Even after IQR filtering, a pixel's mean absorbance value might not always represent a meaningful material signal.
It is possible that a pixel with a "normal" mean simply averaged out random noise,
rather than containing a true absorbance spectrum from the sample.
Therefore, when selecting pixels for plotting spectra, we prefer pixels with higher-than-average mean absorbance,
as these are more likely to correspond to real material interaction rather than background noise.
"""

mean_absorbances_per_pixel = get_mean_absorbance_list(data)

import os
from dotenv import load_dotenv
import spectral

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
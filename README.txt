This is a materials classification project based on IR spectroscopy. This is my second project at MIPPR.


In spectra.py:
We are able to use .squeeze() directly because data is a spectral.ImageArray, which is a subclass of NumPy's ndarray under the hood.
So even though squeeze() is technically a NumPy method, it's available as a method of the array object itself: array.squeeze() is the same as np.squeeze(array)


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


"""
spectrum = data[0, 0, :].squeeze()  # Creates list of absorbances for each wavenumber at pixel (0,0)
# Find mean absorbance at (0,0) using loop
total = 0
for value in spectrum:
    total += value

mean_absorbance = total / len(spectrum)
print("Mean absorbance at pixel (0, 0):", mean_absorbance)
"""
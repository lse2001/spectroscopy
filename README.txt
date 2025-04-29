This is a materials classification project based on FTIR spectroscopy. This is my second project at MIPPR.
Below are some personal notes for myself on Python.


In spectra.py:
We are able to use .squeeze() directly because data is a spectral.ImageArray, which is a subclass of NumPy's ndarray under the hood.
So even though squeeze() is technically a NumPy method, it's available as a method of the array object itself: array.squeeze() is the same as np.squeeze(array)

# .flatten() turns any array into a 1D array.
# It collapses all dimensions into a single list of values.
# Example: (583, 571) -> (332893,)
# Always creates a copy.

# .squeeze() removes dimensions of size 1.
# It only removes axes where the size is 1.
# Example: (1, 427) -> (427,)
# Does nothing if there are no singleton dimensions.


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
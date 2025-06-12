import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

# Load good pixels
good_pixels = np.load('good_pixels.npy', allow_pickle=True)
print("Processing {} good pixels...".format(len(good_pixels)))

# Load and prepare training data
training_data = pd.read_csv("raw.csv", low_memory=False)
spectral_columns = [col for col in training_data.columns if col.replace('.','').isdigit()]

# Clean spectral data - replace inf and fill NaNs with column mean
spectral_data = training_data[spectral_columns]
spectral_data = spectral_data.replace([np.inf, -np.inf], np.nan)
spectral_data = spectral_data.fillna(spectral_data.mean())

# Verify no NaN/inf values remain
if np.any(np.isnan(spectral_data)) or np.any(np.isinf(spectral_data)):
    raise ValueError("Data still contains NaN or inf values after cleaning")

labels = training_data['plastic_or_not']

# Train model
model = LogisticRegression(max_iter=3000, solver='lbfgs')
model.fit(spectral_data, labels)

# Load wavelengths for interpolation
original_wavelengths = np.load('wavelengths.npy').astype(float)
target_wavelengths = np.array([float(w) for w in spectral_columns])

# Process all pixels
start_time = time.time()
predictions = []

for pixel in good_pixels:
    y, x, absorbance_array = pixel
    interpolated = np.interp(target_wavelengths, original_wavelengths, absorbance_array)
    prediction = model.predict(interpolated.reshape(1, -1))[0]
    predictions.append([y, x, prediction])

# Convert predictions to DataFrame and save
results_df = pd.DataFrame(predictions, columns=['row', 'column', 'is_plastic'])
results_df.to_csv('pixel_predictions.csv', index=False)

# Print summary
print("Processing complete in {:.2f} seconds".format(time.time() - start_time))
print("Average time per pixel: {:.2f} ms".format((time.time() - start_time) * 1000 / len(good_pixels)))
print("Results saved to pixel_predictions.csv")

# Print helpful model attributes
print("Model coefficients (first 5 features):")
print(model.coef_[0][:5])

print("\nModel intercept:")
print(model.intercept_)

print("\nClasses:")
print(model.classes_)

print("\nNumber of iterations until convergence:")
print(model.n_iter_)

# Model accuracy on training data
train_score = model.score(spectral_data, labels)
print("\nTraining accuracy: %.2f%%" % (train_score * 100))

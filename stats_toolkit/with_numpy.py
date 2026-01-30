import numpy as np

data = [2, 4, 6, 8]

arr = np.array(data)
n = arr.size

print("Data: ", arr.tolist())
print("-" * 50)

mean = np.mean(arr)
print(f"Mean (μ): {mean}")

deviations = arr - mean
print(f"Deviations (x - μ): {deviations.tolist()}")

squared_deviations = deviations ** 2
print(f"Squared deviations: {squared_deviations.tolist()}")
print(f"Sum of squared deviations: {np.sum(squared_deviations)}")

variance = np.var(arr, ddof=1)
print(f"Variance (σ², sample): {variance}")

std = np.std(arr, ddof=1)
print(f"Standard deviation (σ): {std}")

standard_error = std / np.sqrt(n)
print(f"Standard error (SE): {standard_error}")

z = 1.96
ci = (mean - z * standard_error, mean + z * standard_error)
print(f"95% confidence interval for mean: {ci}")

z_scores = (arr - mean) / std
print(f"Z-score normalized data: {z_scores.tolist()}")

print("-" * 50)
print("INTERPRETATION:")
print(f"- Typical deviation from mean: ~{std:.2f}")
print(f"- Noise of the mean estimate: ~{standard_error:.2f}")
print(f"- More data → smaller SE → narrower CI")
print()
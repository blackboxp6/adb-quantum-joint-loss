
import numpy as np

from src.classical_copula_runner import run_classical_copula

# Generate the exact Student-t copula terminal distribution
result = run_classical_copula(
    family="student_t",
    df=5,
    include_distribution=True,
)

# Inspect the returned object
print("Result type:", type(result))
print("Available keys:", result.keys())

# Extract the terminal probability distribution
distribution = np.asarray(result["term_distribution"])

print("\nDistribution shape:", distribution.shape)
print("Number of states:", distribution.size)
print("Probability sum:", distribution.sum())
print("Minimum probability:", distribution.min())
print("Maximum probability:", distribution.max())

# Flatten into a one-dimensional probability vector
probabilities = distribution.reshape(-1)

print("\nFirst 10 probabilities:")
print(probabilities[:10])

assert np.all(probabilities >= 0)
assert np.isclose(probabilities.sum(), 1.0)
assert probabilities.size == 100

print("\nQuantum input validation: PASSED")

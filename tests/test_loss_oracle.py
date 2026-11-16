
import numpy as np

from src.classical_copula_runner import run_classical_copula
from src.quantum.loss_oracle import validate_loss_oracle


result = run_classical_copula(
    family="student_t",
    df=5,
    include_distribution=True,
)

probabilities = np.asarray(
    result["term_distribution"]
).reshape(-1)

verification = validate_loss_oracle(probabilities)

print("=" * 60)
print("QUANTUM LOSS ORACLE VALIDATION")
print("=" * 60)

print("Total states:", verification["number_of_states"])
print("Favorable states:", verification["number_of_good_states"])

print(
    "Classical exceedance probability:",
    verification["classical_probability"]
)

print("\nExpected benchmark: 0.01336411")

assert np.isclose(
    verification["classical_probability"],
    0.01336411,
    atol=1e-7,
)

print("\nLOSS ORACLE VALIDATION: PASSED")


import numpy as np
import pandas as pd

from src.classical_copula_runner import run_classical_copula
from src.quantum.loss_oracle import validate_loss_oracle
from src.quantum.grover import simulate_amplification


result = run_classical_copula(
    family="student_t",
    df=5,
    include_distribution=True,
)

probabilities = np.asarray(
    result["term_distribution"]
).reshape(-1)

oracle_result = validate_loss_oracle(probabilities)

results = simulate_amplification(
    probabilities,
    oracle_result["good_states"],
    max_iterations=10,
)

df = pd.DataFrame(results)

print("=" * 70)
print("GROVER AMPLITUDE AMPLIFICATION")
print("=" * 70)

print(df.to_string(index=False))

print("\nInitial probability:")
print(oracle_result["classical_probability"])

print("\nGrover simulation completed.")

assert np.allclose(
    df["quantum_probability"],
    df["theoretical_probability"],
    atol=1e-9,
)

print("\nGROVER VALIDATION: PASSED")

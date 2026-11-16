
import numpy as np

from src.classical_copula_runner import run_classical_copula

from src.quantum.quantum_state import (
    verify_quantum_state
)

# Obtain the validated Student-t distribution
result = run_classical_copula(
    family="student_t",
    df=5,
    include_distribution=True,
)

distribution = np.asarray(
    result["term_distribution"]
)

probabilities = distribution.reshape(-1)

# Prepare and verify the quantum state
verification = verify_quantum_state(probabilities)

print("=" * 60)
print("QUANTUM STATE PREPARATION")
print("=" * 60)

print("Number of classical states:", verification["n_states"])
print("Number of qubits:", verification["n_qubits"])

print(
    "Quantum state dimension:",
    len(verification["amplitudes"])
)

print(
    "Maximum probability error:",
    verification["max_probability_error"]
)

print(
    "Validation passed:",
    verification["validation_passed"]
)

print("\nQuantum circuit:")
print(verification["circuit"].draw())

assert verification["validation_passed"]

print("\nQUANTUM STATE VALIDATION: PASSED")

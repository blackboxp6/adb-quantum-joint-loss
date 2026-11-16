
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


def validate_probabilities(probabilities, atol=1e-10):
    """
    Validate a classical probability distribution.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array.")

    if probabilities.size == 0:
        raise ValueError("Probability distribution cannot be empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(probabilities.sum(), 1.0, atol=atol):
        raise ValueError("Probabilities must sum to one.")

    return probabilities


def build_state_preparation(probabilities):
    """
    Construct a quantum circuit encoding a classical
    probability distribution into quantum amplitudes.

    Returns:
        circuit: Quantum state preparation circuit
        amplitudes: Padded quantum amplitude vector
        n_qubits: Number of qubits
    """
    probabilities = validate_probabilities(probabilities)

    n_states = len(probabilities)

    # Minimum number of qubits required
    n_qubits = int(np.ceil(np.log2(n_states)))

    # Quantum state dimension must be a power of two
    dimension = 2 ** n_qubits

    # Pad unused basis states with zero probability
    padded_probabilities = np.zeros(dimension)
    padded_probabilities[:n_states] = probabilities

    # Convert probabilities into quantum amplitudes
    amplitudes = np.sqrt(padded_probabilities)

    # Construct quantum state preparation circuit
    circuit = QuantumCircuit(n_qubits)

    preparation = StatePreparation(amplitudes)

    circuit.append(preparation, range(n_qubits))

    return circuit, amplitudes, n_qubits


def verify_quantum_state(probabilities):
    """
    Verify that the prepared quantum state reproduces
    the original classical probability distribution.
    """
    probabilities = validate_probabilities(probabilities)

    circuit, amplitudes, n_qubits = build_state_preparation(
        probabilities
    )

    # Simulate the quantum statevector
    state = Statevector.from_instruction(circuit)

    # Recover probabilities from quantum amplitudes
    quantum_probabilities = state.probabilities()

    # Compare the first N basis states
    recovered_probabilities = quantum_probabilities[:len(probabilities)]

    max_error = np.max(
        np.abs(recovered_probabilities - probabilities)
    )

    return {
        "n_qubits": n_qubits,
        "n_states": len(probabilities),
        "circuit": circuit,
        "amplitudes": amplitudes,
        "quantum_probabilities": quantum_probabilities,
        "max_probability_error": max_error,
        "validation_passed": np.allclose(
            recovered_probabilities,
            probabilities,
            atol=1e-10
        )
    }

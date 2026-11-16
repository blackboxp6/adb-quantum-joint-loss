
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

from src.quantum.quantum_state import build_state_preparation


def build_phase_oracle(good_states, n_qubits):
    """
    Apply a phase flip to every favorable basis state.
    """

    dimension = 2 ** n_qubits
    good_states = np.asarray(good_states, dtype=bool)

    if len(good_states) > dimension:
        raise ValueError("Good-state mask exceeds quantum dimension.")

    padded_mask = np.zeros(dimension, dtype=bool)
    padded_mask[:len(good_states)] = good_states

    diagonal = np.ones(dimension)
    diagonal[padded_mask] = -1.0

    oracle = QuantumCircuit(n_qubits)
    oracle.append(DiagonalGate(diagonal), range(n_qubits))

    return oracle


def build_zero_reflection(n_qubits):
    """
    Reflection about the all-zero computational basis state.

    The global sign convention is chosen so that this
    operator is 2|0><0| - I.
    """

    dimension = 2 ** n_qubits

    diagonal = -np.ones(dimension)
    diagonal[0] = 1.0

    reflection = QuantumCircuit(n_qubits)
    reflection.append(
        DiagonalGate(diagonal),
        range(n_qubits)
    )

    return reflection


def build_grover_operator(preparation, good_states):
    """
    Construct Q = A S0 A† Sf.

    A is the state-preparation operator.
    Sf is the favorable-state phase oracle.
    S0 reflects about the zero state.
    """

    n_qubits = preparation.num_qubits

    oracle = build_phase_oracle(
        good_states,
        n_qubits
    )

    zero_reflection = build_zero_reflection(n_qubits)

    grover = QuantumCircuit(n_qubits)

    # Apply Sf
    grover.compose(oracle, inplace=True)

    # Apply A†
    grover.compose(preparation.inverse(), inplace=True)

    # Apply S0
    grover.compose(zero_reflection, inplace=True)

    # Apply A
    grover.compose(preparation, inplace=True)

    return grover


def simulate_amplification(
    probabilities,
    good_states,
    max_iterations=10,
):
    """
    Simulate Grover amplitude amplification and
    compare with its theoretical probabilities.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    good_states = np.asarray(good_states, dtype=bool)

    preparation, _, n_qubits = build_state_preparation(
        probabilities
    )

    grover = build_grover_operator(
        preparation,
        good_states
    )

    initial_probability = probabilities[good_states].sum()

    theta = np.arcsin(np.sqrt(initial_probability))

    results = []

    circuit = preparation.copy()

    for k in range(max_iterations + 1):

        state = Statevector.from_instruction(circuit)

        measured_probabilities = state.probabilities()

        quantum_probability = measured_probabilities[
            :len(good_states)
        ][good_states].sum()

        theoretical_probability = np.sin(
            (2 * k + 1) * theta
        ) ** 2

        results.append({
            "iterations": k,
            "quantum_probability": quantum_probability,
            "theoretical_probability": theoretical_probability,
            "absolute_error": abs(
                quantum_probability - theoretical_probability
            )
        })

        # Add one Grover iteration for the next step
        if k < max_iterations:
            circuit.compose(grover, inplace=True)

    return results

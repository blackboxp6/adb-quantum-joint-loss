
import time
import numpy as np

from qiskit.quantum_info import Statevector

from src.quantum.quantum_state import build_state_preparation
from src.quantum.grover import build_grover_operator


def theoretical_probability(a, k):
    """Favorable-state probability after k Grover iterations."""
    theta = np.arcsin(np.sqrt(a))
    return np.sin((2 * k + 1) * theta) ** 2


def normalize_measurement_probabilities(quantum_probabilities):
    """
    Correct floating-point errors in quantum measurement probabilities.

    Ensures that probabilities are finite, nonnegative, and sum to one.
    """
    quantum_probabilities = np.asarray(
        quantum_probabilities,
        dtype=float,
    )

    if not np.all(np.isfinite(quantum_probabilities)):
        raise ValueError(
            "Quantum measurement probabilities must be finite."
        )

    # Remove negligible negative numerical errors.
    quantum_probabilities = np.clip(
        quantum_probabilities,
        0.0,
        None,
    )

    total_probability = quantum_probabilities.sum()

    if total_probability <= 0:
        raise ValueError(
            "Invalid quantum probability distribution."
        )

    # Normalize the complete probability distribution.
    quantum_probabilities /= total_probability

    # Correct the final entry to avoid sum(p[:-1]) > 1.
    if len(quantum_probabilities) > 1:
        first_sum = quantum_probabilities[:-1].sum()

        if first_sum >= 1.0:
            quantum_probabilities[:-1] *= (
                (1.0 - 1e-15) / first_sum
            )

        quantum_probabilities[-1] = (
            1.0 - quantum_probabilities[:-1].sum()
        )

    return quantum_probabilities


def run_qae_experiment(
    probabilities,
    good_states,
    iterations=(0, 1, 2, 3, 4, 5, 6),
    shots=3000,
    seed=42,
):
    """
    Statevector simulation of finite-shot MLAE measurements.

    shots can be:
      - one integer used at every depth; or
      - a dictionary {Grover depth: number of shots}.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    good_states = np.asarray(
        good_states,
        dtype=bool,
    )

    if len(probabilities) != len(good_states):
        raise ValueError(
            "Probability and oracle dimensions differ."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Probabilities must be finite."
        )

    if np.any(probabilities < 0) or not np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Probabilities must be nonnegative and sum to 1."
        )

    iterations = tuple(sorted(set(iterations)))

    if not iterations or min(iterations) < 0:
        raise ValueError(
            "Iterations must be nonempty and nonnegative."
        )

    if not np.any(good_states):
        raise ValueError(
            "No favorable states were supplied."
        )

    if isinstance(shots, dict):
        shot_allocation = {
            k: int(shots[k]) for k in iterations
        }
    else:
        shot_allocation = {
            k: int(shots) for k in iterations
        }

    if any(value <= 0 for value in shot_allocation.values()):
        raise ValueError(
            "Every Grover depth must receive positive shots."
        )

    rng = np.random.default_rng(seed)

    print(
        "\n[QAE] Building state preparation...",
        flush=True,
    )

    preparation, _, n_qubits = build_state_preparation(
        probabilities
    )

    print(
        f"[QAE] State preparation complete: {n_qubits} qubits.",
        flush=True,
    )

    print(
        "[QAE] Building Grover operator...",
        flush=True,
    )

    grover = build_grover_operator(
        preparation,
        good_states,
    )

    state = Statevector.from_instruction(preparation)

    observations = []
    selected_depths = set(iterations)

    for k in range(max(iterations) + 1):

        if k in selected_depths:

            start = time.perf_counter()

            # Obtain quantum measurement probabilities.
            quantum_probabilities = state.probabilities()

            # Fix floating-point normalization errors.
            quantum_probabilities = (
                normalize_measurement_probabilities(
                    quantum_probabilities
                )
            )

            # Generate finite-shot measurement counts.
            counts = rng.multinomial(
                shot_allocation[k],
                quantum_probabilities,
            )

            # Count favorable computational basis states.
            favorable_count = int(
                counts[:len(good_states)][good_states].sum()
            )

            elapsed = time.perf_counter() - start

            observations.append({
                "iterations": k,
                "shots": shot_allocation[k],
                "favorable_count": favorable_count,
                "observed_probability": (
                    favorable_count / shot_allocation[k]
                ),
                "execution_time": elapsed,
            })

            print(
                f"[QAE] k={k}: "
                f"{favorable_count}/"
                f"{shot_allocation[k]} favorable",
                flush=True,
            )

        # Apply one Grover iteration before the next depth.
        if k < max(iterations):
            state = state.evolve(grover)

    return observations


def estimate_amplitude_mle(
    observations,
    grid_points=100001,
    upper_bound=0.10,
):
    """
    Grid-based maximum-likelihood amplitude estimation.

    Returns all disconnected profile-likelihood regions and
    the region containing the global MLE.
    """

    if not observations:
        raise ValueError(
            "No measurement observations supplied."
        )

    if grid_points < 2:
        raise ValueError(
            "grid_points must be at least 2."
        )

    if not 0 < upper_bound <= 1:
        raise ValueError(
            "upper_bound must be between 0 and 1."
        )

    amplitudes = np.linspace(
        1e-10,
        upper_bound,
        grid_points,
    )

    log_likelihood = np.zeros_like(amplitudes)

    for observation in observations:

        k = int(observation["iterations"])
        shots_k = int(observation["shots"])
        successes = int(observation["favorable_count"])
        failures = shots_k - successes

        if not 0 <= successes <= shots_k:
            raise ValueError(
                "Invalid favorable count."
            )

        predicted = theoretical_probability(
            amplitudes,
            k,
        )

        predicted = np.clip(
            predicted,
            1e-12,
            1 - 1e-12,
        )

        log_likelihood += (
            successes * np.log(predicted)
            + failures * np.log1p(-predicted)
        )

    best_index = int(np.argmax(log_likelihood))

    estimate = float(amplitudes[best_index])

    maximum_log_likelihood = float(
        log_likelihood[best_index]
    )

    # 95% profile-likelihood acceptance threshold.
    accepted = (
        2 * (maximum_log_likelihood - log_likelihood)
        <= 3.841459
    )

    # Find every contiguous accepted region.
    indices = np.flatnonzero(accepted)

    accepted_regions = []

    if len(indices):

        split_positions = (
            np.where(np.diff(indices) > 1)[0] + 1
        )

        for group in np.split(
            indices,
            split_positions,
        ):
            accepted_regions.append((
                float(amplitudes[group[0]]),
                float(amplitudes[group[-1]]),
            ))

    # Identify the region containing the global MLE.
    mle_region = None

    for lower, upper in accepted_regions:

        if lower <= estimate <= upper:
            mle_region = (lower, upper)
            break

    return {
        "estimated_amplitude": estimate,
        "confidence_interval_95": mle_region,
        "all_accepted_intervals_95": accepted_regions,
        "maximum_log_likelihood": maximum_log_likelihood,
        "grid_resolution": float(
            amplitudes[1] - amplitudes[0]
        ),
    }


import time
import numpy as np
import pandas as pd

from src.quantum.amplitude_estimation import (
    run_qae_experiment,
    estimate_amplitude_mle,
)


def run_classical_mc(
    probabilities,
    good_states,
    n_samples,
    seed=42,
):
    """
    Classical Monte Carlo estimation of the
    portfolio loss exceedance probability.
    """

    rng = np.random.default_rng(seed)

    start = time.perf_counter()

    sampled_states = rng.choice(
        len(probabilities),
        size=n_samples,
        p=probabilities,
    )

    exceedances = good_states[sampled_states]

    estimate = exceedances.mean()

    elapsed = time.perf_counter() - start

    standard_error = np.sqrt(
        estimate * (1 - estimate) / n_samples
    )

    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "samples": n_samples,
        "execution_time": elapsed,
    }


def run_benchmark(
    probabilities,
    good_states,
    shots_per_setting=(100, 300, 1000, 3000),
    repetitions=5,
    iterations=(0, 1, 2, 3, 4, 5, 6),
    seed=42,
):
    """
    Compare statevector-simulated MLAE with
    classical Monte Carlo at equal oracle-query
    and classical-sample budgets.

    This is a simulation benchmark, not a
    hardware quantum advantage experiment.
    """

    records = []

    classical_probability = np.sum(
        probabilities[good_states]
    )

    total_grover_iterations = sum(iterations)

    for shots in shots_per_setting:

        oracle_queries = (
            shots * total_grover_iterations
        )

        classical_samples = oracle_queries

        print("\n" + "=" * 65)
        print(f"SHOTS PER QAE SETTING: {shots}")
        print(f"QAE oracle queries: {oracle_queries}")
        print(f"Classical MC samples: {classical_samples}")
        print("=" * 65)

        for repetition in range(repetitions):

            current_seed = seed + repetition

            print(
                f"\nRepetition {repetition + 1}/{repetitions}",
                flush=True,
            )

            # ------------------------------------------
            # Classical Monte Carlo
            # ------------------------------------------

            mc = run_classical_mc(
                probabilities=probabilities,
                good_states=good_states,
                n_samples=classical_samples,
                seed=current_seed,
            )

            # ------------------------------------------
            # Quantum amplitude estimation
            # ------------------------------------------

            qae_start = time.perf_counter()

            observations = run_qae_experiment(
                probabilities=probabilities,
                good_states=good_states,
                iterations=iterations,
                shots=shots,
                seed=current_seed,
            )

            estimation = estimate_amplitude_mle(
                observations,
                upper_bound=0.10,
                grid_points=50001,
            )

            qae_time = time.perf_counter() - qae_start

            qae_estimate = estimation[
                "estimated_amplitude"
            ]

            qae_interval = estimation[
                "confidence_interval_95"
            ]

            # ------------------------------------------
            # Store results
            # ------------------------------------------

            records.append({
                "shots_per_setting": shots,
                "repetition": repetition + 1,
                "oracle_queries": oracle_queries,
                "classical_samples": classical_samples,

                "classical_estimate": mc["estimate"],
                "classical_se": mc["standard_error"],
                "classical_time": mc["execution_time"],

                "qae_estimate": qae_estimate,
                "qae_absolute_error": abs(
                    qae_estimate - classical_probability
                ),
                "qae_relative_error": (
                    abs(qae_estimate - classical_probability)
                    / classical_probability
                ),
                "qae_time": qae_time,

                "qae_ci_lower": qae_interval[0],
                "qae_ci_upper": qae_interval[1],

                "qae_ci_contains_truth": (
                    qae_interval[0]
                    <= classical_probability
                    <= qae_interval[1]
                ),

                "classical_absolute_error": abs(
                    mc["estimate"] - classical_probability
                ),
            })

            print(
                f"MC estimate:  {mc['estimate']:.8f}",
                flush=True,
            )

            print(
                f"QAE estimate: {qae_estimate:.8f}",
                flush=True,
            )

    return pd.DataFrame(records)


def summarize_benchmark(results):
    """
    Summarize accuracy, uncertainty and timing
    over independent repetitions.
    """

    summary = results.groupby(
        "shots_per_setting"
    ).agg(
        oracle_queries=("oracle_queries", "first"),
        classical_samples=("classical_samples", "first"),

        mc_mean=("classical_estimate", "mean"),
        mc_mae=("classical_absolute_error", "mean"),

        qae_mean=("qae_estimate", "mean"),
        qae_mae=("qae_absolute_error", "mean"),
        qae_mean_relative_error=("qae_relative_error", "mean"),

        qae_time_mean=("qae_time", "mean"),
        mc_time_mean=("classical_time", "mean"),

        qae_interval_coverage=("qae_ci_contains_truth", "mean"),
    )

    return summary.reset_index()

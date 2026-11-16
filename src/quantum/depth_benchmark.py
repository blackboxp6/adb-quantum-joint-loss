
import time
import numpy as np
import pandas as pd

from src.quantum.amplitude_estimation import (
    run_qae_experiment,
    estimate_amplitude_mle,
)


FIXED_DEPTHS = (0, 1, 2, 3, 4, 5, 6)

DEPTH_SCHEDULES = {
    2100: (0, 1, 2, 3, 4, 5, 6),
    6300: (0, 1, 2, 3, 4, 6, 8),
    21000: (0, 1, 2, 4, 6, 8, 12),
    63000: (0, 1, 2, 4, 8, 12, 16, 24),
}


def run_classical_mc(
    probabilities,
    good_states,
    n_samples,
    seed=42,
):
    """Classical Monte Carlo with a specified sample budget."""

    rng = np.random.default_rng(seed)

    start = time.perf_counter()

    sampled_states = rng.choice(
        len(probabilities),
        size=n_samples,
        p=probabilities,
    )

    estimate = good_states[sampled_states].mean()

    elapsed = time.perf_counter() - start

    standard_error = np.sqrt(
        estimate * (1 - estimate) / n_samples
    )

    return {
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "samples": int(n_samples),
        "execution_time": elapsed,
    }


def calculate_query_budget(iterations, shots):
    """
    Simplified Grover-oracle query count.

    Each shot at depth k is assigned k Grover iterations.
    """

    return int(shots * sum(iterations))


def run_one_qae_strategy(
    probabilities,
    good_states,
    iterations,
    query_budget,
    seed,
    upper_bound=0.10,
    grid_points=50001,
):
    """
    Allocate an equal number of shots to each depth,
    without exceeding the specified query budget.
    """

    total_depth = sum(iterations)

    if total_depth <= 0:
        raise ValueError("The schedule must contain positive depth.")

    shots = query_budget // total_depth

    if shots < 1:
        raise ValueError(
            "Query budget is too small for this depth schedule."
        )

    actual_queries = calculate_query_budget(
        iterations,
        shots,
    )

    start = time.perf_counter()

    observations = run_qae_experiment(
        probabilities=probabilities,
        good_states=good_states,
        iterations=iterations,
        shots=shots,
        seed=seed,
    )

    estimation = estimate_amplitude_mle(
        observations,
        upper_bound=upper_bound,
        grid_points=grid_points,
    )

    elapsed = time.perf_counter() - start

    estimate = estimation["estimated_amplitude"]

    return {
        "estimate": estimate,
        "intervals": estimation["all_accepted_intervals_95"],
        "iterations": tuple(iterations),
        "shots_per_depth": shots,
        "measurement_shots": shots * len(iterations),
        "oracle_queries": actual_queries,
        "unused_query_budget": query_budget - actual_queries,
        "execution_time": elapsed,
        "maximum_log_likelihood": (
            estimation["maximum_log_likelihood"]
        ),
    }


def interval_contains_truth(intervals, truth):
    """Check whether any accepted likelihood region contains truth."""

    return any(
        lower <= truth <= upper
        for lower, upper in intervals
    )


def run_depth_benchmark(
    probabilities,
    good_states,
    query_budgets=(2100, 6300, 21000, 63000),
    repetitions=50,
    seed=42,
    upper_bound=0.10,
    grid_points=50001,
):
    """
    Compare:
      1. Classical Monte Carlo
      2. Fixed-depth MLAE
      3. Depth-escalated MLAE

    Classical MC is matched to the actual Grover-query
    count used by each QAE strategy.

    This is an ideal statevector simulation benchmark.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    good_states = np.asarray(good_states, dtype=bool)

    truth = float(probabilities[good_states].sum())

    records = []

    for budget in query_budgets:

        if budget not in DEPTH_SCHEDULES:
            raise ValueError(
                f"No depth schedule defined for budget {budget}."
            )

        strategies = {
            "fixed_depth": FIXED_DEPTHS,
            "depth_escalated": DEPTH_SCHEDULES[budget],
        }

        print("\n" + "=" * 70)
        print(f"TARGET QUERY BUDGET: {budget:,}")
        print(f"True favorable probability: {truth:.10f}")
        print("=" * 70)

        for strategy_name, iterations in strategies.items():

            print(f"\nSTRATEGY: {strategy_name}")
            print(f"Grover depths: {iterations}")

            for repetition in range(repetitions):

                current_seed = seed + repetition

                print(
                    f"Repetition {repetition + 1}/{repetitions}",
                    flush=True,
                )

                qae = run_one_qae_strategy(
                    probabilities=probabilities,
                    good_states=good_states,
                    iterations=iterations,
                    query_budget=budget,
                    seed=current_seed,
                    upper_bound=upper_bound,
                    grid_points=grid_points,
                )

                # Match MC samples to the actual QAE query count.
                mc = run_classical_mc(
                    probabilities=probabilities,
                    good_states=good_states,
                    n_samples=qae["oracle_queries"],
                    seed=current_seed,
                )

                qae_error = qae["estimate"] - truth
                mc_error = mc["estimate"] - truth

                records.append({
                    "query_budget": budget,
                    "strategy": strategy_name,
                    "repetition": repetition + 1,

                    "grover_depths": str(iterations),
                    "maximum_grover_depth": max(iterations),
                    "shots_per_depth": qae["shots_per_depth"],
                    "measurement_shots": qae["measurement_shots"],

                    "oracle_queries": qae["oracle_queries"],
                    "unused_query_budget": qae["unused_query_budget"],
                    "classical_samples": mc["samples"],

                    "true_probability": truth,

                    "qae_estimate": qae["estimate"],
                    "qae_error": qae_error,
                    "qae_absolute_error": abs(qae_error),
                    "qae_squared_error": qae_error ** 2,
                    "qae_relative_error": (
                        abs(qae_error) / truth
                    ),

                    "qae_intervals": str(qae["intervals"]),
                    "qae_interval_contains_truth": (
                        interval_contains_truth(
                            qae["intervals"],
                            truth,
                        )
                    ),

                    "mc_estimate": mc["estimate"],
                    "mc_error": mc_error,
                    "mc_absolute_error": abs(mc_error),
                    "mc_squared_error": mc_error ** 2,

                    "qae_time": qae["execution_time"],
                    "mc_time": mc["execution_time"],
                })

                print(
                    f"QAE: {qae['estimate']:.8f} | "
                    f"MC: {mc['estimate']:.8f} | "
                    f"QAE intervals: {qae['intervals']}",
                    flush=True,
                )

    return pd.DataFrame(records)


def summarize_depth_benchmark(results):
    """Calculate bias, MAE, RMSE, coverage and timing."""

    summary = results.groupby(
        ["query_budget", "strategy"]
    ).agg(
        actual_oracle_queries=("oracle_queries", "mean"),
        maximum_grover_depth=("maximum_grover_depth", "first"),
        measurement_shots=("measurement_shots", "mean"),

        qae_bias=("qae_error", "mean"),
        qae_mae=("qae_absolute_error", "mean"),
        qae_rmse=(
            "qae_squared_error",
            lambda x: np.sqrt(x.mean()),
        ),

        mc_bias=("mc_error", "mean"),
        mc_mae=("mc_absolute_error", "mean"),
        mc_rmse=(
            "mc_squared_error",
            lambda x: np.sqrt(x.mean()),
        ),

        qae_interval_coverage=(
            "qae_interval_contains_truth",
            "mean",
        ),

        qae_time_mean=("qae_time", "mean"),
        mc_time_mean=("mc_time", "mean"),
    ).reset_index()

    summary["qae_mae_reduction_percent"] = (
        100
        * (summary["mc_mae"] - summary["qae_mae"])
        / summary["mc_mae"]
    )

    return summary


def save_depth_benchmark(results, summary):
    results.to_csv(
        "results/depth_benchmark_raw.csv",
        index=False,
    )

    summary.to_csv(
        "results/depth_benchmark_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print("results/depth_benchmark_raw.csv")
    print("results/depth_benchmark_summary.csv")

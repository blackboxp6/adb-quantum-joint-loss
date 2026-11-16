
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.classical_copula_runner import run_classical_copula
from src.quantum.loss_oracle import validate_loss_oracle
from src.quantum.benchmark import run_benchmark


# ==================================================
# CONFIGURATION
# ==================================================

SHOTS_PER_SETTING = (100, 300, 1000, 3000)
GROVER_ITERATIONS = (0, 1, 2, 3, 4, 5, 6)

REPETITIONS = 50
SEED = 2026

COPULA_FAMILY = "student_t"
COPULA_DF = 5

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 1. LOAD CLASSICAL PROBABILITY DISTRIBUTION
# ==================================================

print("[1] Loading Student-t distribution...", flush=True)

result = run_classical_copula(
    family=COPULA_FAMILY,
    df=COPULA_DF,
    include_distribution=True,
)

probabilities = np.asarray(
    result["term_distribution"],
    dtype=float,
).reshape(-1)

if not np.isclose(probabilities.sum(), 1.0):
    raise ValueError("Probability distribution is not normalized.")

if np.any(probabilities < 0):
    raise ValueError("Probability distribution contains negative values.")

print("Number of terminal states:", len(probabilities))
print("Probability sum:", probabilities.sum())


# ==================================================
# 2. CONSTRUCT LOSS ORACLE
# ==================================================

print("\n[2] Constructing loss oracle...", flush=True)

oracle = validate_loss_oracle(probabilities)

good_states = oracle["good_states"]

true_probability = float(
    oracle["classical_probability"]
)

print("Favorable states:", np.sum(good_states))
print("Exact exceedance probability:", true_probability)
print(
    "Exact exceedance percentage:",
    f"{true_probability * 100:.6f}%"
)


# ==================================================
# 3. RUN 50 INDEPENDENT REPETITIONS
# ==================================================

print("\n[3] Starting benchmark...", flush=True)

results = run_benchmark(
    probabilities=probabilities,
    good_states=good_states,
    shots_per_setting=SHOTS_PER_SETTING,
    repetitions=REPETITIONS,
    iterations=GROVER_ITERATIONS,
    seed=SEED,
)

# Calculate signed errors.

results["qae_error"] = (
    results["qae_estimate"] - true_probability
)

results["mc_error"] = (
    results["classical_estimate"] - true_probability
)


# ==================================================
# 4. SUMMARIZE STATISTICAL PERFORMANCE
# ==================================================

print("\n[4] Calculating statistical metrics...", flush=True)

summary = results.groupby("shots_per_setting").agg(

    oracle_queries=("oracle_queries", "first"),

    mc_bias=("mc_error", "mean"),

    mc_mae=("classical_absolute_error", "mean"),

    mc_rmse=(
        "mc_error",
        lambda x: np.sqrt(np.mean(x**2))
    ),

    qae_bias=("qae_error", "mean"),

    qae_mae=("qae_absolute_error", "mean"),

    qae_rmse=(
        "qae_error",
        lambda x: np.sqrt(np.mean(x**2))
    ),

    qae_interval_coverage=(
        "qae_ci_contains_truth",
        "mean"
    ),

    qae_time=("qae_time", "mean"),

    mc_time=("classical_time", "mean"),

).reset_index()


# Calculate relative improvement in MAE.

summary["mae_reduction_percent"] = (
    1 - summary["qae_mae"] / summary["mc_mae"]
) * 100


# ==================================================
# 5. EMPIRICAL CONVERGENCE-RATE ANALYSIS
# ==================================================

print("\n[5] Estimating convergence rates...", flush=True)

query_budget = summary["oracle_queries"].to_numpy(
    dtype=float
)

mc_mae = summary["mc_mae"].to_numpy(
    dtype=float
)

qae_mae = summary["qae_mae"].to_numpy(
    dtype=float
)

if (
    np.any(query_budget <= 0)
    or np.any(mc_mae <= 0)
    or np.any(qae_mae <= 0)
):
    raise ValueError(
        "Convergence analysis requires positive "
        "query budgets and MAE values."
    )

# Fit:
# log(MAE) = intercept + slope * log(query budget)

mc_slope, mc_intercept = np.polyfit(
    np.log(query_budget),
    np.log(mc_mae),
    1,
)

qae_slope, qae_intercept = np.polyfit(
    np.log(query_budget),
    np.log(qae_mae),
    1,
)

# Calculate fitted values and R-squared.

mc_log_predicted = (
    mc_intercept + mc_slope * np.log(query_budget)
)

qae_log_predicted = (
    qae_intercept + qae_slope * np.log(query_budget)
)

mc_r_squared = (
    1 - np.sum(
        (np.log(mc_mae) - mc_log_predicted) ** 2
    ) / np.sum(
        (np.log(mc_mae) - np.mean(np.log(mc_mae))) ** 2
    )
)

qae_r_squared = (
    1 - np.sum(
        (np.log(qae_mae) - qae_log_predicted) ** 2
    ) / np.sum(
        (np.log(qae_mae) - np.mean(np.log(qae_mae))) ** 2
    )
)

convergence = pd.DataFrame({

    "method": [
        "Classical Monte Carlo",
        "Quantum MLAE",
    ],

    "empirical_slope": [
        mc_slope,
        qae_slope,
    ],

    "log_log_r_squared": [
        mc_r_squared,
        qae_r_squared,
    ],

    "theoretical_reference_slope": [
        -0.5,
        -1.0,
    ],

})


# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 100)
print("CLASSICAL MC VS QUANTUM MLAE")
print("=" * 100)

print("\nSTATISTICAL PERFORMANCE")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

print("\n" + "=" * 100)
print("EMPIRICAL CONVERGENCE-RATE ANALYSIS")
print("=" * 100)

print(convergence.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nTHEORETICAL REFERENCES")
print("Classical Monte Carlo: O(1/sqrt(N)), slope = -0.5")
print("Ideal Quantum Amplitude Estimation: O(1/N), slope = -1.0")

print("\nNOTE:")
print(
    "Empirical slopes are descriptive estimates from four "
    "query budgets. They do not establish asymptotic "
    "quantum complexity."
)


# ==================================================
# 7. SAVE EXPERIMENT CONFIGURATION
# ==================================================

configuration = {

    "copula_family": COPULA_FAMILY,

    "degrees_of_freedom": COPULA_DF,

    "repetitions": REPETITIONS,

    "shots_per_setting": list(SHOTS_PER_SETTING),

    "grover_iterations": list(GROVER_ITERATIONS),

    "seed": SEED,

    "true_probability": true_probability,

    "favorable_state_count": int(np.sum(good_states)),

    "query_budget_definition": (
        "shots_per_setting multiplied by sum of Grover iterations"
    ),

}


# ==================================================
# 8. SAVE RESULTS
# ==================================================

print("\n[6] Saving results...", flush=True)

results.to_csv(
    RESULTS_DIR / "qae_mc_benchmark_50_raw.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR / "qae_mc_benchmark_50_summary.csv",
    index=False,
)

convergence.to_csv(
    RESULTS_DIR / "qae_mc_convergence_rates.csv",
    index=False,
)

with open(
    RESULTS_DIR / "qae_mc_benchmark_50_config.json",
    "w",
) as file:

    json.dump(
        configuration,
        file,
        indent=4,
    )


# ==================================================
# 9. COMPLETION
# ==================================================

print("\nResults saved successfully.")

print(" - qae_mc_benchmark_50_raw.csv")
print(" - qae_mc_benchmark_50_summary.csv")
print(" - qae_mc_convergence_rates.csv")
print(" - qae_mc_benchmark_50_config.json")

print("\nBENCHMARK COMPLETED")

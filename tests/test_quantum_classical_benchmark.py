
import numpy as np

from src.classical_copula_runner import run_classical_copula

from src.quantum.loss_oracle import validate_loss_oracle

from src.quantum.benchmark import (
    run_benchmark,
    summarize_benchmark,
)


# --------------------------------------------------
# 1. Load validated Student-t distribution
# --------------------------------------------------

print("\n[1] Loading classical distribution...", flush=True)

result = run_classical_copula(
    family="student_t",
    df=5,
    include_distribution=True,
)

probabilities = np.asarray(
    result["term_distribution"]
).reshape(-1)


# --------------------------------------------------
# 2. Construct favorable-state mask
# --------------------------------------------------

print("\n[2] Constructing loss oracle...", flush=True)

oracle = validate_loss_oracle(probabilities)

good_states = oracle["good_states"]

benchmark_probability = oracle[
    "classical_probability"
]

print(
    "Exact benchmark probability:",
    benchmark_probability,
)


# --------------------------------------------------
# 3. Run comparison
# --------------------------------------------------

print("\n[3] Starting benchmark...", flush=True)

results = run_benchmark(
    probabilities=probabilities,
    good_states=good_states,
    shots_per_setting=(100, 300, 1000, 3000),
    repetitions=5,
    seed=42,
)


# --------------------------------------------------
# 4. Generate summary
# --------------------------------------------------

summary = summarize_benchmark(results)

print("\n" + "=" * 80)
print("CLASSICAL MONTE CARLO VS QUANTUM MLAE")
print("=" * 80)

print(summary.to_string(index=False))


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

results.to_csv(
    "results/qae_mc_benchmark_raw.csv",
    index=False,
)

summary.to_csv(
    "results/qae_mc_benchmark_summary.csv",
    index=False,
)

print("\nResults saved:")
print("results/qae_mc_benchmark_raw.csv")
print("results/qae_mc_benchmark_summary.csv")

print("\nBENCHMARK COMPLETED")

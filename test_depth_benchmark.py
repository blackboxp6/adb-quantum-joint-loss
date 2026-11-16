
from src.classical_copula_runner import run_classical_copula
from src.quantum.loss_oracle import validate_loss_oracle

from src.quantum.depth_benchmark import (
    run_depth_benchmark,
    summarize_depth_benchmark,
    save_depth_benchmark,
)


# --------------------------------------------------
# 1. Load the Student-t probability distribution
# --------------------------------------------------

print("[1] Loading Student-t distribution...", flush=True)

result = run_classical_copula(
    family="student_t",
    df=5,
    include_distribution=True,
)

probabilities = result["term_distribution"].reshape(-1)

# --------------------------------------------------
# 2. Construct the favorable-state oracle
# --------------------------------------------------

print("\n[2] Constructing loss oracle...", flush=True)

oracle_result = validate_loss_oracle(probabilities)

print("Oracle result type:", type(oracle_result))

if isinstance(oracle_result, dict):
    good_states = oracle_result["good_states"]
    classical_probability = oracle_result["classical_probability"]

else:
    good_states = oracle_result[0]
    classical_probability = oracle_result[1]

print(
    f"Reference probability: {classical_probability:.10f}",
    flush=True,
)

# --------------------------------------------------
# 3. Run depth benchmark
# --------------------------------------------------

print("\n[3] Running depth benchmark...", flush=True)

results = run_depth_benchmark(
    probabilities=probabilities,
    good_states=good_states,
    query_budgets=(2100, 6300, 21000, 63000),
    repetitions=50,
    seed=42,
    upper_bound=0.10,
    grid_points=50001,
)

# --------------------------------------------------
# 4. Summarize results
# --------------------------------------------------

print("\n[4] Summarizing results...", flush=True)

summary = summarize_depth_benchmark(results)

print("\nDEPTH BENCHMARK RESULTS")
print(summary.to_string(index=False))

# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

save_depth_benchmark(results, summary)

print("\nBenchmark completed.", flush=True)

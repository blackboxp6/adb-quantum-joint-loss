
import numpy as np
import pandas as pd
import traceback
import time

print("\n[START] Quantum Amplitude Estimation experiment", flush=True)

try:
    # --------------------------------------------------
    # 1. Import modules
    # --------------------------------------------------

    print("\n[1/5] Importing project modules...", flush=True)

    from src.classical_copula_runner import run_classical_copula

    from src.quantum.loss_oracle import validate_loss_oracle

    from src.quantum.amplitude_estimation import (
        run_qae_experiment,
        estimate_amplitude_mle,
    )

    print("[SUCCESS] All modules imported.", flush=True)

    # --------------------------------------------------
    # 2. Load classical probability distribution
    # --------------------------------------------------

    print("\n[2/5] Loading Student-t copula distribution...", flush=True)

    result = run_classical_copula(
        family="student_t",
        df=5,
        include_distribution=True,
    )

    probabilities = np.asarray(
        result["term_distribution"]
    ).reshape(-1)

    print("Number of states:", len(probabilities))
    print("Probability sum:", probabilities.sum())
    print("[SUCCESS] Classical distribution loaded.", flush=True)

    # --------------------------------------------------
    # 3. Construct favorable-state oracle
    # --------------------------------------------------

    print("\n[3/5] Constructing loss oracle...", flush=True)

    oracle_result = validate_loss_oracle(probabilities)

    good_states = oracle_result["good_states"]

    classical_probability = oracle_result[
        "classical_probability"
    ]

    print("Number of favorable states:", good_states.sum())
    print(
        "Classical exceedance probability:",
        classical_probability
    )

    print("[SUCCESS] Loss oracle constructed.", flush=True)

    # --------------------------------------------------
    # 4. Run quantum measurements
    # --------------------------------------------------

    print("\n[4/5] Running QAE quantum simulations...", flush=True)

    start_time = time.time()

    observations = run_qae_experiment(
    probabilities=probabilities,
    good_states=good_states,
    iterations=(0, 1, 2, 3, 4, 5, 6),
    shots=3000,
    seed=42,
    )

    elapsed_time = time.time() - start_time

    print("[SUCCESS] Quantum simulations completed.", flush=True)
    print(f"Simulation time: {elapsed_time:.2f} seconds")

    # --------------------------------------------------
    # 5. Estimate original amplitude using MLE
    # --------------------------------------------------

    print("\n[5/5] Estimating amplitude using MLE...", flush=True)

    estimation = estimate_amplitude_mle(
        observations,
        upper_bound=0.10,
        grid_points=100001,
    )

    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("QUANTUM AMPLITUDE ESTIMATION RESULTS")
    print("=" * 70)

    print("\nMeasurement results:")

    print(
        pd.DataFrame(observations).to_string(index=False)
    )

    print("\nAmplitude estimation:")

    print(
        f"Classical benchmark: "
        f"{classical_probability:.10f}"
    )

    print(
        f"QAE estimate: "
        f"{estimation['estimated_amplitude']:.10f}"
    )

    print(
        f"95% profile-likelihood interval: "
        f"{estimation['confidence_interval_95']}"
    )

    print(
        f"Absolute estimation error: "
        f"{abs(estimation['estimated_amplitude'] - classical_probability):.10f}"
    )

    print(
        f"Grid resolution: "
        f"{estimation['grid_resolution']:.10f}"
    )

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

    print("\n[COMPLETE] QAE EXPERIMENT FINISHED SUCCESSFULLY.")

except Exception:
    print("\n[ERROR] The QAE experiment failed.")
    traceback.print_exc()

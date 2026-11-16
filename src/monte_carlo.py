
import numpy as np
import pandas as pd

from src.earthquake import (
    calculate_all_country_rates,
    calculate_annual_earthquake_distribution,
)

from src.precipitation import (
    calculate_precipitation_joint_distribution,
)

from src.copulas import (
    discrete_copula_joint,
    calibrate_copula_rho,
)

from src.joint_distribution import (
    build_annual_joint_distribution,
)

from src.classical_copula_runner import get_adb_marginals


def build_adb_annual_distribution():

    country_rates = calculate_all_country_rates()

    earthquake_joint, _ = calculate_annual_earthquake_distribution(
        country_rates,
        shared_multiplier=1.0
    )

    earthquake_joint = np.asarray(earthquake_joint).reshape(5, 5)

    precipitation = calculate_precipitation_joint_distribution(
        q_kyrgyz=0.02,
        q_tajikistan=0.02,
        rate_multiplier=1.0,
        shared_fraction=0.25
    )

    return build_annual_joint_distribution(
        earthquake_joint,
        precipitation["probabilities"]
    )


def build_copula_annual_distribution(
    family="gaussian",
    df=5
):

    marginals = get_adb_marginals()

    rho_earthquake = calibrate_copula_rho(
        marginals["earthquake_x"],
        marginals["earthquake_y"],
        marginals["earthquake_target"],
        family=family,
        df=df
    )

    rho_precipitation = calibrate_copula_rho(
        marginals["precipitation_x"],
        marginals["precipitation_y"],
        marginals["precipitation_target"],
        family=family,
        df=df
    )

    earthquake_joint = discrete_copula_joint(
        marginals["earthquake_x"],
        marginals["earthquake_y"],
        rho_earthquake,
        family=family,
        df=df
    )

    precipitation_joint = discrete_copula_joint(
        marginals["precipitation_x"],
        marginals["precipitation_y"],
        rho_precipitation,
        family=family,
        df=df
    )

    precipitation_probabilities = np.array([
        precipitation_joint[0, 0],
        precipitation_joint[1, 0],
        precipitation_joint[0, 1],
        precipitation_joint[1, 1]
    ])

    return build_annual_joint_distribution(
        earthquake_joint,
        precipitation_probabilities
    )


def simulate_portfolio_losses(
    annual_distribution,
    n_simulations,
    rng,
    batch_size=250_000
):
    """
    Simulate three-year portfolio losses.

    Annual state: (a, b, r1, r2)
    a, b: earthquake payout states, 0 to 4
    r1, r2: precipitation payout indicators, 0 or 1

    Losses are returned in USD millions.
    """

    annual_distribution = np.asarray(
        annual_distribution,
        dtype=float
    )

    if annual_distribution.shape != (5, 5, 2, 2):
        raise ValueError(
            "Annual distribution must have shape (5, 5, 2, 2)."
        )

    probabilities = annual_distribution.ravel()

    if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
        raise ValueError("Annual probabilities must sum to 1.")

    if np.any(probabilities < 0):
        raise ValueError("Annual probabilities cannot be negative.")

    n_states = probabilities.size

    losses = np.empty(n_simulations, dtype=float)

    completed = 0

    while completed < n_simulations:

        current_batch = min(
            batch_size,
            n_simulations - completed
        )

        # Initialize accumulated three-year states
        a = np.zeros(current_batch, dtype=np.int8)
        b = np.zeros(current_batch, dtype=np.int8)
        r1 = np.zeros(current_batch, dtype=np.int8)
        r2 = np.zeros(current_batch, dtype=np.int8)

        # Simulate three independent annual draws
        for _ in range(3):

            sampled_indices = rng.choice(
                n_states,
                size=current_batch,
                p=probabilities
            )

            sampled_states = np.array(
                np.unravel_index(
                    sampled_indices,
                    (5, 5, 2, 2)
                )
            )

            annual_a = sampled_states[0]
            annual_b = sampled_states[1]
            annual_r1 = sampled_states[2]
            annual_r2 = sampled_states[3]

            # Accumulate and cap earthquake states at 4
            a = np.minimum(a + annual_a, 4)
            b = np.minimum(b + annual_b, 4)

            # Precipitation pays at most once per term
            r1 = np.minimum(r1 + annual_r1, 1)
            r2 = np.minimum(r2 + annual_r2, 1)

        # Calculate final country-level losses
        loss_kyrgyz = 70.0 * a / 4.0 + 10.0 * r1
        loss_tajikistan = 70.0 * b / 4.0 + 10.0 * r2

        losses[completed:completed + current_batch] = (
            loss_kyrgyz + loss_tajikistan
        )

        completed += current_batch

    return losses


def calculate_mc_metrics(losses):

    losses = np.asarray(losses, dtype=float)

    n = len(losses)

    if n < 2:
        raise ValueError("At least two simulations are required.")

    exceedance_indicator = (losses > 80.0).astype(float)
    stop_losses = np.maximum(losses - 80.0, 0.0)

    p_hat = exceedance_indicator.mean()
    expected_loss = losses.mean()
    stop_loss = stop_losses.mean()

    p_se = np.sqrt(p_hat * (1.0 - p_hat) / n)

    expected_loss_se = losses.std(ddof=1) / np.sqrt(n)
    stop_loss_se = stop_losses.std(ddof=1) / np.sqrt(n)

    return {
        "simulations": n,

        "expected_loss_millions": expected_loss,
        "expected_loss_se": expected_loss_se,

        "exceedance_probability": p_hat,
        "exceedance_se": p_se,
        "exceedance_ci_lower": max(0.0, p_hat - 1.96 * p_se),
        "exceedance_ci_upper": min(1.0, p_hat + 1.96 * p_se),

        "stop_loss_millions": stop_loss,
        "stop_loss_se": stop_loss_se,
    }


def run_monte_carlo_comparison(
    sample_sizes=(10_000, 100_000, 1_000_000),
    seed=42,
    batch_size=250_000
):

    sample_sizes = sorted(set(sample_sizes))

    if not sample_sizes or sample_sizes[0] < 2:
        raise ValueError("Sample sizes must be at least 2.")

    max_simulations = max(sample_sizes)

    models = [
        ("ADB benchmark", build_adb_annual_distribution()),
        ("Gaussian", build_copula_annual_distribution("gaussian")),
        ("Student-t df=3", build_copula_annual_distribution("student_t", 3)),
        ("Student-t df=5", build_copula_annual_distribution("student_t", 5)),
        ("Student-t df=10", build_copula_annual_distribution("student_t", 10)),
        ("Student-t df=30", build_copula_annual_distribution("student_t", 30)),
    ]

    results = []

    for model_index, (model_name, annual_distribution) in enumerate(models):

        print(f"\nSimulating: {model_name}")

        rng = np.random.default_rng(seed + model_index)

        losses = simulate_portfolio_losses(
            annual_distribution=annual_distribution,
            n_simulations=max_simulations,
            rng=rng,
            batch_size=batch_size
        )

        for n in sample_sizes:

            metrics = calculate_mc_metrics(losses[:n])

            metrics["model"] = model_name

            results.append(metrics)

            print(
                f"N={n:>10,} | "
                f"EL=${metrics['expected_loss_millions']:.4f}M | "
                f"P(L>80M)={metrics['exceedance_probability']:.6%} | "
                f"Stop-loss=${metrics['stop_loss_millions']:.4f}M"
            )

    return pd.DataFrame(results)


def save_monte_carlo_results(
    results,
    output_path="results/monte_carlo_results.csv"
):

    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_path, index=False)

    print(f"\nMonte Carlo results saved to: {output_path.resolve()}")

    return output_path

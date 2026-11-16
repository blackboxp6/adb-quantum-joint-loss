
import numpy as np
import pandas as pd

from pathlib import Path

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

from src.term_distribution import (
    calculate_three_year_distribution,
)

from src.loss_metrics import (
    calculate_loss_metrics,
)


def get_adb_marginals():

    # Original ADB earthquake model
    country_rates = calculate_all_country_rates()

    earthquake_joint, generator = (
        calculate_annual_earthquake_distribution(
            country_rates,
            shared_multiplier=1.0
        )
    )

    earthquake_joint = np.asarray(
        earthquake_joint
    ).reshape(5, 5)

    earthquake_x = earthquake_joint.sum(axis=1)
    earthquake_y = earthquake_joint.sum(axis=0)

    earthquake_target = earthquake_joint[1:, 1:].sum()

    # Original ADB precipitation model
    precipitation = calculate_precipitation_joint_distribution(
        q_kyrgyz=0.02,
        q_tajikistan=0.02,
        rate_multiplier=1.0,
        shared_fraction=0.25
    )

    p = precipitation["probabilities"]

    precipitation_joint = np.array([
        [p[0], p[2]],
        [p[1], p[3]]
    ])

    precipitation_x = precipitation_joint.sum(axis=1)
    precipitation_y = precipitation_joint.sum(axis=0)

    precipitation_target = precipitation_joint[1, 1]

    return {
        "earthquake_x": earthquake_x,
        "earthquake_y": earthquake_y,
        "earthquake_target": earthquake_target,
        "precipitation_x": precipitation_x,
        "precipitation_y": precipitation_y,
        "precipitation_target": precipitation_target,
    }


def get_adb_base_term_distribution():

    # Original ADB earthquake dependence
    country_rates = calculate_all_country_rates()

    earthquake_joint, _ = calculate_annual_earthquake_distribution(
        country_rates,
        shared_multiplier=1.0
    )

    earthquake_joint = np.asarray(
        earthquake_joint
    ).reshape(5, 5)

    # Original ADB precipitation dependence
    precipitation = calculate_precipitation_joint_distribution(
        q_kyrgyz=0.02,
        q_tajikistan=0.02,
        rate_multiplier=1.0,
        shared_fraction=0.25
    )

    precipitation_probabilities = precipitation["probabilities"]

    # Original ADB ordering: P00, P10, P01, P11
    annual_joint = build_annual_joint_distribution(
        earthquake_joint,
        precipitation_probabilities
    )

    return calculate_three_year_distribution(annual_joint)


def run_classical_copula(
    family="gaussian",
    df=5,
    include_distribution=False
):

    marginals = get_adb_marginals()

    # 1. Calibrate earthquake dependence
    rho_earthquake = calibrate_copula_rho(
        marginals["earthquake_x"],
        marginals["earthquake_y"],
        marginals["earthquake_target"],
        family=family,
        df=df
    )

    # 2. Calibrate precipitation dependence
    rho_precipitation = calibrate_copula_rho(
        marginals["precipitation_x"],
        marginals["precipitation_y"],
        marginals["precipitation_target"],
        family=family,
        df=df
    )

    # 3. Construct earthquake joint distribution
    earthquake_joint = discrete_copula_joint(
        marginals["earthquake_x"],
        marginals["earthquake_y"],
        rho_earthquake,
        family=family,
        df=df
    )

    # 4. Construct precipitation joint distribution
    precipitation_joint = discrete_copula_joint(
        marginals["precipitation_x"],
        marginals["precipitation_y"],
        rho_precipitation,
        family=family,
        df=df
    )

    # Convert precipitation matrix into ADB ordering:
    # P00, P10, P01, P11
    precipitation_probabilities = np.array([
        precipitation_joint[0, 0],
        precipitation_joint[1, 0],
        precipitation_joint[0, 1],
        precipitation_joint[1, 1]
    ])

    # 5. Combine the two hazard distributions
    annual_joint = build_annual_joint_distribution(
        earthquake_joint,
        precipitation_probabilities
    )

    # 6. Accumulate losses over three years
    term_distribution = calculate_three_year_distribution(
        annual_joint
    )

    # 7. Calculate portfolio risk metrics
    metrics = calculate_loss_metrics(
        term_distribution,
        earthquake_limit=70.0,
        precipitation_limit=10.0,
        portfolio_attachment=80.0
    )

    result = {
        "family": family,
        "degrees_of_freedom": (
            df if family == "student_t" else np.nan
        ),
        "rho_earthquake": rho_earthquake,
        "rho_precipitation": rho_precipitation,
        **metrics
    }

    # Include the full distribution only when requested
    if include_distribution:
        result["term_distribution"] = term_distribution

    return result


def run_classical_comparison():

    results = []

    # Gaussian copula
    results.append(
        run_classical_copula(
            family="gaussian"
        )
    )

    # Student-t copulas with different tail behaviour
    for df in [3, 5, 10, 30]:

        results.append(
            run_classical_copula(
                family="student_t",
                df=df
            )
        )

    return pd.DataFrame(results)


def calculate_loss_exceedance_curve(
    term_distribution,
    thresholds
):
    """
    Calculate P(L > x) for a three-year term distribution.

    Losses and thresholds are expressed in USD millions.
    """

    term_distribution = np.asarray(term_distribution)

    if term_distribution.shape != (5, 5, 2, 2):
        raise ValueError(
            "Expected term distribution with shape (5, 5, 2, 2). "
            f"Received {term_distribution.shape}."
        )

    # Generate the four state-coordinate arrays
    a, b, r1, r2 = np.indices(
        term_distribution.shape
    )

    # Country-level losses, in USD millions
    loss_kyrgyz = 70.0 * a / 4.0 + 10.0 * r1
    loss_tajikistan = 70.0 * b / 4.0 + 10.0 * r2

    # Combined portfolio loss
    portfolio_loss = loss_kyrgyz + loss_tajikistan

    probabilities = term_distribution.ravel()
    losses = portfolio_loss.ravel()

    rows = []

    for threshold in thresholds:

        exceedance_probability = probabilities[
            losses > threshold
        ].sum()

        rows.append({
            "threshold_millions": threshold,
            "exceedance_probability": exceedance_probability
        })

    return pd.DataFrame(rows)


def run_loss_exceedance_comparison(
    thresholds=(20, 40, 60, 80, 100, 120, 140, 160)
):
    """
    Compare portfolio loss exceedance curves for:
    - Original ADB benchmark
    - Gaussian copula
    - Student-t copulas (df = 3, 5, 10, 30)
    """

    results = []

    # Original ADB benchmark
    adb_distribution = get_adb_base_term_distribution()

    adb_curve = calculate_loss_exceedance_curve(
        adb_distribution,
        thresholds
    )

    adb_curve["model"] = "ADB benchmark"
    adb_curve["family"] = "ADB"
    adb_curve["degrees_of_freedom"] = np.nan

    results.append(adb_curve)

    # Gaussian copula
    model_specs = [
        ("gaussian", np.nan),
        ("student_t", 3),
        ("student_t", 5),
        ("student_t", 10),
        ("student_t", 30),
    ]

    for family, df in model_specs:

        model_result = run_classical_copula(
            family=family,
            df=5 if np.isnan(df) else df,
            include_distribution=True
        )

        model_curve = calculate_loss_exceedance_curve(
            model_result["term_distribution"],
            thresholds
        )

        model_name = (
            "Gaussian"
            if family == "gaussian"
            else f"Student-t (df={df})"
        )

        model_curve["model"] = model_name
        model_curve["family"] = family
        model_curve["degrees_of_freedom"] = df

        results.append(model_curve)

    return pd.concat(
        results,
        ignore_index=True
    )


def save_classical_results(
    results,
    output_path="results/classical_copula_results.csv"
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(output_path, index=False)

    print(f"Results successfully saved to: {output_path.resolve()}")

    return output_path


def save_loss_exceedance_results(
    results,
    output_path="results/loss_exceedance_curves.csv"
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(output_path, index=False)

    print(f"Loss exceedance results saved to: {output_path.resolve()}")

    return output_path

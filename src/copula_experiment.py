
import numpy as np

from src.benchmark_inputs import BASE_PARAMETERS

from src.earthquake import (
    calculate_all_country_rates,
    calculate_annual_earthquake_distribution,
)

from src.precipitation import (
    calculate_precipitation_joint_distribution,
)

from src.copulas import (
    discrete_gaussian_copula_joint,
    calibrate_gaussian_rho,
)

from src.joint_distribution import (
    build_annual_joint_distribution,
)

from src.term_distribution import (
    calculate_three_year_distribution,
)

from src.loss_metrics import calculate_loss_metrics


def run_gaussian_copula_experiment():

    # --------------------------------------------------
    # 1. Generate original ADB earthquake distribution
    # --------------------------------------------------

    country_rates = calculate_all_country_rates()

    earthquake_adb, _ = calculate_annual_earthquake_distribution(
        country_rates,
        shared_multiplier=1.0,
    )

    earthquake_adb = np.asarray(earthquake_adb).reshape(5, 5)

    # Extract the original marginal distributions
    marginal_kyrgyz_eq = earthquake_adb.sum(axis=1)
    marginal_tajik_eq = earthquake_adb.sum(axis=0)

    # ADB target: probability both countries experience payout
    target_earthquake_joint = earthquake_adb[1:, 1:].sum()

    # --------------------------------------------------
    # 2. Calibrate earthquake Gaussian copula
    # --------------------------------------------------

    rho_earthquake = calibrate_gaussian_rho(
        marginal_kyrgyz_eq,
        marginal_tajik_eq,
        target_earthquake_joint,
    )

    earthquake_copula = discrete_gaussian_copula_joint(
        marginal_kyrgyz_eq,
        marginal_tajik_eq,
        rho_earthquake,
    )

    # --------------------------------------------------
    # 3. Generate original ADB precipitation distribution
    # --------------------------------------------------

    precipitation_adb = calculate_precipitation_joint_distribution(
        q_kyrgyz=BASE_PARAMETERS["precipitation_probability"],
        q_tajikistan=BASE_PARAMETERS["precipitation_probability"],
        shared_fraction=BASE_PARAMETERS["precipitation_shared_fraction"],
    )

    p = precipitation_adb["probabilities"]

    # Matrix axes: Kyrgyz trigger, Tajik trigger
    precipitation_matrix_adb = np.array([
        [p[0], p[2]],
        [p[1], p[3]],
    ])

    marginal_kyrgyz_rain = precipitation_matrix_adb.sum(axis=1)
    marginal_tajik_rain = precipitation_matrix_adb.sum(axis=0)

    target_precipitation_joint = precipitation_matrix_adb[1, 1]

    # --------------------------------------------------
    # 4. Calibrate precipitation Gaussian copula
    # --------------------------------------------------

    rho_precipitation = calibrate_gaussian_rho(
        marginal_kyrgyz_rain,
        marginal_tajik_rain,
        target_precipitation_joint,
    )

    precipitation_copula = discrete_gaussian_copula_joint(
        marginal_kyrgyz_rain,
        marginal_tajik_rain,
        rho_precipitation,
    )

    # Convert to the ordering expected by the existing model:
    # P00, P10, P01, P11
    precipitation_probabilities = np.array([
        precipitation_copula[0, 0],
        precipitation_copula[1, 0],
        precipitation_copula[0, 1],
        precipitation_copula[1, 1],
    ])

    # --------------------------------------------------
    # 5. Combine the copula distributions
    # --------------------------------------------------

    annual_joint = build_annual_joint_distribution(
        earthquake_copula,
        precipitation_probabilities,
    )

    # --------------------------------------------------
    # 6. Calculate three-year risk metrics
    # --------------------------------------------------

    term_distribution = calculate_three_year_distribution(
        annual_joint
    )

    metrics = calculate_loss_metrics(
        term_distribution,
        earthquake_limit=BASE_PARAMETERS["earthquake_limit"],
        precipitation_limit=BASE_PARAMETERS["precipitation_limit"],
        portfolio_attachment=BASE_PARAMETERS["portfolio_attachment"],
    )

    return {
        "rho_earthquake": rho_earthquake,
        "rho_precipitation": rho_precipitation,
        "earthquake_joint": earthquake_copula,
        "precipitation_joint": precipitation_copula,
        "annual_joint": annual_joint,
        "term_distribution": term_distribution,
        "metrics": metrics,
    }


if __name__ == "__main__":

    results = run_gaussian_copula_experiment()

    print("\nGAUSSIAN COPULA CALIBRATION")
    print("=" * 60)

    print(
        f"Earthquake Gaussian rho: "
        f"{results['rho_earthquake']:.8f}"
    )

    print(
        f"Precipitation Gaussian rho: "
        f"{results['rho_precipitation']:.8f}"
    )

    print("\nGAUSSIAN COPULA RISK RESULTS")
    print("=" * 60)

    metrics = results["metrics"]

    print(
        f"Expected principal loss: "
        f"${metrics['expected_loss_millions']:.6f} million"
    )

    print(
        f"Joint attachment probability: "
        f"{metrics['joint_attachment_probability']:.8%}"
    )

    print(
        f"Portfolio loss exceedance probability: "
        f"{metrics['loss_exceedance_probability']:.8%}"
    )

    print(
        f"Portfolio stop-loss: "
        f"${metrics['stop_loss_usd']:,.2f}"
    )

    print(
        f"Both bonds exhausted probability: "
        f"{metrics['both_exhausted_probability']:.8%}"
    )

from src.benchmark_inputs import BASE_PARAMETERS, SCENARIOS

from src.earthquake import (
    calculate_all_country_rates,
    calculate_annual_earthquake_distribution,
)

from src.precipitation import (
    calculate_precipitation_joint_distribution,
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


def run_scenario(scenario_name):

    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    parameters = BASE_PARAMETERS.copy()
    parameters.update(SCENARIOS[scenario_name])

    # 1. Earthquake model
    country_rates = calculate_all_country_rates()

    earthquake_joint, generator = calculate_annual_earthquake_distribution(
    country_rates,
    shared_multiplier=parameters["earthquake_shared_multiplier"],
)

# Convert flattened 25-state vector into a 5 x 5 matrix
    earthquake_joint = earthquake_joint.reshape(5, 5)

# 2. Precipitation model
    precipitation = calculate_precipitation_joint_distribution(
    q_kyrgyz=parameters["precipitation_probability"],
    q_tajikistan=parameters["precipitation_probability"],
    rate_multiplier=parameters["precipitation_rate_multiplier"],
    shared_fraction=parameters["precipitation_shared_fraction"],
)   

    # 3. Combine earthquake and precipitation
    annual_joint = build_annual_joint_distribution(
        earthquake_joint,
        precipitation["probabilities"],
    )

    # 4. Three-year capped accumulation
    term_distribution = calculate_three_year_distribution(annual_joint)

    # 5. Calculate risk metrics
    metrics = calculate_loss_metrics(
        term_distribution,
        earthquake_limit=parameters["earthquake_limit"],
        precipitation_limit=parameters["precipitation_limit"],
        portfolio_attachment=parameters["portfolio_attachment"],
    )

    return metrics


def run_all_scenarios():

    results = {}

    for scenario_name in SCENARIOS:
        results[scenario_name] = run_scenario(scenario_name)

    return results
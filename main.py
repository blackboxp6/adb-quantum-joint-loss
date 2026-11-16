
import numpy as np
import pandas as pd

from scipy.linalg import expm

from src.term_distribution import calculate_three_year_distribution
from src.loss_metrics import calculate_loss_metrics
from src.precipitation import calculate_precipitation_joint_distribution
from src.joint_distribution import build_annual_joint_distribution
from src.scenario_runner import run_all_scenarios
from src.classical_copula_runner import (
    run_loss_exceedance_comparison,
    save_loss_exceedance_results,
)
from src.earthquake import (
    calculate_all_country_rates,
    calculate_annual_earthquake_distribution,
)

from src.classical_copula_runner import (
    run_classical_comparison,
    save_classical_results,
)
from src.monte_carlo import (
    run_monte_carlo_comparison,
    save_monte_carlo_results,
)

from src.copulas import calculate_tail_dependence


# ============================================================
# 1. ADB EARTHQUAKE MODEL
# ============================================================

rates = calculate_all_country_rates()

print("\nADB EARTHQUAKE MODEL")
print("=" * 60)

for country, result in rates.items():

    print(f"\nCountry: {country}")
    print(f"Total rate: {result['total_rate']:.8f}")

    for k, rate in enumerate(result["event_rates"], start=1):
        print(f"Lambda {k}: {rate:.8f}")


distribution, G = calculate_annual_earthquake_distribution(
    rates,
    shared_multiplier=1.0,
)

earthquake_joint = distribution.reshape(5, 5)

print("\nANNUAL JOINT EARTHQUAKE DISTRIBUTION")
print("=" * 60)

for a in range(5):
    for b in range(5):
        print(f"State ({a},{b}): {earthquake_joint[a,b]:.8f}")

print(f"\nProbability sum: {distribution.sum():.12f}")
print(f"Probability of no earthquake payout: {distribution[0]:.8f}")


# ============================================================
# 2. EARTHQUAKE MARGINAL DISTRIBUTIONS
# ============================================================

marginal_kyrgyz = earthquake_joint.sum(axis=1)
marginal_tajik = earthquake_joint.sum(axis=0)

print("\nMARGINAL EARTHQUAKE DISTRIBUTIONS")
print("=" * 55)

print("\nKyrgyz Republic:")
for category, probability in enumerate(marginal_kyrgyz):
    print(f"Category {category}: {probability:.8f}")

print("\nTajikistan:")
for category, probability in enumerate(marginal_tajik):
    print(f"Category {category}: {probability:.8f}")


# ============================================================
# 3. JOINT DEPENDENCE DIAGNOSTICS
# ============================================================

joint_attachment = earthquake_joint[1:, 1:].sum()

p_kyrgyz = marginal_kyrgyz[1:].sum()
p_tajik = marginal_tajik[1:].sum()

independent_joint = p_kyrgyz * p_tajik
dependence_ratio = joint_attachment / independent_joint

print("\nJOINT DEPENDENCE DIAGNOSTICS")
print("=" * 55)

print(f"P(Kyrgyz payout):       {p_kyrgyz:.8%}")
print(f"P(Tajikistan payout):   {p_tajik:.8%}")
print(f"P(Both payouts):        {joint_attachment:.8%}")
print(f"Independent benchmark:  {independent_joint:.8%}")
print(f"Dependence ratio:       {dependence_ratio:.4f}")


# ============================================================
# 4. THREE-YEAR EARTHQUAKE MODEL
# ============================================================

three_year_transition = expm(3.0 * G)

three_year_distribution = three_year_transition[0, :]
three_year_joint = three_year_distribution.reshape(5, 5)

quarter_value = 70.0 / 4.0

expected_kyrgyz_loss = quarter_value * sum(
    a * three_year_joint[a, b]
    for a in range(5)
    for b in range(5)
)

expected_tajik_loss = quarter_value * sum(
    b * three_year_joint[a, b]
    for a in range(5)
    for b in range(5)
)

print("\nTHREE-YEAR EARTHQUAKE RESULTS")
print("=" * 55)

print(f"Probability sum: {three_year_distribution.sum():.12f}")

print(
    f"Expected Kyrgyz earthquake loss: "
    f"${expected_kyrgyz_loss:.6f} million"
)

print(
    f"Expected Tajikistan earthquake loss: "
    f"${expected_tajik_loss:.6f} million"
)

print(
    f"Expected combined earthquake loss: "
    f"${expected_kyrgyz_loss + expected_tajik_loss:.6f} million"
)


# ============================================================
# 5. PRECIPITATION MODEL
# ============================================================

precipitation_result = calculate_precipitation_joint_distribution(
    q_kyrgyz=0.02,
    q_tajikistan=0.02,
    rate_multiplier=1.0,
    shared_fraction=0.25,
)

precipitation_probabilities = precipitation_result["probabilities"]

print("\nANNUAL PRECIPITATION DISTRIBUTION")
print("=" * 55)

print(f"Kyrgyz Poisson rate: {precipitation_result['mu_kyrgyz']:.8f}")
print(f"Tajikistan Poisson rate: {precipitation_result['mu_tajikistan']:.8f}")
print(f"Shared precipitation rate: {precipitation_result['shared_rate']:.8f}")

print(f"\nP(neither):           {precipitation_probabilities[0]:.8%}")
print(f"P(Kyrgyz only):       {precipitation_probabilities[1]:.8%}")
print(f"P(Tajikistan only):   {precipitation_probabilities[2]:.8%}")
print(f"P(both):              {precipitation_probabilities[3]:.8%}")


# ============================================================
# 6. COMBINE EARTHQUAKE AND PRECIPITATION
# ============================================================

annual_joint = build_annual_joint_distribution(
    earthquake_joint,
    precipitation_probabilities,
)

annual_flat = annual_joint.reshape(100)

print("\n100-STATE ANNUAL JOINT DISTRIBUTION")
print("=" * 55)

print(f"Number of states: {annual_flat.size}")
print(f"Probability sum: {annual_flat.sum():.12f}")

probability_no_payout = annual_joint[0, 0, 0, 0]

print(
    f"Probability of no payout of any kind: "
    f"{probability_no_payout:.8%}"
)


# ============================================================
# 7. THREE-YEAR TERM DISTRIBUTION
# ============================================================

term_distribution = calculate_three_year_distribution(annual_joint)

print("\nTHREE-YEAR TERM DISTRIBUTION")
print("=" * 55)

print(f"Number of terminal states: {term_distribution.size}")
print(f"Probability sum: {term_distribution.sum():.12f}")


# ============================================================
# 8. ADB FIVE FOCUS TARGETS
# ============================================================

metrics = calculate_loss_metrics(term_distribution)

print("\nTHREE-YEAR ADB RISK RESULTS")
print("=" * 55)

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

print(f"Portfolio stop-loss: ${metrics['stop_loss_usd']:,.2f}")

print(
    f"Both bonds exhausted probability: "
    f"{metrics['both_exhausted_probability']:.8%}"
)


# ============================================================
# 9. ADB SCENARIO ANALYSIS
# ============================================================

scenario_results = run_all_scenarios()

print("\nADB SCENARIO ANALYSIS")
print("=" * 90)

for scenario, scenario_metrics in scenario_results.items():

    print(f"\n{scenario.upper()}")
    print("-" * 60)

    print(
        f"Expected principal loss: "
        f"${scenario_metrics['expected_loss_millions']:.6f} million"
    )

    print(
        f"Joint attachment probability: "
        f"{scenario_metrics['joint_attachment_probability']:.8%}"
    )

    print(
        f"Portfolio loss exceedance probability: "
        f"{scenario_metrics['loss_exceedance_probability']:.8%}"
    )

    print(f"Portfolio stop-loss: ${scenario_metrics['stop_loss_usd']:,.2f}")

    print(
        f"Both bonds exhausted probability: "
        f"{scenario_metrics['both_exhausted_probability']:.8%}"
    )


# ============================================================
# 10. CLASSICAL COPULA COMPARISON
# ============================================================

print("\nCLASSICAL COPULA COMPARISON")
print("=" * 90)

copula_results = run_classical_comparison()

# Ensure that results are represented as a DataFrame.
if isinstance(copula_results, dict):
    copula_results = pd.DataFrame(copula_results)

# ============================================================
# 11. TAIL-DEPENDENCE ANALYSIS
# ============================================================

copula_results["earthquake_tail_dependence"] = copula_results.apply(
    lambda row: calculate_tail_dependence(
        rho=row["rho_earthquake"],
        family=row["family"],
        df=5 if row["family"] == "gaussian"
        else row["degrees_of_freedom"],
    ),
    axis=1,
)

copula_results["precipitation_tail_dependence"] = copula_results.apply(
    lambda row: calculate_tail_dependence(
        rho=row["rho_precipitation"],
        family=row["family"],
        df=5 if row["family"] == "gaussian"
        else row["degrees_of_freedom"],
    ),
    axis=1,
)

# ============================================================
# 12. DISPLAY CLASSICAL COPULA RESULTS
# ============================================================

display_columns = [
    "family",
    "degrees_of_freedom",
    "rho_earthquake",
    "rho_precipitation",
    "earthquake_tail_dependence",
    "precipitation_tail_dependence",
    "expected_loss_millions",
    "joint_attachment_probability",
    "loss_exceedance_probability",
    "stop_loss_usd",
    "both_exhausted_probability",
]

print("\nCLASSICAL COPULA AND TAIL-DEPENDENCE RESULTS")
print("=" * 110)

print(
    copula_results[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

save_classical_results(copula_results)

print("\nResults saved to results/classical_copula_results.csv")


# LOSS EXCEEDANCE CURVE COMPARISON


print("\n")
print("=" * 70)
print("PORTFOLIO LOSS EXCEEDANCE COMPARISON")
print("=" * 70)

exceedance_results = run_loss_exceedance_comparison()

# Display results
print(
    exceedance_results.pivot(
        index="threshold_millions",
        columns="model",
        values="exceedance_probability"
    ).to_string()
)

# Save results
save_loss_exceedance_results(exceedance_results)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for model, group in exceedance_results.groupby("model"):

    group = group.sort_values("threshold_millions")

    plt.plot(
        group["threshold_millions"],
        group["exceedance_probability"] * 100,
        marker="o",
        label=model
    )

plt.xlabel("Portfolio Loss Threshold (USD Millions)")
plt.ylabel("Exceedance Probability (%)")
plt.title("Three-Year Portfolio Loss Exceedance Curves")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/loss_exceedance_curves.png",
    dpi=300
)

plt.show()


# CLASSICAL MONTE CARLO SIMULATION

print("\n")
print("=" * 70)
print("CLASSICAL MONTE CARLO SIMULATION")
print("=" * 70)

mc_results = run_monte_carlo_comparison(
    sample_sizes=(10_000, 100_000, 1_000_000),
    seed=42
)

print("\nMONTE CARLO RESULTS")
print("=" * 70)

display_columns = [
    "model",
    "simulations",
    "expected_loss_millions",
    "expected_loss_se",
    "exceedance_probability",
    "exceedance_se",
    "exceedance_ci_lower",
    "exceedance_ci_upper",
    "stop_loss_millions",
    "stop_loss_se"
]

print(
    mc_results[display_columns].to_string(index=False)
)

save_monte_carlo_results(mc_results)

import numpy as np


def calculate_precipitation_rates(
    q_kyrgyz=0.02,
    q_tajikistan=0.02,
    rate_multiplier=1.0,
):
    """
    Convert annual trigger probabilities into Poisson rates.

    The scenario multiplier scales the Poisson rates,
    as specified in the ADB challenge brief.
    """

    mu_kyrgyz = -np.log(1.0 - q_kyrgyz) * rate_multiplier
    mu_tajikistan = -np.log(1.0 - q_tajikistan) * rate_multiplier

    return mu_kyrgyz, mu_tajikistan


def calculate_precipitation_joint_distribution(
    q_kyrgyz=0.02,
    q_tajikistan=0.02,
    rate_multiplier=1.0,
    shared_fraction=0.25,
):
    """
    Calculate the four annual precipitation outcomes:
    neither, Kyrgyz only, Tajikistan only, both.
    """

    mu1, mu2 = calculate_precipitation_rates(
        q_kyrgyz,
        q_tajikistan,
        rate_multiplier,
    )

    # Shared precipitation trigger rate
    shared_rate = shared_fraction * min(mu1, mu2)

    # Four outcomes from the ADB specification
    p00 = np.exp(-(mu1 + mu2 - shared_rate))
    p10 = np.exp(-mu2) - p00
    p01 = np.exp(-mu1) - p00
    p11 = 1.0 - p00 - p10 - p01

    probabilities = np.array([p00, p10, p01, p11])

    if np.any(probabilities < -1e-12):
        raise ValueError("Invalid precipitation probabilities.")

    probabilities = np.maximum(probabilities, 0.0)

    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Precipitation probabilities do not sum to one.")

    return {
        "mu_kyrgyz": mu1,
        "mu_tajikistan": mu2,
        "shared_rate": shared_rate,
        "probabilities": probabilities,
    }


"""
Earthquake model for the ADB Quantum Challenge.

Implements:
1. Conversion of exceedance probabilities to point probabilities.
2. Compound Poisson event rates.
3. Joint earthquake CTMC for two countries.
"""

import numpy as np
from scipy.linalg import expm

from src.benchmark_inputs import EARTHQUAKE_TAIL_PROBABILITIES


def calculate_poisson_rates(tail_probabilities):
    """
    Calculate compound Poisson rates from four annual
    earthquake payout exceedance probabilities.

    Parameters
    ----------
    tail_probabilities : dict
        Keys: 1, 2, 3, 4
        Values: probabilities of payout >= 25%, 50%,
                75%, and 100%.

    Returns
    -------
    dict
        Point probabilities and compound Poisson rates.
    """

    t1 = tail_probabilities[1]
    t2 = tail_probabilities[2]
    t3 = tail_probabilities[3]
    t4 = tail_probabilities[4]

    # Convert exceedance probabilities to point probabilities
    p0 = 1.0 - t1
    p1 = t1 - t2
    p2 = t2 - t3
    p3 = t3 - t4

    # Total Poisson arrival rate
    Lambda = -np.log(p0)

    # Individual event-size rates
    lambda1 = p1 / p0

    lambda2 = p2 / p0 - (lambda1**2) / 2.0

    lambda3 = (
        p3 / p0
        - lambda1 * lambda2
        - lambda1**3 / 6.0
    )

    lambda4 = Lambda - lambda1 - lambda2 - lambda3

    rates = np.array([
        lambda1,
        lambda2,
        lambda3,
        lambda4
    ])

    if np.any(rates < -1e-12):
        raise ValueError("Calculated Poisson rates contain negative values.")

    return {
        "point_probabilities": np.array([p0, p1, p2, p3]),
        "event_rates": rates,
        "total_rate": Lambda
    }


def calculate_all_country_rates():
    """Calculate earthquake rates for both countries."""

    results = {}

    for country, tails in EARTHQUAKE_TAIL_PROBABILITIES.items():
        results[country] = calculate_poisson_rates(tails)

    return results


def state_to_index(a, b):
    """Convert CTMC state (a,b) into an index from 0 to 24."""
    return 5 * a + b


def index_to_state(index):
    """Convert index from 0 to 24 into state (a,b)."""
    return divmod(index, 5)


def build_earthquake_generator(
    country_rates,
    shared_multiplier=1.0
):
    """
    Construct the 25 x 25 joint earthquake CTMC generator.

    The two countries retain their own total event rates.
    Shared events arrive at rate c.

    State:
        (a,b), where a and b are capped at 4.

    Returns
    -------
    numpy.ndarray
        25 x 25 CTMC generator matrix.
    """

    country1 = "Kyrgyz Republic"
    country2 = "Tajikistan"

    rates1 = country_rates[country1]
    rates2 = country_rates[country2]

    Lambda1 = rates1["total_rate"]
    Lambda2 = rates2["total_rate"]

    lambda1 = rates1["event_rates"]
    lambda2 = rates2["event_rates"]

    # Conditional probabilities from the ADB brief
    # Tajikistan given Kyrgyz Republic = 7.5%
    # Kyrgyz Republic given Tajikistan = 7.4%

    c = shared_multiplier * 0.5 * (
        0.075 * Lambda1 + 0.074 * Lambda2
    )

    if c < 0 or c > min(Lambda1, Lambda2):
        raise ValueError("Invalid shared-event rate.")

    # Conditional event-size distributions
    w1 = lambda1 / Lambda1
    w2 = lambda2 / Lambda2

    G = np.zeros((25, 25), dtype=float)

    # Construct off-diagonal transition rates
    for a in range(5):
        for b in range(5):

            current = state_to_index(a, b)

            # Events hitting country 1 only
            for k in range(1, 5):

                new_a = min(a + k, 4)
                new_b = b

                destination = state_to_index(new_a, new_b)

                rate = (Lambda1 - c) * w1[k - 1]

                if destination != current:
                    G[current, destination] += rate

            # Events hitting country 2 only
            for j in range(1, 5):

                new_a = a
                new_b = min(b + j, 4)

                destination = state_to_index(new_a, new_b)

                rate = (Lambda2 - c) * w2[j - 1]

                if destination != current:
                    G[current, destination] += rate

            # Shared events hitting both countries
            for k in range(1, 5):
                for j in range(1, 5):

                    new_a = min(a + k, 4)
                    new_b = min(b + j, 4)

                    destination = state_to_index(new_a, new_b)

                    rate = c * w1[k - 1] * w2[j - 1]

                    if destination != current:
                        G[current, destination] += rate

    # CTMC generator condition:
    # Every row must sum to zero.

    for i in range(25):
        G[i, i] = -np.sum(G[i, :])

    return G


def calculate_annual_earthquake_distribution(
    country_rates,
    shared_multiplier=1.0
):
    """
    Calculate the exact annual joint earthquake distribution.

    Initial state: (0,0)
    Annual distribution: first row of exp(G).
    """

    G = build_earthquake_generator(
        country_rates,
        shared_multiplier
    )

    transition_matrix = expm(G)

    annual_distribution = transition_matrix[0, :]

    return annual_distribution, G

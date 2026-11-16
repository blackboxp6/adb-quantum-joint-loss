
import numpy as np

from src.earthquake import (
    calculate_all_country_rates,
    calculate_annual_earthquake_distribution,
)


def test_poisson_rates():

    results = calculate_all_country_rates()

    # ADB Table 2 reference rates
    expected = {
        "Kyrgyz Republic": [
            0.016147,
            0.005498,
            0.006215,
            0.011743,
        ],

        "Tajikistan": [
            0.022711,
            0.004020,
            0.002915,
            0.010395,
        ],
    }

    for country in expected:

        calculated = results[country]["event_rates"]

        np.testing.assert_allclose(
            calculated,
            expected[country],
            atol=5e-7
        )


def test_annual_distribution():

    rates = calculate_all_country_rates()

    distribution, G = calculate_annual_earthquake_distribution(
        rates,
        shared_multiplier=1.0
    )

    # Probability distribution must sum to one
    assert np.isclose(distribution.sum(), 1.0)

    # No negative probabilities
    assert np.all(distribution >= -1e-12)

    # Generator rows must sum to zero
    assert np.allclose(G.sum(axis=1), 0.0)

    # ADB reports approximately 92.62%
    # probability of no earthquake payout in either country.
    assert np.isclose(distribution[0], 0.9262, atol=0.0002)

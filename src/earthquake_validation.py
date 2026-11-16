
import numpy as np
from scipy.stats import poisson


def compound_poisson_capped_expectation(rates, years=3):
    """
    Expected accumulated earthquake payout in quarter steps,
    capped at four quarters.
    """

    rates = np.asarray(rates, dtype=float)

    # Total event intensity over the full term.
    total_rate = years * rates.sum()

    # Probability of zero, one, ..., n events.
    max_events = 100
    event_counts = np.arange(max_events + 1)

    # Event size probabilities.
    size_probabilities = rates / rates.sum()

    # Dynamic programming for the accumulated payout.
    distribution = np.zeros(5)
    distribution[0] = 1.0

    # Compound Poisson event-count probabilities.
    count_probabilities = poisson.pmf(
        event_counts, total_rate
    )

    # Conditional payout distribution given n events.
    conditional = np.zeros((max_events + 1, 5))
    conditional[0, 0] = 1.0

    for n in range(1, max_events + 1):
        for state in range(5):
            for size, probability in enumerate(
                size_probabilities, start=1
            ):
                new_state = min(state + size, 4)
                conditional[n, new_state] += (
                    conditional[n - 1, state] * probability
                )

    distribution = (
        count_probabilities[:, None] * conditional
    ).sum(axis=0)

    expected_quarters = sum(
        state * distribution[state]
        for state in range(5)
    )

    return expected_quarters * 17.5, distribution


if __name__ == "__main__":

    kyrgyz_rates = [
        0.01614699,
        0.00549819,
        0.00621534,
        0.01174347,
    ]

    tajik_rates = [
        0.02271142,
        0.00402000,
        0.00291481,
        0.01039481,
    ]

    for name, rates in [
        ("Kyrgyz Republic", kyrgyz_rates),
        ("Tajikistan", tajik_rates),
    ]:

        loss, distribution = (
            compound_poisson_capped_expectation(rates)
        )

        print(f"\n{name}")
        print(f"Expected earthquake loss: ${loss:.6f}M")
        print(f"Probability sum: {distribution.sum():.12f}")

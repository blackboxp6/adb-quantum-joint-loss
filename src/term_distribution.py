
import numpy as np


STATE_SHAPE = (5, 5, 2, 2)


def capped_state_add(state_1, state_2):
    """
    Add two annual or cumulative states, applying
    the ADB term caps: (4, 4, 1, 1).
    """

    return (
        min(state_1[0] + state_2[0], 4),
        min(state_1[1] + state_2[1], 4),
        min(state_1[2] + state_2[2], 1),
        min(state_1[3] + state_2[3], 1),
    )


def convolve_annual_distributions(distribution_1, distribution_2):
    """
    Combine two distributions over the 100 possible states.
    The result is also a 5x5x2x2 distribution.
    """

    distribution_1 = np.asarray(distribution_1, dtype=float)
    distribution_2 = np.asarray(distribution_2, dtype=float)

    if distribution_1.shape != STATE_SHAPE:
        raise ValueError("First distribution must have shape (5,5,2,2).")

    if distribution_2.shape != STATE_SHAPE:
        raise ValueError("Second distribution must have shape (5,5,2,2).")

    result = np.zeros(STATE_SHAPE, dtype=float)

    # Only iterate over states with nonzero probability.
    states_1 = [
        (state, distribution_1[state])
        for state in np.ndindex(STATE_SHAPE)
        if distribution_1[state] > 0
    ]

    states_2 = [
        (state, distribution_2[state])
        for state in np.ndindex(STATE_SHAPE)
        if distribution_2[state] > 0
    ]

    for state_1, probability_1 in states_1:
        for state_2, probability_2 in states_2:
            combined_state = capped_state_add(state_1, state_2)

            result[combined_state] += probability_1 * probability_2

    return result


def calculate_three_year_distribution(annual_distribution):
    """
    Apply the annual distribution three times,
    starting from the zero-payout state.
    """

    annual_distribution = np.asarray(
        annual_distribution,
        dtype=float,
    )

    if annual_distribution.shape != STATE_SHAPE:
        raise ValueError("Annual distribution must have shape (5,5,2,2).")

    term_distribution = np.zeros(STATE_SHAPE, dtype=float)
    term_distribution[0, 0, 0, 0] = 1.0

    for year in range(3):
        term_distribution = convolve_annual_distributions(
            term_distribution,
            annual_distribution,
        )

    if not np.isclose(term_distribution.sum(), 1.0):
        raise ValueError("Three-year distribution does not sum to one.")

    return term_distribution

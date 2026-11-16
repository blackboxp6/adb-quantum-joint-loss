
import numpy as np

from src.term_distribution import (
    calculate_three_year_distribution,
    capped_state_add,
)


def test_capped_state_addition():
    result = capped_state_add(
        (3, 2, 1, 0),
        (2, 4, 1, 1),
    )

    assert result == (4, 4, 1, 1)


def test_three_year_distribution_normalizes():
    annual = np.zeros((5, 5, 2, 2))
    annual[0, 0, 0, 0] = 0.9
    annual[1, 1, 1, 1] = 0.1

    term = calculate_three_year_distribution(annual)

    assert term.shape == (5, 5, 2, 2)
    assert np.isclose(term.sum(), 1.0)


def test_three_year_distribution_caps_accumulation():
    annual = np.zeros((5, 5, 2, 2))
    annual[2, 3, 1, 1] = 1.0

    term = calculate_three_year_distribution(annual)

    assert term[4, 4, 1, 1] == 1.0

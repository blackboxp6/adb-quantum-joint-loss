
import numpy as np

from src.precipitation import calculate_precipitation_joint_distribution
from src.joint_distribution import build_annual_joint_distribution


def test_precipitation_probabilities_sum_to_one():
    result = calculate_precipitation_joint_distribution()

    assert np.isclose(result["probabilities"].sum(), 1.0)


def test_base_case_shared_rate():
    result = calculate_precipitation_joint_distribution()

    assert np.isclose(
        result["shared_rate"],
        0.0050506,
        atol=1e-6,
    )


def test_annual_joint_distribution_has_100_states():
    earthquake = np.zeros((5, 5))
    earthquake[0, 0] = 1.0

    precipitation = calculate_precipitation_joint_distribution()

    annual = build_annual_joint_distribution(
        earthquake,
        precipitation["probabilities"],
    )

    assert annual.shape == (5, 5, 2, 2)
    assert np.isclose(annual.sum(), 1.0)


def test_no_payout_probability():
    # ADB base-case annual earthquake no-payout probability
    earthquake_no_payout = 0.9261876974923822

    precipitation = calculate_precipitation_joint_distribution()
    p00 = precipitation["probabilities"][0]

    no_payout = earthquake_no_payout * p00

    assert np.isclose(no_payout, 0.8940, atol=0.001)


import numpy as np

from src.copulas import (
    discrete_copula_joint,
    calibrate_copula_rho,
)


# ============================================================
# GAUSSIAN COPULA TESTS
# ============================================================

def test_independent_gaussian_copula():
    marginal_x = np.array([0.7, 0.2, 0.1])
    marginal_y = np.array([0.6, 0.3, 0.1])

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=0.0,
        family="gaussian",
        df=5,
    )

    expected = np.outer(marginal_x, marginal_y)

    assert np.allclose(joint, expected, atol=1e-8)


def test_gaussian_copula_preserves_marginals():
    marginal_x = np.array([0.7, 0.2, 0.1])
    marginal_y = np.array([0.6, 0.3, 0.1])

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=0.5,
        family="gaussian",
        df=5,
    )

    assert np.allclose(joint.sum(axis=1), marginal_x, atol=1e-8)
    assert np.allclose(joint.sum(axis=0), marginal_y, atol=1e-8)


def test_gaussian_copula_calibration():
    marginal_x = np.array([0.7, 0.2, 0.1])
    marginal_y = np.array([0.6, 0.3, 0.1])

    original_rho = 0.5

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=original_rho,
        family="gaussian",
        df=5,
    )

    target = joint[1:, 1:].sum()

    estimated_rho = calibrate_copula_rho(
        marginal_x,
        marginal_y,
        target,
        family="gaussian",
        df=5,
    )

    assert np.isclose(estimated_rho, original_rho, atol=1e-7)


def test_gaussian_joint_distribution_normalizes():
    marginal_x = np.array([0.8, 0.15, 0.05])
    marginal_y = np.array([0.75, 0.20, 0.05])

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=0.4,
        family="gaussian",
        df=5,
    )

    assert np.isclose(joint.sum(), 1.0, atol=1e-8)
    assert np.all(joint >= 0)


# ============================================================
# STUDENT-t COPULA TESTS
# ============================================================

def test_student_t_copula_preserves_marginals():
    marginal_x = np.array([0.7, 0.2, 0.1])
    marginal_y = np.array([0.6, 0.3, 0.1])

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=0.5,
        family="student_t",
        df=5,
    )

    assert np.allclose(joint.sum(axis=1), marginal_x, atol=1e-8)
    assert np.allclose(joint.sum(axis=0), marginal_y, atol=1e-8)


def test_student_t_copula_normalizes():
    marginal_x = np.array([0.8, 0.15, 0.05])
    marginal_y = np.array([0.75, 0.20, 0.05])

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=0.4,
        family="student_t",
        df=5,
    )

    assert np.isclose(joint.sum(), 1.0, atol=1e-8)
    assert np.all(joint >= 0)


def test_student_t_copula_calibration():
    marginal_x = np.array([0.7, 0.2, 0.1])
    marginal_y = np.array([0.6, 0.3, 0.1])

    original_rho = 0.5
    degrees_of_freedom = 5

    joint = discrete_copula_joint(
        marginal_x,
        marginal_y,
        rho=original_rho,
        family="student_t",
        df=degrees_of_freedom,
    )

    target = joint[1:, 1:].sum()

    estimated_rho = calibrate_copula_rho(
        marginal_x,
        marginal_y,
        target,
        family="student_t",
        df=degrees_of_freedom,
    )

    assert np.isclose(estimated_rho, original_rho, atol=1e-6)


from src.copulas import calculate_tail_dependence


def test_gaussian_has_zero_asymptotic_tail_dependence():
    coefficient = calculate_tail_dependence(
        rho=0.5,
        family="gaussian",
    )

    assert coefficient == 0.0


def test_student_t_has_positive_tail_dependence():
    coefficient = calculate_tail_dependence(
        rho=0.5,
        family="student_t",
        df=5,
    )

    assert 0.0 < coefficient < 1.0

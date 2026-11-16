
import numpy as np

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm, t


def copula_cdf(u, v, rho, family="gaussian", df=5):
    """
    Evaluate a bivariate Gaussian or Student-t copula CDF.
    """

    if not (-1 < rho < 1):
        raise ValueError("rho must be between -1 and 1.")

    if not (0 <= u <= 1 and 0 <= v <= 1):
        raise ValueError("u and v must be between 0 and 1.")

    if u == 0 or v == 0:
        return 0.0

    if u == 1:
        return v

    if v == 1:
        return u

    if family == "gaussian":

        if np.isclose(rho, 0):
            return u * v

        x = norm.ppf(u)
        y = norm.ppf(v)

        denominator = np.sqrt(1 - rho**2)

        def integrand(z):
            return norm.pdf(z) * norm.cdf(
                (y - rho * z) / denominator
            )

        result, _ = quad(
            integrand, -np.inf, x,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=150
        )

        return result

    elif family == "student_t":

        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")

        x = t.ppf(u, df)
        y = t.ppf(v, df)

        def integrand(z):

            conditional_scale = np.sqrt(
                (df + z**2) * (1 - rho**2) / (df + 1)
            )

            conditional_value = (
                y - rho * z
            ) / conditional_scale

            return (
                t.pdf(z, df)
                * t.cdf(conditional_value, df + 1)
            )

        result, _ = quad(
            integrand, -np.inf, x,
            epsabs=1e-9,
            epsrel=1e-9,
            limit=150
        )

        return result

    else:
        raise ValueError(
            "Family must be 'gaussian' or 'student_t'."
        )


def discrete_copula_joint(
    marginal_x,
    marginal_y,
    rho,
    family="gaussian",
    df=5,
):
    """
    Construct a discrete joint probability matrix.

    Rows correspond to X states.
    Columns correspond to Y states.
    """

    marginal_x = np.asarray(marginal_x, dtype=float)
    marginal_y = np.asarray(marginal_y, dtype=float)

    for marginal in (marginal_x, marginal_y):

        if marginal.ndim != 1:
            raise ValueError("Marginals must be one-dimensional.")

        if np.any(marginal < 0):
            raise ValueError("Negative probabilities are invalid.")

        if not np.isclose(marginal.sum(), 1.0):
            raise ValueError("Marginal probabilities must sum to one.")

    cdf_x = np.concatenate(([0.0], np.cumsum(marginal_x)))
    cdf_y = np.concatenate(([0.0], np.cumsum(marginal_y)))

    cdf_x[-1] = 1.0
    cdf_y[-1] = 1.0

    grid = np.zeros((len(cdf_x), len(cdf_y)))

    for i, u in enumerate(cdf_x):
        for j, v in enumerate(cdf_y):

            grid[i, j] = copula_cdf(
                u, v, rho,
                family=family,
                df=df
            )

    joint = (
        grid[1:, 1:]
        - grid[:-1, 1:]
        - grid[1:, :-1]
        + grid[:-1, :-1]
    )

    if np.any(joint < -1e-8):
        raise ValueError("Negative joint probabilities detected.")

    joint = np.maximum(joint, 0.0)
    joint /= joint.sum()

    if not np.allclose(
        joint.sum(axis=1), marginal_x, atol=1e-7
    ):
        raise ValueError("First marginal not preserved.")

    if not np.allclose(
        joint.sum(axis=0), marginal_y, atol=1e-7
    ):
        raise ValueError("Second marginal not preserved.")

    return joint


def calibrate_copula_rho(
    marginal_x,
    marginal_y,
    target_joint_payout,
    family="gaussian",
    df=5,
):
    """
    Calibrate rho to match P(X > 0, Y > 0).
    """

    def objective(rho):

        cdf_zero = copula_cdf(
            marginal_x[0],
            marginal_y[0],
            rho,
            family=family,
            df=df
        )

        joint_payout = (
            1
            - marginal_x[0]
            - marginal_y[0]
            + cdf_zero
        )

        return joint_payout - target_joint_payout

    lower = -0.995
    upper = 0.995

    if objective(lower) * objective(upper) > 0:
        raise ValueError(
            "Target probability cannot be attained "
            "within the selected rho interval."
        )

    return brentq(
        objective,
        lower,
        upper,
        xtol=1e-9
    )


from scipy.stats import t as student_t


def calculate_tail_dependence(rho, family="gaussian", df=5):
    """
    Calculate the upper and lower asymptotic tail-dependence
    coefficients of a Gaussian or Student-t copula.

    For the Student-t copula, upper and lower tail dependence
    are equal.
    """

    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1.")

    if family == "gaussian":
        return 0.0

    elif family == "student_t":
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")

        argument = -np.sqrt(
            (df + 1) * (1 - rho) / (1 + rho)
        )

        coefficient = 2 * student_t.cdf(
            argument,
            df=df + 1
        )

        return float(coefficient)

    else:
        raise ValueError(
            "family must be 'gaussian' or 'student_t'."
        )

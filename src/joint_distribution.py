
import numpy as np


def build_annual_joint_distribution(
    earthquake_joint,
    precipitation_probabilities,
):
    """
    Combine earthquake and precipitation distributions.

    earthquake_joint: shape (5, 5)
    precipitation_probabilities: [P00, P10, P01, P11]

    Output shape: (5, 5, 2, 2)
    """

    earthquake_joint = np.asarray(earthquake_joint, dtype=float)
    precipitation_probabilities = np.asarray(
        precipitation_probabilities,
        dtype=float,
    )

    if earthquake_joint.shape != (5, 5):
        raise ValueError("Earthquake distribution must have shape (5, 5).")

    if precipitation_probabilities.shape != (4,):
        raise ValueError("Precipitation distribution must have four outcomes.")

    if not np.isclose(earthquake_joint.sum(), 1.0):
        raise ValueError("Earthquake probabilities must sum to one.")

    if not np.isclose(precipitation_probabilities.sum(), 1.0):
        raise ValueError("Precipitation probabilities must sum to one.")

    # Arrange precipitation outcomes as:
    # rows = Kyrgyz trigger (0, 1)
    # columns = Tajikistan trigger (0, 1)
    precipitation_joint = np.array([
        [precipitation_probabilities[0],
         precipitation_probabilities[2]],
        [precipitation_probabilities[1],
         precipitation_probabilities[3]],
    ])

    # Independent product of earthquake and precipitation outcomes
    annual_joint = np.einsum(
        "ab,ij->abij",
        earthquake_joint,
        precipitation_joint,
    )

    return annual_joint


def flatten_annual_distribution(annual_joint):
    """
    Convert the 5x5x2x2 distribution into 100 probabilities.
    """

    if annual_joint.shape != (5, 5, 2, 2):
        raise ValueError("Annual distribution must have shape (5, 5, 2, 2).")

    return annual_joint.reshape(100)

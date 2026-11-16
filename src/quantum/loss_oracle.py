
import numpy as np


def build_loss_vector(
    earthquake_limit=70.0,
    precipitation_limit=10.0,
    portfolio_attachment=80.0,
):
    """
    Calculate the total portfolio loss for each of
    the 100 terminal states (a, b, r1, r2).
    """

    losses = np.zeros((5, 5, 2, 2))

    for a in range(5):
        for b in range(5):
            for r1 in range(2):
                for r2 in range(2):

                    loss_1 = (
                        earthquake_limit * a / 4
                        + precipitation_limit * r1
                    )

                    loss_2 = (
                        earthquake_limit * b / 4
                        + precipitation_limit * r2
                    )

                    losses[a, b, r1, r2] = loss_1 + loss_2

    return losses.reshape(-1)


def build_good_state_mask(losses, attachment=80.0):
    """
    Identify states whose total portfolio loss
    strictly exceeds the attachment point.
    """

    losses = np.asarray(losses, dtype=float)

    return losses > attachment


def validate_loss_oracle(
    probabilities,
    earthquake_limit=70.0,
    precipitation_limit=10.0,
    attachment=80.0,
):
    """
    Validate the favorable-state probability
    against the classical distribution.
    """

    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    losses = build_loss_vector(
        earthquake_limit,
        precipitation_limit,
        attachment,
    )

    good_states = build_good_state_mask(
        losses,
        attachment,
    )

    if len(probabilities) != len(losses):
        raise ValueError("Probability and loss vectors must have equal length.")

    classical_probability = probabilities[good_states].sum()

    return {
        "losses": losses,
        "good_states": good_states,
        "number_of_states": len(losses),
        "number_of_good_states": int(good_states.sum()),
        "classical_probability": classical_probability,
    }

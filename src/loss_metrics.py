import numpy as np


def calculate_loss_metrics(
    term_distribution,
    earthquake_limit=70.0,
    precipitation_limit=10.0,
    portfolio_attachment=80.0,
):
    term_distribution = np.asarray(term_distribution, dtype=float)

    if term_distribution.shape != (5, 5, 2, 2):
        raise ValueError("Expected a (5, 5, 2, 2) distribution.")

    if not np.isclose(term_distribution.sum(), 1.0):
        raise ValueError("Distribution must sum to one.")

    expected_loss = 0.0
    joint_attachment = 0.0
    loss_exceedance = 0.0
    stop_loss = 0.0
    both_exhausted = 0.0

    bond_limit = earthquake_limit + precipitation_limit

    for a in range(5):
        for b in range(5):
            for r1 in range(2):
                for r2 in range(2):

                    probability = term_distribution[a, b, r1, r2]

                    loss_kyrgyz = (
                        earthquake_limit * a / 4.0
                        + precipitation_limit * r1
                    )

                    loss_tajikistan = (
                        earthquake_limit * b / 4.0
                        + precipitation_limit * r2
                    )

                    portfolio_loss = loss_kyrgyz + loss_tajikistan

                    expected_loss += probability * portfolio_loss

                    if loss_kyrgyz > 0 and loss_tajikistan > 0:
                        joint_attachment += probability

                    if portfolio_loss > portfolio_attachment:
                        loss_exceedance += probability

                    stop_loss += probability * max(
                        portfolio_loss - portfolio_attachment, 0.0
                    )

                    if (
                        np.isclose(loss_kyrgyz, bond_limit)
                        and np.isclose(loss_tajikistan, bond_limit)
                    ):
                        both_exhausted += probability

    return {
        "expected_loss_millions": expected_loss,
        "joint_attachment_probability": joint_attachment,
        "loss_exceedance_probability": loss_exceedance,
        "stop_loss_usd": stop_loss * 1_000_000,
        "both_exhausted_probability": both_exhausted,
    }
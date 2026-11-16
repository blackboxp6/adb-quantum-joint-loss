
import numpy as np

from src.monte_carlo import (
    build_copula_annual_distribution,
    simulate_portfolio_losses,
    calculate_mc_metrics,
)

annual_distribution = build_copula_annual_distribution(
    family="student_t",
    df=5,
)

seeds = [42, 123, 456, 789, 2026]
n_simulations = 1_000_000

for seed in seeds:
    rng = np.random.default_rng(seed)

    losses = simulate_portfolio_losses(
        annual_distribution=annual_distribution,
        n_simulations=n_simulations,
        rng=rng,
    )

    metrics = calculate_mc_metrics(losses)

    print(f"\nSeed: {seed}")
    print(f"Expected loss: {metrics['expected_loss_millions']:.6f}")
    print(f"Expected loss SE: {metrics['expected_loss_se']:.6f}")
    print(f"Exceedance: {metrics['exceedance_probability']:.6%}")
    print(f"Exceedance SE: {metrics['exceedance_se']:.6f}")
    print(f"Stop-loss: {metrics['stop_loss_millions']:.6f}")
    print(f"Stop-loss SE: {metrics['stop_loss_se']:.6f}")

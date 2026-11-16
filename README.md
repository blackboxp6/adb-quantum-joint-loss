# A Hybrid Classical–Quantum Framework for Joint Catastrophe-Loss Modelling

Research prototype developed for the **ADB Quantum Joint-Loss Challenge**. The project reproduces the challenge's two-country catastrophe-bond loss model and investigates classical dependence modelling, Monte Carlo estimation, and quantum amplitude estimation for rare-event risk metrics.

> **Research status:** The classical two-country benchmark has been reproduced against the ADB reference results. A simulator-based quantum amplitude-estimation prototype and resource-budget benchmarks have also been implemented. This is a research prototype: no quantum-hardware advantage has been demonstrated.

## Research objectives

1. Reproduce the ADB joint-loss model for the Kyrgyz Republic and Tajikistan.
2. Validate the five term-level risk metrics against the challenge benchmark.
3. Examine alternative dependence assumptions using Gaussian and Student-t copulas.
4. Compare exact distribution calculations with classical Monte Carlo estimation.
5. Build a quantum state-preparation and loss-marking oracle for the portfolio attachment event.
6. Evaluate maximum-likelihood quantum amplitude estimation (MLAE) against classical Monte Carlo under comparable oracle-query budgets.

## Model and risk metrics

The project models the earthquake and precipitation components of two catastrophe bonds over a three-year term. The portfolio loss is formed from the two country-level bond losses, with the relevant sublimits and portfolio attachment point.

The principal risk metrics are:

- **Expected principal loss:** \(E[L]\)
- **Joint attachment probability:** \(P(L_1 > 0, L_2 > 0)\)
- **Portfolio loss exceedance probability:** \(P(L > 80\text{ million USD})\)
- **Portfolio stop-loss:** \(E[\max(L - 80\text{ million USD}, 0)]\)
- **Both-bonds-exhausted probability:** \(P(L_1 = 80\text{ million USD}, L_2 = 80\text{ million USD})\)

The implementation uses the challenge's stated two-country model as the baseline. Additional country configurations, where explored, should be treated as extensions rather than as requirements of the original benchmark.

## Preliminary validation

The exact classical implementation reproduces the ADB scenario results to the precision reported in the challenge brief.

| Scenario | Expected loss (USD M) | Joint attachment | Exceedance above USD 80M | Stop-loss (USD M) | Both exhausted |
|---|---:|---:|---:|---:|---:|
| ADB base | 10.063513 | 4.4138% | 1.2047% | 0.282355 | 0.003163% |
| Independent | 10.063513 | 2.7156% | 0.7186% | 0.171266 | 0.000392% |
| Common shock | 10.063513 | 5.0526% | 1.6066% | 0.385043 | 0.004400% |
| Rain stress | 11.170506 | 7.0464% | 1.3215% | 0.306286 | 0.006925% |
| Rain cover | 9.970051 | 4.4138% | 0.6997% | 0.170927 | 0.003163% |

The quantum experiments currently use statevector-based simulation and classical maximum-likelihood post-processing. Preliminary depth-escalation benchmarks show lower estimation error for MLAE than the tested classical Monte Carlo estimator at the reported query budgets. These results do **not** establish end-to-end quantum speedup: state preparation, circuit synthesis, hardware noise, error correction, and full execution costs require further evaluation.

## Repository structure

```text
adb-quantum-joint-loss/
├── data/
│   └── processed/
├── src/
│   ├── benchmark_inputs.py
│   ├── earthquake.py
│   ├── precipitation.py
│   ├── joint_distribution.py
│   ├── loss_metrics.py
│   ├── scenarios.py
│   ├── term_distribution.py
│   ├── copulas.py
│   ├── classical_copula_runner.py
│   ├── monte_carlo.py
│   └── quantum/
│       ├── quantum_state.py
│       ├── loss_oracle.py
│       ├── grover.py
│       ├── amplitude_estimation.py
│       ├── benchmark.py
│       └── depth_benchmark.py
├── tests/
├── notebooks/
├── results/
├── requirements.txt
└── main.py
```

The exact contents may evolve as the project develops.

## Requirements

- Python 3.11 (the current development environment uses Python 3.11.9)
- Git
- Qiskit and Qiskit Aer (for the quantum simulation components)
- Packages listed in `requirements.txt`

## Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/adb-quantum-joint-loss.git
cd adb-quantum-joint-loss
```

If the repository is private, sign in to GitHub when prompted or configure GitHub authentication before cloning.

## Set up the Python environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

Run the main classical benchmark from the repository root:

```bash
python main.py
```

Individual experiment scripts can be run from the repository root, for example:

```bash
python test_quantum_classical_benchmark.py
python test_depth_benchmark.py
```

Use the actual script names present in your checked-out version. Some experiment scripts may be maintained at the repository root or in a dedicated experiments directory as the project is organized.

Run the test suite, if `pytest` is installed:

```bash
python -m pytest -q
```

## Reproducibility

- Randomized experiments should record their random seed.
- Preserve raw benchmark outputs alongside summary tables.
- Record package versions and hardware/simulator details for reported experiments.
- Distinguish exact-distribution results from Monte Carlo estimates.
- Report query-budget definitions and clarify which state-preparation, oracle, circuit, and post-processing costs are included.

## Research limitations

- The validated baseline is the ADB two-country model.
- Copula experiments are sensitivity analyses under alternative dependence assumptions; they are not empirical proof that a particular copula is the true dependence structure.
- Current quantum results are simulator-based.
- Oracle-query comparisons are not equivalent to a complete hardware resource or wall-clock comparison.
- Practical quantum advantage remains an open question requiring more complete resource estimation and, where feasible, hardware experiments.

## Suggested next steps

- Add a fully reproducible experiment runner and configuration files.
- Add unit tests for all model components and oracle logic.
- Document the quantum oracle's reversible implementation and gate-level resource estimates.
- Extend benchmark studies across more distributions, attachment thresholds, query budgets, and random seeds.
- Evaluate noise-aware circuits and hardware feasibility.
- Add figures and raw result files with clear provenance.

## References

- Asian Development Bank, **Quantum Joint-Loss Challenge**, challenge brief and benchmark specification: https://challenges.adb.org/en/challenges/quantum-joint-loss-challenge
- Brassard, G., Høyer, P., Mosca, M., & Tapp, A. (2002). *Quantum Amplitude Amplification and Estimation*. Contemporary Mathematics, 305, 53–74. https://doi.org/10.1090/conm/305/05215

## Disclaimer

This repository is an independent research and prototyping effort. It is not an official ADB implementation, endorsement, investment recommendation, or evidence of deployable quantum advantage. Verify the challenge's current rules and submission requirements before submitting any materials.

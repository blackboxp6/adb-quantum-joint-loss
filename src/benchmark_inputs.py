
"""
ADB Quantum Challenge
Joint-Loss Modelling for Natural Catastrophe Risk

Benchmark inputs transcribed from the challenge document.
All probabilities are represented as decimals.
"""


# 1. COUNTRIES

COUNTRIES = ["Kyrgyz Republic", "Tajikistan"]



# 2. EARTHQUAKE PAYOUT EXCEEDANCE PROBABILITIES


# Annual probabilities that earthquake payout reaches
# at least 25%, 50%, 75%, and 100% of its sub-limit.

EARTHQUAKE_TAIL_PROBABILITIES = {
    "Kyrgyz Republic": {
        1: 0.03883,
        2: 0.02331,
        3: 0.01790,
        4: 0.01184,
    },

    "Tajikistan": {
        1: 0.03925,
        2: 0.01743,
        3: 0.01332,
        4: 0.01043,
    },
}


# Published annual expected losses (USD millions)
PUBLISHED_ANNUAL_EL = {
    "Kyrgyz Republic": 2.296,
    "Tajikistan": 2.011,
}



# 3. BASE CONTRACT PARAMETERS


BASE_PARAMETERS = {
    "earthquake_limit": 70.0,
    "precipitation_limit": 10.0,

    "precipitation_probability": 0.02,

    "earthquake_shared_multiplier": 1.0,
    "precipitation_shared_fraction": 0.25,

    "term_years": 3,
    "portfolio_attachment": 80.0,
}



# 4. BENCHMARK SCENARIOS


SCENARIOS = {
    "adb-base": {
        "earthquake_shared_multiplier": 1.0,
        "precipitation_shared_fraction": 0.25,
        "precipitation_rate_multiplier": 1.0,
        "earthquake_limit": 70.0,
        "precipitation_limit": 10.0,
    },

    "adb-independent": {
        "earthquake_shared_multiplier": 0.0,
        "precipitation_shared_fraction": 0.0,
        "precipitation_rate_multiplier": 1.0,
        "earthquake_limit": 70.0,
        "precipitation_limit": 10.0,
    },

    "adb-common-shock": {
        "earthquake_shared_multiplier": 2.0,
        "precipitation_shared_fraction": 0.25,
        "precipitation_rate_multiplier": 1.0,
        "earthquake_limit": 70.0,
        "precipitation_limit": 10.0,
    },

    "adb-rain-stress": {
        "earthquake_shared_multiplier": 1.0,
        "precipitation_shared_fraction": 0.25,
        "precipitation_rate_multiplier": 2.0,
        "earthquake_limit": 70.0,
        "precipitation_limit": 10.0,
    },

    "adb-rain-cover": {
        "earthquake_shared_multiplier": 1.0,
        "precipitation_shared_fraction": 0.25,
        "precipitation_rate_multiplier": 1.0,
        "earthquake_limit": 60.0,
        "precipitation_limit": 20.0,
    },
}



# 5. REFERENCE RESULTS


REFERENCE_RESULTS = {

    "adb-base": {
        "expected_loss": 10.06,
        "joint_attachment": 0.04414,
        "loss_exceedance": 0.01205,
        "stop_loss": 282355.0,
        "both_exhausted": 0.000032,
    },

    "adb-independent": {
        "expected_loss": 10.06,
        "joint_attachment": 0.02716,
        "loss_exceedance": 0.00719,
        "stop_loss": 171266.0,
        "both_exhausted": 0.000004,
    },

    "adb-common-shock": {
        "expected_loss": 10.06,
        "joint_attachment": 0.05053,
        "loss_exceedance": 0.01607,
        "stop_loss": 385043.0,
        "both_exhausted": 0.000044,
    },

    "adb-rain-stress": {
        "expected_loss": 11.17,
        "joint_attachment": 0.07046,
        "loss_exceedance": 0.01322,
        "stop_loss": 306286.0,
        "both_exhausted": 0.000069,
    },

    "adb-rain-cover": {
        "expected_loss": 9.97,
        "joint_attachment": 0.04414,
        "loss_exceedance": 0.00700,
        "stop_loss": 170927.0,
        "both_exhausted": 0.000032,
    },
}

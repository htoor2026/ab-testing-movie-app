from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


# ============================================================
# Configuration
# ============================================================

SEED = 777

INPUT_FILE = Path(
    "simulation/data/experiment_users.csv"
)

OUTPUT_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

TRUTH_FILE = Path(
    "simulation/data/simulation_truth.csv"
)

rng = np.random.default_rng(SEED)


# ============================================================
# Baseline click probability
# ============================================================

def calculate_baseline_click_logit(
    df: pd.DataFrame,
) -> np.ndarray:

    logit = np.full(
        len(df),
        -2.65,
        dtype=float,
    )

    # Returning users engage slightly more
    logit += np.where(
        df["user_type"] == "returning",
        0.18,
        0.0,
    )

    # Engagement
    logit += np.where(
        df["engagement_level"] == "medium",
        0.15,
        0.0,
    )

    logit += np.where(
        df["engagement_level"] == "high",
        0.38,
        0.0,
    )

    # Mobile slightly lower baseline engagement
    logit += np.where(
        df["device"] == "mobile",
        -0.05,
        0.0,
    )

    # Users with more rating history are more engaged
    logit += (
        np.minimum(
            df["movies_rated"].to_numpy(),
            30,
        )
        * 0.012
    )

    return logit


# ============================================================
# Heterogeneous treatment effect
# ============================================================

def calculate_treatment_effect(
    df: pd.DataFrame,
) -> np.ndarray:

    # Small general personalization effect
    effect = np.full(
        len(df),
        0.01,
        dtype=float,
    )

    # Personalization works better for returning users
    effect += np.where(
        df["user_type"] == "returning",
        0.10,
        0.0,
    )

    # More engaged users benefit more
    effect += np.where(
        df["engagement_level"] == "high",
        0.08,
        0.0,
    )

    # Slightly stronger effect on mobile
    effect += np.where(
        df["device"] == "mobile",
        0.04,
        0.0,
    )

    # More historical information improves personalization
    effect += np.where(
        df["movies_rated"] >= 10,
        0.05,
        0.0,
    )

    # Cold-start penalty
    effect += np.where(
        df["user_type"] == "new",
        -0.06,
        0.0,
    )

    return effect


# ============================================================
# Click simulation
# ============================================================

def simulate_clicks(
    df: pd.DataFrame,
):

    baseline_logit = (
        calculate_baseline_click_logit(df)
    )

    treatment_effect = (
        calculate_treatment_effect(df)
    )

    # Potential outcome probabilities
    control_probability = expit(
        baseline_logit
    )

    treatment_probability = expit(
        baseline_logit
        + treatment_effect
    )

    # User only experiences assigned arm
    actual_probability = np.where(
        df["treatment"].to_numpy() == 1,
        treatment_probability,
        control_probability,
    )

    clicked = rng.binomial(
        n=1,
        p=actual_probability,
    )

    return (
        clicked,
        control_probability,
        treatment_probability,
        actual_probability,
    )


# ============================================================
# Watchlist outcome
# ============================================================

def simulate_watchlist(
    df: pd.DataFrame,
    clicked: np.ndarray,
) -> np.ndarray:

    logit = np.full(
        len(df),
        -2.0,
        dtype=float,
    )

    logit += np.where(
        df["user_type"] == "returning",
        0.12,
        0.0,
    )

    logit += np.where(
        df["engagement_level"] == "high",
        0.22,
        0.0,
    )

    # Treatment can improve downstream intent
    logit += np.where(
        df["treatment"] == 1,
        0.15,
        0.0,
    )

    probability = expit(
        logit
    )

    watchlisted = rng.binomial(
        n=1,
        p=probability,
    )

    # Cannot watchlist without first clicking
    watchlisted = (
        watchlisted
        * clicked
    )

    return watchlisted


# ============================================================
# Latency guardrail
# ============================================================

def simulate_latency(
    df: pd.DataFrame,
) -> np.ndarray:

    latency = rng.normal(
        loc=120,
        scale=18,
        size=len(df),
    )

    # Personalized recommender costs more latency
    latency += np.where(
        df["treatment"] == 1,
        28,
        0,
    )

    # High-engagement profiles slightly more expensive
    latency += np.where(
        df["engagement_level"] == "high",
        6,
        0,
    )

    latency = np.clip(
        latency,
        60,
        250,
    )

    return np.round(
        latency,
        1,
    )


# ============================================================
# Experiment day
# ============================================================

def simulate_experiment_day(
    n_users: int,
) -> np.ndarray:

    days = np.arange(
        1,
        15,
    )

    weights = np.array(
        [
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.10,
            1.15,
            1.20,
            1.25,
            1.30,
            1.35,
        ],
        dtype=float,
    )

    weights /= weights.sum()

    return rng.choice(
        days,
        size=n_users,
        p=weights,
    )


# ============================================================
# Main simulation
# ============================================================

def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    (
        clicked,
        control_probability,
        treatment_probability,
        actual_probability,
    ) = simulate_clicks(
        df
    )

    outcomes = df.copy()

    outcomes["clicked"] = clicked

    outcomes["watchlisted"] = (
        simulate_watchlist(
            outcomes,
            clicked,
        )
    )

    outcomes["latency_ms"] = (
        simulate_latency(
            outcomes
        )
    )

    outcomes["experiment_day"] = (
        simulate_experiment_day(
            len(outcomes)
        )
    )

    # This is useful for simulation diagnostics only.
    outcomes[
        "simulated_click_probability"
    ] = actual_probability


    # ========================================================
    # Hidden simulation truth
    # ========================================================

    truth = pd.DataFrame(
        {
            "user_id":
                df["user_id"],

            "true_control_click_prob":
                control_probability,

            "true_treatment_click_prob":
                treatment_probability,

            "true_individual_uplift":
                (
                    treatment_probability
                    - control_probability
                ),
        }
    )


    # ========================================================
    # Save
    # ========================================================

    outcomes.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    truth.to_csv(
        TRUTH_FILE,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SIMULATED A/B EXPERIMENT"
    )

    print(
        "=" * 70
    )

    summary = (
        outcomes
        .groupby("variant")
        .agg(
            users=(
                "user_id",
                "count",
            ),
            ctr=(
                "clicked",
                "mean",
            ),
            watchlist_rate=(
                "watchlisted",
                "mean",
            ),
            latency_ms=(
                "latency_ms",
                "mean",
            ),
        )
    )

    summary["ctr"] *= 100
    summary["watchlist_rate"] *= 100

    print(
        summary.round(3)
    )

    control_ctr = (
        outcomes.loc[
            outcomes["variant"] == "control",
            "clicked",
        ].mean()
    )

    treatment_ctr = (
        outcomes.loc[
            outcomes["variant"] == "treatment",
            "clicked",
        ].mean()
    )

    print(
        f"\nAbsolute CTR lift: "
        f"{(treatment_ctr - control_ctr) * 100:+.3f} pp"
    )

    print(
        f"\nSaved:\n"
        f"{OUTPUT_FILE}\n"
        f"{TRUTH_FILE}"
    )


if __name__ == "__main__":
    main()
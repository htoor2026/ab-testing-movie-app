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

RELEVANCE_FILE = Path(
    "recommender/data/recommendation_relevance.csv"
)

OUTPUT_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

rng = np.random.default_rng(SEED)


# ============================================================
# 1. Baseline click tendency
# ============================================================

def calculate_baseline_click_logit(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Estimate each user's baseline tendency to click a
    recommendation BEFORE accounting for recommendation quality.

    This captures general engagement differences between users.
    """

    # Starting log-odds.
    # expit(-2.65) is roughly a 6-7% click probability.
    logit = np.full(
        len(df),
        -2.65,
    )

    # Returning users generally know the product better
    # and engage somewhat more.
    logit += np.where(
        df["user_type"] == "returning",
        0.18,
        0.0,
    )

    # Medium engagement
    logit += np.where(
        df["engagement_level"] == "medium",
        0.15,
        0.0,
    )

    # High engagement
    logit += np.where(
        df["engagement_level"] == "high",
        0.38,
        0.0,
    )

    # Slightly lower baseline click behavior on mobile.
    logit += np.where(
        df["device"] == "mobile",
        -0.05,
        0.0,
    )

    # Users with more historical ratings tend to be
    # more engaged with movie discovery.
    logit += (
        np.minimum(
            df["movies_rated"],
            30,
        )
        * 0.012
    )

    return logit


# ============================================================
# 2. Simulate clicks from ACTUAL recommendation relevance
# ============================================================

def simulate_clicks(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Simulate whether each user clicks a recommendation.

    IMPORTANT:
    There is NO direct treatment bonus here.

    Click probability depends on:
        1. baseline user engagement
        2. actual recommendation relevance

    Therefore, treatment only wins if the personalized
    recommender actually produces better recommendations.
    """

    baseline_logit = (
        calculate_baseline_click_logit(
            df
        )
    )

    relevance = (
        df[
            "recommendation_relevance"
        ]
        .fillna(0)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Recommendation relevance effect
    # --------------------------------------------------------
    #
    # recommendation_relevance is usually fairly small,
    # so scale it onto the log-odds scale.
    #
    # IMPORTANT:
    # This is not a treatment coefficient.
    #
    # Both control and treatment users receive this same
    # transformation. Better recommendation quality simply
    # leads to higher click probability.
    # --------------------------------------------------------

    RELEVANCE_STRENGTH = 8.0

    relevance_effect = (
        relevance
        * RELEVANCE_STRENGTH
    )

    click_logit = (
        baseline_logit
        + relevance_effect
    )

    actual_probability = expit(
        click_logit
    )

    # Generate observed binary outcome.
    clicked = rng.binomial(
        n=1,
        p=actual_probability,
    )

    return (
        clicked,
        actual_probability,
    )


# ============================================================
# 3. Simulate watchlist behavior
# ============================================================

def simulate_watchlist(
    df: pd.DataFrame,
    clicked: np.ndarray,
) -> np.ndarray:
    """
    Simulate whether a clicked recommendation is added
    to the user's watchlist.

    This also depends on recommendation relevance rather
    than directly on treatment assignment.
    """

    logit = np.full(
        len(df),
        -2.0,
    )

    # Returning users are more likely to maintain a watchlist.
    logit += np.where(
        df["user_type"] == "returning",
        0.12,
        0.0,
    )

    # Highly engaged users are more likely to save content.
    logit += np.where(
        df["engagement_level"] == "high",
        0.22,
        0.0,
    )

    # Recommendation quality affects downstream engagement.
    relevance = (
        df[
            "recommendation_relevance"
        ]
        .fillna(0)
        .to_numpy()
    )

    logit += (
        relevance * 5.0
    )

    watchlist_probability = expit(
        logit
    )

    watchlisted = rng.binomial(
        n=1,
        p=watchlist_probability,
    )

    # Cannot add a recommendation to the watchlist
    # without interacting with it first.
    watchlisted = (
        watchlisted
        * clicked
    )

    return watchlisted


# ============================================================
# 4. Simulate recommendation latency
# ============================================================

def simulate_latency(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Simulate recommendation-generation latency.

    Personalized collaborative filtering requires more
    computation than a cached popularity ranking, so treatment
    still has an engineering cost.

    This is a system-performance effect, not a behavioral bonus.
    """

    # Base popularity recommendation latency
    latency = rng.normal(
        loc=120,
        scale=18,
        size=len(df),
    )

    # Personalized algorithm costs additional computation.
    latency += np.where(
        df["variant"] == "treatment",
        28,
        0,
    )

    # More history can require somewhat more work.
    latency += np.where(
        df["visible_history_count"] >= 10,
        4,
        0,
    )

    latency += np.where(
        df["visible_history_count"] >= 25,
        3,
        0,
    )

    latency = np.clip(
        latency,
        60,
        300,
    )

    return np.round(
        latency,
        1,
    )


# ============================================================
# 5. Simulate experiment day
# ============================================================

def simulate_experiment_day(
    n_users: int,
) -> np.ndarray:
    """
    Spread users over a 14-day experiment.

    Traffic gradually increases over time.
    """

    days = np.arange(
        1,
        15,
    )

    traffic_weights = np.array(
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
        ]
    )

    traffic_weights = (
        traffic_weights
        / traffic_weights.sum()
    )

    return rng.choice(
        days,
        size=n_users,
        p=traffic_weights,
    )


# ============================================================
# 6. Merge recommendation quality into experiment users
# ============================================================

def merge_relevance(
    experiment_users: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach recommendation-quality metrics produced from
    actual Control/Treatment recommendation lists.
    """

    relevance = pd.read_csv(
        RELEVANCE_FILE
    )

    columns = [
        "user_id",
        "visible_history_count",
        "future_rating_count",
        "future_positive_count",
        "recommendation_hits",
        "precision_at_10",
        "recommendation_relevance",
    ]

    df = experiment_users.merge(
        relevance[columns],
        on="user_id",
        how="left",
    )

    # --------------------------------------------------------
    # Validate merge
    # --------------------------------------------------------

    missing_relevance = (
        df[
            "recommendation_relevance"
        ]
        .isna()
        .sum()
    )

    if missing_relevance > 0:
        raise ValueError(
            f"{missing_relevance:,} users are missing "
            "recommendation relevance."
        )

    return df


# ============================================================
# 7. Simulate behavior
# ============================================================

def simulate_behavior(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Click
    # --------------------------------------------------------

    (
        clicked,
        click_probability,
    ) = simulate_clicks(
        df
    )

    df["clicked"] = clicked

    # Keep probability for debugging / simulation diagnostics.
    # Do NOT treat this as observed real-world information.
    df[
        "simulated_click_probability"
    ] = click_probability


    # --------------------------------------------------------
    # Watchlist
    # --------------------------------------------------------

    df["watchlisted"] = (
        simulate_watchlist(
            df,
            clicked,
        )
    )


    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    df["latency_ms"] = (
        simulate_latency(
            df
        )
    )


    # --------------------------------------------------------
    # Experiment day
    # --------------------------------------------------------

    df["experiment_day"] = (
        simulate_experiment_day(
            len(df)
        )
    )

    return df


# ============================================================
# 8. Validation
# ============================================================

def validate_outcomes(
    df: pd.DataFrame,
) -> None:

    assert (
        df["user_id"]
        .is_unique
    )

    assert (
        df["clicked"]
        .isin([0, 1])
        .all()
    )

    assert (
        df["watchlisted"]
        .isin([0, 1])
        .all()
    )

    assert (
        df["recommendation_relevance"]
        .notna()
        .all()
    )

    assert (
        df["precision_at_10"]
        .between(
            0,
            1,
        )
        .all()
    )

    assert (
        df["latency_ms"]
        .gt(0)
        .all()
    )

    # Watchlist must imply click.
    invalid_watchlist = (
        (
            df["watchlisted"] == 1
        )
        &
        (
            df["clicked"] == 0
        )
    ).sum()

    assert invalid_watchlist == 0

    print(
        "\n✅ Outcome validation passed."
    )


# ============================================================
# 9. Summary
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SIMULATED EXPERIMENT OUTCOMES "
        "FROM REAL RECOMMENDATION QUALITY"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # Overall A/B metrics
    # --------------------------------------------------------

    summary = (
        df.groupby("variant")
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
            avg_latency_ms=(
                "latency_ms",
                "mean",
            ),
            avg_relevance=(
                "recommendation_relevance",
                "mean",
            ),
            avg_precision_at_10=(
                "precision_at_10",
                "mean",
            ),
        )
    )

    summary[
        "ctr"
    ] *= 100

    summary[
        "watchlist_rate"
    ] *= 100

    summary[
        "avg_precision_at_10"
    ] *= 100

    print(
        "\nOverall metrics:"
    )

    print(
        summary.round(4)
    )


    # --------------------------------------------------------
    # CTR by user type
    # --------------------------------------------------------

    print(
        "\nCTR by user type:"
    )

    user_type_ctr = (
        df.groupby(
            [
                "variant",
                "user_type",
            ]
        )[
            "clicked"
        ]
        .mean()
        .mul(100)
        .round(3)
    )

    print(
        user_type_ctr
    )


    # --------------------------------------------------------
    # Relevance by user type
    # --------------------------------------------------------

    print(
        "\nRecommendation relevance "
        "by user type:"
    )

    relevance_segment = (
        df.groupby(
            [
                "variant",
                "user_type",
            ]
        )[
            "recommendation_relevance"
        ]
        .mean()
        .round(5)
    )

    print(
        relevance_segment
    )


    # --------------------------------------------------------
    # CTR by visible history
    # --------------------------------------------------------

    history_bins = pd.cut(
        df[
            "visible_history_count"
        ],
        bins=[
            -1,
            1,
            5,
            10,
            20,
            1000,
        ],
        labels=[
            "0-1",
            "2-5",
            "6-10",
            "11-20",
            "21+",
        ],
    )

    temp = df.copy()

    temp[
        "history_bucket"
    ] = history_bins

    print(
        "\nCTR by visible history:"
    )

    history_ctr = (
        temp.groupby(
            [
                "variant",
                "history_bucket",
            ],
            observed=True,
        )[
            "clicked"
        ]
        .mean()
        .mul(100)
        .round(3)
    )

    print(
        history_ctr
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load randomized users
    # --------------------------------------------------------

    experiment_users = pd.read_csv(
        INPUT_FILE
    )


    # --------------------------------------------------------
    # Attach real recommendation quality
    # --------------------------------------------------------

    experiment_users = (
        merge_relevance(
            experiment_users
        )
    )


    # --------------------------------------------------------
    # Generate behavior
    # --------------------------------------------------------

    outcomes = simulate_behavior(
        experiment_users
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_outcomes(
        outcomes
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    outcomes.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        outcomes
    )


    print(
        f"\nSaved experiment outcomes to:\n"
        f"{OUTPUT_FILE}"
    )
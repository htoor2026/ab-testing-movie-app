from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

TRUTH_FILE = Path(
    "simulation/data/simulation_truth.csv"
)

OUTPUT_DIR = Path("analysis/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 2026


# Because our experiment randomized users 50/50
PROPENSITY = 0.50


# ============================================================
# Features
# ============================================================

CATEGORICAL_FEATURES = [
    "user_type",
    "device",
    "engagement_level",
    "preferred_genre",
]

NUMERIC_FEATURES = [
    "account_age_days",
    "previous_sessions",
    "movies_rated",
    "avg_session_minutes",
]

FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# ============================================================
# Preprocessing
# ============================================================

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


# ============================================================
# Outcome models
# ============================================================

def create_outcome_model():

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# Treatment-effect models
# ============================================================

def create_effect_model():

    model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=10,
        min_samples_leaf=100,
        l2_regularization=5.0,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# Train X-Learner
# ============================================================

def train_x_learner(train_df):

    control = train_df[
        train_df["treatment"] == 0
    ].copy()

    treated = train_df[
        train_df["treatment"] == 1
    ].copy()

    # --------------------------------------------------------
    # Stage 1:
    # Learn outcome functions
    #
    # m0(X) = predicted outcome under control
    # m1(X) = predicted outcome under treatment
    # --------------------------------------------------------

    m0 = create_outcome_model()
    m1 = create_outcome_model()

    m0.fit(
        control[FEATURES],
        control["clicked"],
    )

    m1.fit(
        treated[FEATURES],
        treated["clicked"],
    )

    # --------------------------------------------------------
    # Stage 2:
    # Impute individual treatment effects
    # --------------------------------------------------------

    # Treated users:
    #
    # We know their actual treated outcome Y1.
    # We estimate what would have happened under control.
    #
    # D1 = Y1 - m0(X)

    predicted_control_for_treated = (
        m0.predict_proba(
            treated[FEATURES]
        )[:, 1]
    )

    treated["D1"] = (
        treated["clicked"]
        - predicted_control_for_treated
    )

    # Control users:
    #
    # We know their actual control outcome Y0.
    # Estimate treatment outcome.
    #
    # D0 = m1(X) - Y0

    predicted_treatment_for_control = (
        m1.predict_proba(
            control[FEATURES]
        )[:, 1]
    )

    control["D0"] = (
        predicted_treatment_for_control
        - control["clicked"]
    )

    # --------------------------------------------------------
    # Stage 3:
    # Learn treatment-effect functions
    # --------------------------------------------------------

    tau1_model = create_effect_model()
    tau0_model = create_effect_model()

    tau1_model.fit(
        treated[FEATURES],
        treated["D1"],
    )

    tau0_model.fit(
        control[FEATURES],
        control["D0"],
    )

    return (
        m0,
        m1,
        tau0_model,
        tau1_model,
    )


# ============================================================
# Predict CATE
# ============================================================

def predict_x_learner(
    df,
    m0,
    m1,
    tau0_model,
    tau1_model,
):

    results = df.copy()

    # Outcome-model predictions
    results["pred_control_prob"] = (
        m0.predict_proba(
            results[FEATURES]
        )[:, 1]
    )

    results["pred_treatment_prob"] = (
        m1.predict_proba(
            results[FEATURES]
        )[:, 1]
    )

    # Two treatment-effect estimates
    tau0 = tau0_model.predict(
        results[FEATURES]
    )

    tau1 = tau1_model.predict(
        results[FEATURES]
    )

    results["tau0"] = tau0
    results["tau1"] = tau1

    # --------------------------------------------------------
    # Final X-Learner estimate
    #
    # tau(X) =
    # e(X) * tau0(X)
    # +
    # (1-e(X)) * tau1(X)
    #
    # e(X) = 0.5 because randomization was 50/50
    # --------------------------------------------------------

    results["predicted_cate"] = (
        PROPENSITY * tau0
        +
        (1 - PROPENSITY) * tau1
    )

    return results


# ============================================================
# Uplift@K
# ============================================================

def uplift_at_k(
    df,
    fraction,
):

    ranked = df.sort_values(
        "predicted_cate",
        ascending=False,
    )

    n_users = int(
        np.ceil(
            len(ranked) * fraction
        )
    )

    top = ranked.head(
        n_users
    )

    treated = top[
        top["treatment"] == 1
    ]

    control = top[
        top["treatment"] == 0
    ]

    treated_ctr = (
        treated["clicked"].mean()
    )

    control_ctr = (
        control["clicked"].mean()
    )

    return (
        treated_ctr - control_ctr
    )


# ============================================================
# Evaluate against simulation truth
# ============================================================

def evaluate_truth(predictions):

    truth = pd.read_csv(
        TRUTH_FILE
    )

    merged = predictions.merge(
        truth,
        on="user_id",
        how="left",
    )

    mae = np.mean(
        np.abs(
            merged["predicted_cate"]
            -
            merged["true_individual_uplift"]
        )
    )

    pearson = np.corrcoef(
        merged["predicted_cate"],
        merged["true_individual_uplift"],
    )[0, 1]

    spearman, spearman_p = spearmanr(
        merged["predicted_cate"],
        merged["true_individual_uplift"],
    )

    return (
        merged,
        mae,
        pearson,
        spearman,
        spearman_p,
    )


# ============================================================
# Decile evaluation
# ============================================================

def evaluate_deciles(df):

    df = df.copy()

    ranking = df[
        "predicted_cate"
    ].rank(
        method="first",
        ascending=False,
    )

    df["uplift_decile"] = pd.qcut(
        ranking,
        q=10,
        labels=range(1, 11),
    ).astype(int)

    results = []

    for decile in range(1, 11):

        segment = df[
            df["uplift_decile"]
            == decile
        ]

        treated = segment[
            segment["treatment"] == 1
        ]

        control = segment[
            segment["treatment"] == 0
        ]

        observed_uplift = (
            treated["clicked"].mean()
            -
            control["clicked"].mean()
        )

        results.append(
            {
                "decile": decile,

                "users":
                    len(segment),

                "predicted_uplift":
                    segment[
                        "predicted_cate"
                    ].mean(),

                "observed_uplift":
                    observed_uplift,

                "true_uplift":
                    segment[
                        "true_individual_uplift"
                    ].mean(),

                "returning_share":
                    (
                        segment["user_type"]
                        == "returning"
                    ).mean(),

                "avg_movies_rated":
                    segment[
                        "movies_rated"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(
        DATA_FILE
    )

    # Same split as T-Learner
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["treatment"],
    )

    print("=" * 70)
    print("X-LEARNER")
    print("=" * 70)

    print(
        f"\nTraining users: "
        f"{len(train_df):,}"
    )

    print(
        f"Test users: "
        f"{len(test_df):,}"
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        m0,
        m1,
        tau0_model,
        tau1_model,
    ) = train_x_learner(
        train_df
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = predict_x_learner(
        test_df,
        m0,
        m1,
        tau0_model,
        tau1_model,
    )


    print(
        "\nAverage predicted CATE:"
    )

    print(
        f"{predictions['predicted_cate'].mean() * 100:.3f} "
        "percentage points"
    )


    # --------------------------------------------------------
    # Uplift@K
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("UPLIFT@K")
    print("=" * 70)

    for fraction in [
        0.10,
        0.20,
        0.30,
    ]:

        uplift = uplift_at_k(
            predictions,
            fraction,
        )

        print(
            f"Top {int(fraction * 100)}%: "
            f"{uplift * 100:.3f} "
            "percentage points"
        )


    # --------------------------------------------------------
    # Truth evaluation
    # --------------------------------------------------------

    (
        predictions,
        mae,
        pearson,
        spearman,
        spearman_p,
    ) = evaluate_truth(
        predictions
    )

    print("\n" + "=" * 70)
    print("SIMULATION TRUTH EVALUATION")
    print("=" * 70)

    print(
        f"\nCATE MAE: "
        f"{mae * 100:.3f} "
        "percentage points"
    )

    print(
        f"Pearson correlation: "
        f"{pearson:.3f}"
    )

    print(
        f"Spearman ranking correlation: "
        f"{spearman:.3f}"
    )

    print(
        f"Spearman p-value: "
        f"{spearman_p:.4g}"
    )


    # --------------------------------------------------------
    # Deciles
    # --------------------------------------------------------

    deciles = evaluate_deciles(
        predictions
    )

    display = deciles.copy()

    percentage_columns = [
        "predicted_uplift",
        "observed_uplift",
        "true_uplift",
        "returning_share",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("\n" + "=" * 90)
    print("X-LEARNER UPLIFT DECILES")
    print("=" * 90)

    print(
        display.round(3)
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    predictions.to_csv(
        OUTPUT_DIR
        / "x_learner_predictions.csv",
        index=False,
    )

    deciles.to_csv(
        OUTPUT_DIR
        / "x_learner_deciles.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        "analysis/outputs/"
        "x_learner_predictions.csv"
    )

    print(
        "analysis/outputs/"
        "x_learner_deciles.csv"
    )
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
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
# Model
# ============================================================

def create_model():

    preprocessor = ColumnTransformer(
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

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# Train T-Learner
# ============================================================

def train_t_learner(train_df):

    control = train_df[
        train_df["treatment"] == 0
    ]

    treatment = train_df[
        train_df["treatment"] == 1
    ]

    control_model = create_model()
    treatment_model = create_model()

    # Model 1:
    # P(click | Control, X)

    control_model.fit(
        control[FEATURES],
        control["clicked"],
    )

    # Model 2:
    # P(click | Treatment, X)

    treatment_model.fit(
        treatment[FEATURES],
        treatment["clicked"],
    )

    return control_model, treatment_model


# ============================================================
# Predict CATE
# ============================================================

def predict_uplift(
    df,
    control_model,
    treatment_model,
):

    results = df.copy()

    # Predicted click probability under control

    results["pred_control_prob"] = (
        control_model.predict_proba(
            results[FEATURES]
        )[:, 1]
    )

    # Predicted click probability under treatment

    results["pred_treatment_prob"] = (
        treatment_model.predict_proba(
            results[FEATURES]
        )[:, 1]
    )

    # CATE / predicted uplift

    results["predicted_cate"] = (
        results["pred_treatment_prob"]
        - results["pred_control_prob"]
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

    n = int(
        np.ceil(
            len(ranked) * fraction
        )
    )

    top = ranked.head(n)

    treatment = top[
        top["treatment"] == 1
    ]

    control = top[
        top["treatment"] == 0
    ]

    treatment_rate = (
        treatment["clicked"].mean()
    )

    control_rate = (
        control["clicked"].mean()
    )

    uplift = (
        treatment_rate
        - control_rate
    )

    return uplift


# ============================================================
# Qini curve
# ============================================================

def calculate_qini(df):

    ranked = (
        df.sort_values(
            "predicted_cate",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranked["treated"] = (
        ranked["treatment"] == 1
    ).astype(int)

    ranked["control"] = (
        ranked["treatment"] == 0
    ).astype(int)

    ranked["treated_click"] = (
        ranked["treated"]
        * ranked["clicked"]
    )

    ranked["control_click"] = (
        ranked["control"]
        * ranked["clicked"]
    )

    ranked["cum_treated"] = (
        ranked["treated"].cumsum()
    )

    ranked["cum_control"] = (
        ranked["control"].cumsum()
    )

    ranked["cum_treated_click"] = (
        ranked["treated_click"].cumsum()
    )

    ranked["cum_control_click"] = (
        ranked["control_click"].cumsum()
    )

    ratio = (
        ranked["cum_treated"]
        /
        ranked["cum_control"].replace(
            0,
            np.nan,
        )
    )

    ranked["qini_gain"] = (
        ranked["cum_treated_click"]
        -
        ranked["cum_control_click"]
        * ratio
    )

    ranked["qini_gain"] = (
        ranked["qini_gain"]
        .fillna(0)
    )

    ranked["fraction_users"] = (
        np.arange(1, len(ranked) + 1)
        / len(ranked)
    )

    final_gain = (
        ranked["qini_gain"].iloc[-1]
    )

    ranked["random_gain"] = (
        ranked["fraction_users"]
        * final_gain
    )

    qini_score = np.trapezoid(
    ranked["qini_gain"] - ranked["random_gain"],
    ranked["fraction_users"],
    )

    return ranked, qini_score


# ============================================================
# Compare with simulation truth
# ============================================================

def evaluate_against_truth(
    predictions,
):

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

    correlation = np.corrcoef(
        merged["predicted_cate"],
        merged["true_individual_uplift"],
    )[0, 1]

    return merged, mae, correlation


# ============================================================
# Uplift deciles
# ============================================================

def create_uplift_deciles(df):

    df = df.copy()

    df["uplift_decile"] = pd.qcut(
        df["predicted_cate"].rank(
            method="first"
        ),
        q=10,
        labels=False,
    )

    # Make 1 = highest uplift group
    df["uplift_decile"] = (
        10 - df["uplift_decile"]
    )

    summary = (
        df.groupby("uplift_decile")
        .agg(
            users=("user_id", "count"),
            predicted_uplift=(
                "predicted_cate",
                "mean",
            ),
            returning_share=(
                "user_type",
                lambda x:
                (x == "returning").mean(),
            ),
            avg_movies_rated=(
                "movies_rated",
                "mean",
            ),
        )
        .reset_index()
    )

    return df, summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(
        DATA_FILE
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["treatment"],
    )

    print("=" * 70)
    print("T-LEARNER UPLIFT MODEL")
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
    # Train models
    # --------------------------------------------------------

    control_model, treatment_model = (
        train_t_learner(
            train_df
        )
    )


    # --------------------------------------------------------
    # Predict uplift
    # --------------------------------------------------------

    predictions = predict_uplift(
        test_df,
        control_model,
        treatment_model,
    )


    # --------------------------------------------------------
    # Overall predicted CATE
    # --------------------------------------------------------

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

    for k in [
        0.10,
        0.20,
        0.30,
    ]:

        uplift = uplift_at_k(
            predictions,
            k,
        )

        print(
            f"Top {int(k * 100)}%: "
            f"{uplift * 100:.3f} "
            "percentage points"
        )


    # --------------------------------------------------------
    # Qini
    # --------------------------------------------------------

    qini_df, qini_score = (
        calculate_qini(
            predictions
        )
    )

    print("\nQini score:")
    print(
        f"{qini_score:.4f}"
    )


    # --------------------------------------------------------
    # Compare with synthetic truth
    # --------------------------------------------------------

    (
        predictions_truth,
        mae,
        correlation,
    ) = evaluate_against_truth(
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
        f"Correlation with true uplift: "
        f"{correlation:.3f}"
    )


    # --------------------------------------------------------
    # Decile analysis
    # --------------------------------------------------------

    predictions_truth, deciles = (
        create_uplift_deciles(
            predictions_truth
        )
    )

    print("\n" + "=" * 70)
    print("UPLIFT DECILES")
    print("=" * 70)

    display = deciles.copy()

    display["predicted_uplift"] *= 100
    display["returning_share"] *= 100

    print(
        display.round(3)
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_truth.to_csv(
        OUTPUT_DIR
        / "uplift_predictions.csv",
        index=False,
    )

    deciles.to_csv(
        OUTPUT_DIR
        / "uplift_deciles.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Plot Qini curve
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        qini_df["fraction_users"],
        qini_df["qini_gain"],
        label="T-Learner",
    )

    plt.plot(
        qini_df["fraction_users"],
        qini_df["random_gain"],
        linestyle="--",
        label="Random targeting",
    )

    plt.xlabel(
        "Fraction of users targeted"
    )

    plt.ylabel(
        "Incremental clicks"
    )

    plt.title(
        "Qini Curve — Personalized Recommendations"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "qini_curve.png",
        dpi=150,
    )

    plt.close()


    print(
        "\nSaved:"
    )

    print(
        "analysis/outputs/"
        "uplift_predictions.csv"
    )

    print(
        "analysis/outputs/"
        "uplift_deciles.csv"
    )

    print(
        "analysis/outputs/"
        "qini_curve.png"
    )
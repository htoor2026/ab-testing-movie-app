from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INPUT_FILE = Path(
    "analysis/outputs/uplift_predictions.csv"
)

OUTPUT_FILE = Path(
    "analysis/outputs/uplift_decile_evaluation.csv"
)


df = pd.read_csv(INPUT_FILE)


# --------------------------------------------------
# Create uplift deciles
# 1 = users predicted to benefit the most
# --------------------------------------------------

df["uplift_decile"] = pd.qcut(
    df["predicted_cate"].rank(
        method="first",
        ascending=False,
    ),
    q=10,
    labels=range(1, 11),
)


# --------------------------------------------------
# Evaluate each decile
# --------------------------------------------------

results = []

for decile in range(1, 11):

    segment = df[
        df["uplift_decile"] == decile
    ]

    treatment = segment[
        segment["treatment"] == 1
    ]

    control = segment[
        segment["treatment"] == 0
    ]

    treatment_rate = treatment[
        "clicked"
    ].mean()

    control_rate = control[
        "clicked"
    ].mean()

    observed_uplift = (
        treatment_rate
        - control_rate
    )

    # Standard error for observed difference
    se = np.sqrt(
        treatment_rate
        * (1 - treatment_rate)
        / len(treatment)
        +
        control_rate
        * (1 - control_rate)
        / len(control)
    )

    ci_lower = (
        observed_uplift
        - 1.96 * se
    )

    ci_upper = (
        observed_uplift
        + 1.96 * se
    )

    predicted_uplift = segment[
        "predicted_cate"
    ].mean()

    true_uplift = segment[
        "true_individual_uplift"
    ].mean()

    returning_share = (
        segment["user_type"]
        .eq("returning")
        .mean()
    )

    results.append(
        {
            "decile": decile,
            "users": len(segment),

            "predicted_uplift":
                predicted_uplift,

            "observed_uplift":
                observed_uplift,

            "true_uplift":
                true_uplift,

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            "returning_share":
                returning_share,

            "avg_movies_rated":
                segment[
                    "movies_rated"
                ].mean(),
        }
    )


results_df = pd.DataFrame(results)


# --------------------------------------------------
# Ranking quality
# --------------------------------------------------

spearman_corr, spearman_p = spearmanr(
    df["predicted_cate"],
    df["true_individual_uplift"],
)


# --------------------------------------------------
# Display
# --------------------------------------------------

display = results_df.copy()

percentage_cols = [
    "predicted_uplift",
    "observed_uplift",
    "true_uplift",
    "ci_lower",
    "ci_upper",
    "returning_share",
]

for column in percentage_cols:
    display[column] *= 100


print("=" * 95)
print("UPLIFT DECILE VALIDATION")
print("=" * 95)

print(
    display.round(3)
    .to_string(index=False)
)


print("\nRanking quality:")

print(
    f"Spearman correlation with true uplift: "
    f"{spearman_corr:.3f}"
)

print(
    f"Spearman p-value: "
    f"{spearman_p:.4g}"
)


# --------------------------------------------------
# Save
# --------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)
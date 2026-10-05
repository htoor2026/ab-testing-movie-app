from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


# ============================================================
# Configuration
# ============================================================

DATA_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

X_PREDICTIONS_FILE = Path(
    "analysis/outputs/x_learner_predictions.csv"
)

MDE = 0.01
ALPHA = 0.05

# Business guardrail:
# maximum acceptable additional recommendation latency
MAX_LATENCY_INCREASE_MS = 40


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_FILE)


# ============================================================
# Helper: A/B effect
# ============================================================

def calculate_effect(data):

    control = data[
        data["treatment"] == 0
    ]

    treatment = data[
        data["treatment"] == 1
    ]

    control_ctr = control["clicked"].mean()
    treatment_ctr = treatment["clicked"].mean()

    lift = treatment_ctr - control_ctr

    successes = np.array([
        treatment["clicked"].sum(),
        control["clicked"].sum(),
    ])

    nobs = np.array([
        len(treatment),
        len(control),
    ])

    _, p_value = proportions_ztest(
        successes,
        nobs,
    )

    return {
        "control_ctr": control_ctr,
        "treatment_ctr": treatment_ctr,
        "lift": lift,
        "p_value": p_value,
    }


# ============================================================
# Overall experiment
# ============================================================

overall = calculate_effect(df)


# ============================================================
# Segment effects
# ============================================================

segment_results = []

for column in ["user_type", "device"]:

    for value in df[column].unique():

        subset = df[
            df[column] == value
        ]

        result = calculate_effect(subset)

        result["segment"] = column
        result["group"] = value

        segment_results.append(result)


segments = pd.DataFrame(segment_results)


# Multiple-testing correction
reject, adjusted_p, _, _ = multipletests(
    segments["p_value"],
    alpha=ALPHA,
    method="holm",
)

segments["adjusted_p"] = adjusted_p
segments["significant"] = reject


# ============================================================
# Returning / new users
# ============================================================

returning = segments[
    (segments["segment"] == "user_type")
    &
    (segments["group"] == "returning")
].iloc[0]


new_users = segments[
    (segments["segment"] == "user_type")
    &
    (segments["group"] == "new")
].iloc[0]


# ============================================================
# Interaction test
# ============================================================

df["returning"] = (
    df["user_type"] == "returning"
).astype(int)


interaction_model = smf.ols(
    """
    clicked ~ treatment
            + returning
            + treatment:returning
    """,
    data=df,
).fit(
    cov_type="HC3"
)


interaction_effect = (
    interaction_model.params[
        "treatment:returning"
    ]
)

interaction_p = (
    interaction_model.pvalues[
        "treatment:returning"
    ]
)


# ============================================================
# Latency guardrail
# ============================================================

control_latency = (
    df[df["treatment"] == 0]
    ["latency_ms"]
    .mean()
)

treatment_latency = (
    df[df["treatment"] == 1]
    ["latency_ms"]
    .mean()
)

latency_increase = (
    treatment_latency
    - control_latency
)

latency_pass = (
    latency_increase
    <= MAX_LATENCY_INCREASE_MS
)


# ============================================================
# ML targeting quality
# ============================================================

x_predictions = pd.read_csv(
    X_PREDICTIONS_FILE
)

cate_mae = np.mean(
    np.abs(
        x_predictions["predicted_cate"]
        -
        x_predictions[
            "true_individual_uplift"
        ]
    )
)

spearman = (
    x_predictions[
        [
            "predicted_cate",
            "true_individual_uplift",
        ]
    ]
    .corr(method="spearman")
    .iloc[0, 1]
)


# ============================================================
# Decision logic
# ============================================================

overall_business_success = (
    overall["p_value"] < ALPHA
    and
    overall["lift"] >= MDE
)

returning_success = (
    returning["adjusted_p"] < ALPHA
    and
    returning["lift"] >= MDE
    and
    interaction_p < ALPHA
    and
    latency_pass
)


# Be deliberately conservative.
ml_targeting_ready = (
    spearman >= 0.50
    and
    cate_mae <= 0.02
)


# ============================================================
# Report
# ============================================================

print("=" * 75)
print("FINAL BUSINESS DECISION")
print("=" * 75)


print("\nOVERALL EXPERIMENT")

print(
    f"Control CTR: "
    f"{overall['control_ctr']:.3%}"
)

print(
    f"Treatment CTR: "
    f"{overall['treatment_ctr']:.3%}"
)

print(
    f"Absolute lift: "
    f"{overall['lift'] * 100:.3f} pp"
)

print(
    f"P-value: "
    f"{overall['p_value']:.4f}"
)


print("\nRETURNING USERS")

print(
    f"Lift: "
    f"{returning['lift'] * 100:.3f} pp"
)

print(
    f"Adjusted p-value: "
    f"{returning['adjusted_p']:.4f}"
)


print("\nNEW USERS")

print(
    f"Lift: "
    f"{new_users['lift'] * 100:.3f} pp"
)

print(
    f"Adjusted p-value: "
    f"{new_users['adjusted_p']:.4f}"
)


print("\nHETEROGENEOUS EFFECT")

print(
    f"Treatment × returning interaction: "
    f"{interaction_effect * 100:.3f} pp"
)

print(
    f"Interaction p-value: "
    f"{interaction_p:.4f}"
)


print("\nLATENCY GUARDRAIL")

print(
    f"Control latency: "
    f"{control_latency:.1f} ms"
)

print(
    f"Treatment latency: "
    f"{treatment_latency:.1f} ms"
)

print(
    f"Increase: "
    f"{latency_increase:.1f} ms"
)


print("\nUPLIFT MODEL")

print(
    f"X-Learner CATE MAE: "
    f"{cate_mae * 100:.3f} pp"
)

print(
    f"X-Learner Spearman: "
    f"{spearman:.3f}"
)


print("\n" + "=" * 75)
print("DECISION")
print("=" * 75)


if overall_business_success:
    print(
        "\n✅ Full rollout is supported."
    )
else:
    print(
        "\n❌ Do NOT roll personalization "
        "out to all users."
    )


if returning_success:

    print(
        "\n✅ Roll personalization out "
        "to RETURNING users."
    )

else:

    print(
        "\n❌ Returning-user rollout "
        "is not sufficiently supported."
    )


print(
    "\nKeep NEW users on the "
    "popularity-based recommender."
)


if ml_targeting_ready:

    print(
        "\n✅ ML-based individual targeting "
        "is sufficiently reliable."
    )

else:

    print(
        "\n❌ Do NOT use individual-level "
        "ML targeting yet."
    )

    print(
        "Continue collecting experimental "
        "data and improving CATE estimation."
    )


print("\n" + "=" * 75)
print("RECOMMENDED PRODUCT POLICY")
print("=" * 75)

print(
    """
NEW USER
    → Popularity recommender

RETURNING USER
    → Personalized recommender

Individual uplift model
    → Research / monitoring only
    → Not production decision logic yet
"""
)
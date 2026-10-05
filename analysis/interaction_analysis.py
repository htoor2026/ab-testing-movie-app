from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

ALPHA = 0.05


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# --------------------------------------------------
# Create binary user-type indicator
# --------------------------------------------------

df["returning"] = (
    df["user_type"] == "returning"
).astype(int)


# --------------------------------------------------
# Interaction model
# --------------------------------------------------

model = smf.ols(
    formula="""
        clicked ~ treatment
                + returning
                + treatment:returning
    """,
    data=df,
).fit(
    cov_type="HC3"
)


# --------------------------------------------------
# Extract coefficients
# --------------------------------------------------

control_new_ctr = model.params["Intercept"]

treatment_effect_new = model.params[
    "treatment"
]

baseline_returning_difference = model.params[
    "returning"
]

interaction_effect = model.params[
    "treatment:returning"
]


# Treatment effect for returning users
treatment_effect_returning = (
    treatment_effect_new
    + interaction_effect
)


# CTR estimates
treatment_new_ctr = (
    control_new_ctr
    + treatment_effect_new
)

control_returning_ctr = (
    control_new_ctr
    + baseline_returning_difference
)

treatment_returning_ctr = (
    control_returning_ctr
    + treatment_effect_returning
)


# --------------------------------------------------
# Interaction significance
# --------------------------------------------------

interaction_p = model.pvalues[
    "treatment:returning"
]

interaction_ci = model.conf_int().loc[
    "treatment:returning"
]


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 70)
print("TREATMENT × USER TYPE INTERACTION ANALYSIS")
print("=" * 70)


print("\nEstimated CTRs:")

print(
    f"New users - Control: "
    f"{control_new_ctr:.3%}"
)

print(
    f"New users - Treatment: "
    f"{treatment_new_ctr:.3%}"
)

print(
    f"Returning users - Control: "
    f"{control_returning_ctr:.3%}"
)

print(
    f"Returning users - Treatment: "
    f"{treatment_returning_ctr:.3%}"
)


print("\nTreatment effects:")

print(
    f"New users: "
    f"{treatment_effect_new * 100:.3f} "
    "percentage points"
)

print(
    f"Returning users: "
    f"{treatment_effect_returning * 100:.3f} "
    "percentage points"
)


print("\nDifference in treatment effects:")

print(
    f"Interaction effect: "
    f"{interaction_effect * 100:.3f} "
    "percentage points"
)

print(
    f"95% CI: "
    f"[{interaction_ci.iloc[0] * 100:.3f}, "
    f"{interaction_ci.iloc[1] * 100:.3f}] "
    "percentage points"
)

print(
    f"P-value: "
    f"{interaction_p:.4f}"
)


# --------------------------------------------------
# Interpretation
# --------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if interaction_p < ALPHA:

    if interaction_effect > 0:

        print(
            "\n✅ Significant positive interaction."
        )

        print(
            "Personalization has a significantly larger "
            "effect for returning users than for new users."
        )

    else:

        print(
            "\n✅ Significant negative interaction."
        )

        print(
            "Personalization has a significantly smaller "
            "effect for returning users than for new users."
        )

else:

    print(
        "\n❌ Interaction is not statistically significant."
    )

    print(
        "There is insufficient evidence that the treatment "
        "effect differs between new and returning users."
    )


# --------------------------------------------------
# Full regression output
# --------------------------------------------------

print("\n" + "=" * 70)
print("FULL MODEL")
print("=" * 70)

print(model.summary())
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("simulation/data/experiment_outcomes.csv")

ALPHA = 0.05

# Minimum business-relevant improvement
MDE = 0.01   # 1 percentage point


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

control = df[df["variant"] == "control"]
treatment = df[df["variant"] == "treatment"]


# --------------------------------------------------
# Sample sizes
# --------------------------------------------------

n_control = len(control)
n_treatment = len(treatment)

clicks_control = control["clicked"].sum()
clicks_treatment = treatment["clicked"].sum()


# --------------------------------------------------
# CTR
# --------------------------------------------------

ctr_control = clicks_control / n_control
ctr_treatment = clicks_treatment / n_treatment


# --------------------------------------------------
# Treatment effect
# --------------------------------------------------

absolute_lift = ctr_treatment - ctr_control

relative_lift = (
    absolute_lift / ctr_control
)


# --------------------------------------------------
# Two-proportion Z-test
# --------------------------------------------------

successes = np.array(
    [clicks_treatment, clicks_control]
)

observations = np.array(
    [n_treatment, n_control]
)

z_stat, p_value = proportions_ztest(
    count=successes,
    nobs=observations,
    alternative="two-sided",
)


# --------------------------------------------------
# 95% CI for treatment - control difference
# --------------------------------------------------

standard_error = np.sqrt(
    (
        ctr_treatment
        * (1 - ctr_treatment)
        / n_treatment
    )
    +
    (
        ctr_control
        * (1 - ctr_control)
        / n_control
    )
)

z_critical = 1.96

ci_lower = (
    absolute_lift
    - z_critical * standard_error
)

ci_upper = (
    absolute_lift
    + z_critical * standard_error
)


# --------------------------------------------------
# Statistical significance
# --------------------------------------------------

statistically_significant = (
    p_value < ALPHA
)


# --------------------------------------------------
# Practical significance
# --------------------------------------------------

practically_significant = (
    absolute_lift >= MDE
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("A/B TEST ANALYSIS")
print("=" * 60)

print("\nSample sizes:")
print(f"Control:   {n_control:,}")
print(f"Treatment: {n_treatment:,}")

print("\nClicks:")
print(f"Control:   {clicks_control:,}")
print(f"Treatment: {clicks_treatment:,}")

print("\nCTR:")
print(f"Control CTR:   {ctr_control:.3%}")
print(f"Treatment CTR: {ctr_treatment:.3%}")

print("\nTreatment effect:")
print(
    f"Absolute lift: "
    f"{absolute_lift * 100:.3f} percentage points"
)

print(
    f"Relative lift: "
    f"{relative_lift * 100:.2f}%"
)

print("\nHypothesis test:")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

print("\n95% Confidence Interval:")
print(
    f"[{ci_lower * 100:.3f}, "
    f"{ci_upper * 100:.3f}] "
    "percentage points"
)

print("\nStatistical significance:")

if statistically_significant:
    print("✅ Statistically significant at alpha = 0.05")
else:
    print("❌ Not statistically significant at alpha = 0.05")


print("\nPractical significance:")
print(
    f"Pre-defined MDE: "
    f"{MDE * 100:.2f} percentage point"
)

if practically_significant:
    print("✅ Observed lift meets/exceeds the MDE.")
else:
    print("❌ Observed lift is below the MDE.")


# --------------------------------------------------
# Preliminary decision
# --------------------------------------------------

print("\n" + "=" * 60)
print("PRELIMINARY INTERPRETATION")
print("=" * 60)

if statistically_significant and practically_significant:

    print(
        "Treatment is both statistically and "
        "practically significant."
    )

elif statistically_significant and not practically_significant:

    print(
        "Treatment is statistically significant, "
        "but the effect is smaller than the "
        "pre-defined practical threshold."
    )

elif not statistically_significant and practically_significant:

    print(
        "Observed effect is practically interesting, "
        "but evidence is not statistically conclusive."
    )

else:

    print(
        "Treatment is neither statistically nor "
        "practically significant."
    )
from pathlib import Path

import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("simulation/data/experiment_outcomes.csv")

ALPHA = 0.05
POWER = 0.80

# Minimum effect we care about detecting
MDE_ABSOLUTE = 0.01   # +1 percentage point


# --------------------------------------------------
# Load experiment data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

control = df[df["variant"] == "control"]

baseline_ctr = control["clicked"].mean()

target_ctr = baseline_ctr + MDE_ABSOLUTE


# --------------------------------------------------
# Effect size
# --------------------------------------------------

effect_size = proportion_effectsize(
    baseline_ctr,
    target_ctr,
)


# --------------------------------------------------
# Required sample size
# --------------------------------------------------

analysis = NormalIndPower()

required_per_group = analysis.solve_power(
    effect_size=abs(effect_size),
    alpha=ALPHA,
    power=POWER,
    ratio=1,
    alternative="two-sided",
)

required_per_group = int(required_per_group) + 1

required_total = required_per_group * 2


# --------------------------------------------------
# Current sample size
# --------------------------------------------------

control_n = len(
    df[df["variant"] == "control"]
)

treatment_n = len(
    df[df["variant"] == "treatment"]
)

current_total = len(df)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("POWER ANALYSIS")
print("=" * 60)

print(f"\nBaseline CTR: {baseline_ctr:.4%}")
print(f"MDE: +{MDE_ABSOLUTE:.2%} absolute")
print(f"Target CTR: {target_ctr:.4%}")

print(f"\nAlpha: {ALPHA}")
print(f"Power: {POWER}")

print(f"\nRequired users per group: {required_per_group:,}")
print(f"Required total users: {required_total:,}")

print("\nCurrent experiment:")
print(f"Control: {control_n:,}")
print(f"Treatment: {treatment_n:,}")
print(f"Total: {current_total:,}")

if (
    control_n >= required_per_group
    and treatment_n >= required_per_group
):
    print("\n Required sample size reached.")
else:
    missing = required_total - current_total

    print("\n Required sample size NOT reached.")

    if missing > 0:
        print(f"Approximately {missing:,} additional users needed.")
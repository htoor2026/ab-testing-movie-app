from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("simulation/data/experiment_users.csv")

EXPECTED_CONTROL_SHARE = 0.50
EXPECTED_TREATMENT_SHARE = 0.50

ALPHA = 0.05

CATEGORICAL_COLUMNS = [
    "user_type",
    "device",
    "engagement_level",
    "preferred_genre",
]

NUMERIC_COLUMNS = [
    "account_age_days",
    "previous_sessions",
    "movies_rated",
    "avg_session_minutes",
]


# --------------------------------------------------
# 1. Sample Ratio Mismatch
# --------------------------------------------------

def check_srm(df: pd.DataFrame) -> None:
    """
    Check whether the observed control/treatment split
    is significantly different from the planned 50/50 split.
    """

    counts = (
        df["variant"]
        .value_counts()
        .reindex(["control", "treatment"])
    )

    observed = counts.values

    expected = np.array(
        [
            len(df) * EXPECTED_CONTROL_SHARE,
            len(df) * EXPECTED_TREATMENT_SHARE,
        ]
    )

    chi2_stat, p_value = chisquare(
        f_obs=observed,
        f_exp=expected,
    )

    print("\n" + "=" * 60)
    print("1. SAMPLE RATIO MISMATCH (SRM)")
    print("=" * 60)

    print(f"\nObserved:")
    print(f"Control:   {observed[0]:,}")
    print(f"Treatment: {observed[1]:,}")

    print(f"\nExpected:")
    print(f"Control:   {expected[0]:,.0f}")
    print(f"Treatment: {expected[1]:,.0f}")

    print(f"\nChi-square statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < ALPHA:
        print("\n❌ SRM DETECTED")
        print(
            "Observed allocation differs significantly "
            "from the planned 50/50 split."
        )
    else:
        print("\n✅ No evidence of SRM")
        print("The experiment allocation looks consistent with 50/50 randomization.")


# --------------------------------------------------
# 2. Categorical Covariate Balance
# --------------------------------------------------

def check_categorical_balance(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Use chi-square tests to check whether categorical
    pre-treatment variables are associated with treatment.
    """

    print("\n" + "=" * 60)
    print("2. CATEGORICAL COVARIATE BALANCE")
    print("=" * 60)

    results = []

    for column in columns:

        table = pd.crosstab(
            df["variant"],
            df[column],
        )

        chi2_stat, p_value, dof, expected = chi2_contingency(table)

        # Cramer's V = strength of association
        n = table.to_numpy().sum()

        min_dim = min(table.shape) - 1

        if min_dim > 0:
            cramers_v = np.sqrt(
                chi2_stat / (n * min_dim)
            )
        else:
            cramers_v = 0

        results.append(
            {
                "variable": column,
                "chi2": chi2_stat,
                "p_value": p_value,
                "cramers_v": cramers_v,
            }
        )

    results_df = pd.DataFrame(results)

    print(
        results_df.round(
            {
                "chi2": 3,
                "p_value": 4,
                "cramers_v": 4,
            }
        ).to_string(index=False)
    )

    print(
        "\nInterpretation:"
        "\n- p < 0.05 may indicate imbalance."
        "\n- Cramer's V measures how large that association is."
        "\n- Values close to 0 indicate strong balance."
    )


# --------------------------------------------------
# 3. Standardized Mean Difference
# --------------------------------------------------

def standardized_mean_difference(
    control: pd.Series,
    treatment: pd.Series,
) -> float:
    """
    Calculate standardized mean difference (SMD).

    SMD around 0 means the groups are similar.
    """

    mean_control = control.mean()
    mean_treatment = treatment.mean()

    var_control = control.var(ddof=1)
    var_treatment = treatment.var(ddof=1)

    pooled_sd = np.sqrt(
        (var_control + var_treatment) / 2
    )

    if pooled_sd == 0:
        return 0.0

    return (
        mean_treatment - mean_control
    ) / pooled_sd


# --------------------------------------------------
# 4. Numeric Covariate Balance
# --------------------------------------------------

def check_numeric_balance(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    print("\n" + "=" * 60)
    print("3. NUMERIC COVARIATE BALANCE")
    print("=" * 60)

    control = df[df["variant"] == "control"]
    treatment = df[df["variant"] == "treatment"]

    results = []

    for column in columns:

        smd = standardized_mean_difference(
            control[column],
            treatment[column],
        )

        results.append(
            {
                "variable": column,
                "control_mean": control[column].mean(),
                "treatment_mean": treatment[column].mean(),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )

    results_df = pd.DataFrame(results)

    print(
        results_df.round(
            {
                "control_mean": 3,
                "treatment_mean": 3,
                "smd": 4,
                "abs_smd": 4,
            }
        ).to_string(index=False)
    )

    print(
        "\nRule of thumb:"
        "\n|SMD| < 0.10 → good balance"
        "\n|SMD| 0.10–0.20 → investigate"
        "\n|SMD| > 0.20 → concerning imbalance"
    )


# --------------------------------------------------
# 5. Overall Validation
# --------------------------------------------------

def overall_validation(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("4. BASIC DATA QUALITY")
    print("=" * 60)

    duplicate_users = df["user_id"].duplicated().sum()
    missing_values = df.isnull().sum().sum()

    print(f"\nRows: {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Duplicate users: {duplicate_users}")
    print(f"Missing values: {missing_values}")

    if duplicate_users == 0 and missing_values == 0:
        print("\n Basic dataset validation passed.")
    else:
        print("\n Dataset quality issue detected.")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(INPUT_FILE)

    overall_validation(df)

    check_srm(df)

    check_categorical_balance(
        df,
        CATEGORICAL_COLUMNS,
    )

    check_numeric_balance(
        df,
        NUMERIC_COLUMNS,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT VALIDATION COMPLETE")
    print("=" * 60)
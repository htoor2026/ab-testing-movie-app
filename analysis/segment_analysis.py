from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

ALPHA = 0.05
MDE = 0.01   # 1 percentage point


# --------------------------------------------------
# Analyze one segment
# --------------------------------------------------

def analyze_segment(
    df: pd.DataFrame,
    segment_column: str,
    segment_value: str,
) -> dict:

    segment_df = df[
        df[segment_column] == segment_value
    ]

    control = segment_df[
        segment_df["variant"] == "control"
    ]

    treatment = segment_df[
        segment_df["variant"] == "treatment"
    ]

    n_control = len(control)
    n_treatment = len(treatment)

    clicks_control = control["clicked"].sum()
    clicks_treatment = treatment["clicked"].sum()

    ctr_control = (
        clicks_control / n_control
    )

    ctr_treatment = (
        clicks_treatment / n_treatment
    )

    absolute_lift = (
        ctr_treatment - ctr_control
    )

    relative_lift = (
        absolute_lift / ctr_control
        if ctr_control > 0
        else np.nan
    )

    # ----------------------------------------------
    # Z-test
    # ----------------------------------------------

    successes = np.array(
        [
            clicks_treatment,
            clicks_control,
        ]
    )

    observations = np.array(
        [
            n_treatment,
            n_control,
        ]
    )

    z_stat, p_value = proportions_ztest(
        count=successes,
        nobs=observations,
        alternative="two-sided",
    )

    # ----------------------------------------------
    # Confidence interval for difference
    # ----------------------------------------------

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

    ci_lower = (
        absolute_lift
        - 1.96 * standard_error
    )

    ci_upper = (
        absolute_lift
        + 1.96 * standard_error
    )

    return {
        "segment": segment_column,
        "group": segment_value,

        "control_n": n_control,
        "treatment_n": n_treatment,

        "control_ctr": ctr_control,
        "treatment_ctr": ctr_treatment,

        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,

        "ci_lower": ci_lower,
        "ci_upper": ci_upper,

        "z_stat": z_stat,
        "p_value": p_value,
    }


# --------------------------------------------------
# Run segment analysis
# --------------------------------------------------

def run_segment_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    results = []

    segment_columns = [
        "user_type",
        "device",
    ]

    for column in segment_columns:

        values = (
            df[column]
            .dropna()
            .unique()
        )

        for value in values:

            result = analyze_segment(
                df,
                column,
                value,
            )

            results.append(result)

    results_df = pd.DataFrame(results)

    # ----------------------------------------------
    # Multiple testing correction
    # ----------------------------------------------

    reject, adjusted_p, _, _ = multipletests(
        results_df["p_value"],
        alpha=ALPHA,
        method="holm",
    )

    results_df["adjusted_p"] = adjusted_p

    results_df["significant_after_correction"] = (
        reject
    )

    results_df["practically_significant"] = (
        results_df["absolute_lift"]
        >= MDE
    )

    return results_df


# --------------------------------------------------
# Display
# --------------------------------------------------

def print_results(
    results: pd.DataFrame,
) -> None:

    display = results.copy()

    percentage_columns = [
        "control_ctr",
        "treatment_ctr",
        "absolute_lift",
        "relative_lift",
        "ci_lower",
        "ci_upper",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print("\n" + "=" * 90)
    print("SEGMENT ANALYSIS")
    print("=" * 90)

    columns = [
        "segment",
        "group",
        "control_n",
        "treatment_n",
        "control_ctr",
        "treatment_ctr",
        "absolute_lift",
        "relative_lift",
        "ci_lower",
        "ci_upper",
        "p_value",
        "adjusted_p",
        "significant_after_correction",
        "practically_significant",
    ]

    print(
        display[columns]
        .round(3)
        .to_string(index=False)
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(INPUT_FILE)

    results = run_segment_analysis(df)

    print_results(results)
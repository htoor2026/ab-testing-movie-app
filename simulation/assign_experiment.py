from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026

INPUT_FILE = Path("simulation/data/synthetic_users.csv")
OUTPUT_FILE = Path("simulation/data/experiment_users.csv")

rng = np.random.default_rng(SEED)


def assign_experiment(users: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly assign users 50/50 to control and treatment.
    """

    users = users.copy()

    users["treatment"] = rng.choice(
        [0, 1],
        size=len(users),
        p=[0.50, 0.50],
    )

    users["variant"] = np.where(
        users["treatment"] == 0,
        "control",
        "treatment",
    )

    return users


def check_randomization(df: pd.DataFrame) -> None:
    """
    Check whether randomization produced approximately balanced groups.
    """

    print("\nExperiment sizes:")
    print(df["variant"].value_counts())

    print("\nExperiment proportions:")
    print(df["variant"].value_counts(normalize=True).round(4))

    print("\nUser type by variant:")
    print(
        pd.crosstab(
            df["variant"],
            df["user_type"],
            normalize="index"
        ).round(3)
    )

    print("\nDevice by variant:")
    print(
        pd.crosstab(
            df["variant"],
            df["device"],
            normalize="index"
        ).round(3)
    )

    print("\nEngagement level by variant:")
    print(
        pd.crosstab(
            df["variant"],
            df["engagement_level"],
            normalize="index"
        ).round(3)
    )

    print("\nNumeric averages:")
    print(
        df.groupby("variant")[
            [
                "account_age_days",
                "previous_sessions",
                "movies_rated",
                "avg_session_minutes",
            ]
        ]
        .mean()
        .round(2)
    )


if __name__ == "__main__":

    users = pd.read_csv(INPUT_FILE)

    experiment_users = assign_experiment(users)

    check_randomization(experiment_users)

    experiment_users.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved to: {OUTPUT_FILE}")
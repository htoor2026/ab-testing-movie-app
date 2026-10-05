from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
N_USERS = 28_000

OUTPUT_DIR = Path("simulation/data")
OUTPUT_FILE = OUTPUT_DIR / "synthetic_users.csv"

rng = np.random.default_rng(SEED)


def generate_users(n_users: int = N_USERS) -> pd.DataFrame:
    """
    Generate a realistic synthetic user population for a
    movie recommendation A/B testing project.

    IMPORTANT:
    This function only creates pre-treatment user attributes.
    It does NOT assign control/treatment or generate outcomes.
    """

    user_ids = [f"user_{i:06d}" for i in range(1, n_users + 1)]

    # -----------------------------
    # User lifecycle
    # -----------------------------
    user_type = rng.choice(
        ["new", "returning"],
        size=n_users,
        p=[0.35, 0.65],
    )

    # -----------------------------
    # Device
    # -----------------------------
    device = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=n_users,
        p=[0.55, 0.38, 0.07],
    )

    # -----------------------------
    # Engagement level
    # -----------------------------
    engagement_level = rng.choice(
        ["low", "medium", "high"],
        size=n_users,
        p=[0.30, 0.50, 0.20],
    )

    # -----------------------------
    # Preferred movie genre
    # -----------------------------
    preferred_genre = rng.choice(
        [
            "action",
            "comedy",
            "drama",
            "thriller",
            "sci-fi",
            "romance",
            "horror",
        ],
        size=n_users,
        p=[0.20, 0.18, 0.18, 0.14, 0.12, 0.10, 0.08],
    )

    # -----------------------------
    # Account age
    # New users should naturally
    # have younger accounts.
    # -----------------------------
    account_age_days = np.where(
        user_type == "new",
        rng.integers(0, 31, size=n_users),
        rng.integers(31, 1500, size=n_users),
    )

    # -----------------------------
    # Previous sessions
    # Correlated with engagement.
    # -----------------------------
    session_lambda = np.select(
        [
            engagement_level == "low",
            engagement_level == "medium",
            engagement_level == "high",
        ],
        [2, 8, 20],
        default=5,
    )

    previous_sessions = rng.poisson(session_lambda)

    # New users shouldn't have huge
    # historical session counts.
    previous_sessions = np.where(
        user_type == "new",
        np.minimum(previous_sessions, 5),
        previous_sessions,
    )

    # -----------------------------
    # Movies previously rated
    # Important for personalization:
    # users with history should benefit
    # more from a recommender later.
    # -----------------------------
    rating_lambda = np.select(
        [
            engagement_level == "low",
            engagement_level == "medium",
            engagement_level == "high",
        ],
        [2, 10, 30],
        default=5,
    )

    movies_rated = rng.poisson(rating_lambda)

    movies_rated = np.where(
        user_type == "new",
        np.minimum(movies_rated, 5),
        movies_rated,
    )

    # -----------------------------
    # Average session duration
    # -----------------------------
    session_duration_mean = np.select(
        [
            engagement_level == "low",
            engagement_level == "medium",
            engagement_level == "high",
        ],
        [5, 14, 28],
        default=10,
    )

    avg_session_minutes = rng.normal(
        loc=session_duration_mean,
        scale=4,
    )

    avg_session_minutes = np.clip(
        avg_session_minutes,
        1,
        60,
    ).round(1)

    # -----------------------------
    # Build dataframe
    # -----------------------------
    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "user_type": user_type,
            "device": device,
            "engagement_level": engagement_level,
            "preferred_genre": preferred_genre,
            "account_age_days": account_age_days,
            "previous_sessions": previous_sessions,
            "movies_rated": movies_rated,
            "avg_session_minutes": avg_session_minutes,
        }
    )

    return users


def validate_users(users: pd.DataFrame) -> None:
    """Basic sanity checks."""

    assert users["user_id"].is_unique
    assert users.isnull().sum().sum() == 0

    assert users["account_age_days"].ge(0).all()
    assert users["previous_sessions"].ge(0).all()
    assert users["movies_rated"].ge(0).all()
    assert users["avg_session_minutes"].gt(0).all()

    print("Validation passed.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    users = generate_users()

    validate_users(users)

    users.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nGenerated {len(users):,} users")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nSample:")
    print(users.head())

    print("\nUser type:")
    print(users["user_type"].value_counts(normalize=True).round(3))

    print("\nDevice:")
    print(users["device"].value_counts(normalize=True).round(3))

    print("\nEngagement:")
    print(
        users["engagement_level"]
        .value_counts(normalize=True)
        .round(3)
    )
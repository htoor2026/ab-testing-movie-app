import pandas as pd


results = pd.DataFrame(
    {
        "model": [
            "T-Learner",
            "X-Learner",
        ],
        "cate_mae_pp": [
            4.004,
            2.640,
        ],
        "pearson": [
            0.212,
            0.338,
        ],
        "spearman": [
            0.184,
            0.281,
        ],
        "uplift_at_10_pp": [
            6.373,
            3.759,
        ],
        "uplift_at_20_pp": [
            4.212,
            2.380,
        ],
        "uplift_at_30_pp": [
            2.554,
            1.884,
        ],
    }
)


print("=" * 80)
print("UPLIFT MODEL COMPARISON")
print("=" * 80)

print(
    results
    .round(3)
    .to_string(index=False)
)


best_mae = results.loc[
    results["cate_mae_pp"].idxmin(),
    "model",
]

best_ranking = results.loc[
    results["spearman"].idxmax(),
    "model",
]


print("\nBest CATE accuracy:")
print(best_mae)

print("\nBest ranking:")
print(best_ranking)


print("\n" + "=" * 80)
print("ROLLOUT RECOMMENDATION")
print("=" * 80)

print(
    """
1. Do NOT roll personalization out to all users.

2. Roll personalization out to RETURNING users.
   The randomized experiment showed a +1.595 percentage-point
   CTR lift for returning users.

3. Keep NEW users on the popularity-based recommender.
   No evidence of benefit was found for new users.

4. X-Learner outperformed T-Learner for heterogeneous
   treatment-effect estimation.

5. However, individual CATE estimation is not accurate enough
   yet for fully ML-driven targeting.

6. Continue collecting experiment data before deploying
   individual-level uplift targeting.
"""
)
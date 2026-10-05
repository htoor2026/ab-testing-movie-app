from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = Path(
    "simulation/data/experiment_outcomes.csv"
)

OUTPUT_DIR = Path("analysis/outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)


# ============================================================
# 1. Overall CTR
# ============================================================

overall = (
    df.groupby("variant")["clicked"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(7, 5))

overall.plot(kind="bar")

plt.ylabel("CTR (%)")
plt.xlabel("")
plt.title("Overall Recommendation CTR")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_ctr.png",
    dpi=150,
)

plt.close()


# ============================================================
# 2. CTR by user type
# ============================================================

user_type_ctr = (
    df.groupby(
        ["user_type", "variant"]
    )["clicked"]
    .mean()
    .mul(100)
    .unstack()
)

plt.figure(figsize=(8, 5))

user_type_ctr.plot(
    kind="bar",
    ax=plt.gca(),
)

plt.ylabel("CTR (%)")
plt.xlabel("User type")

plt.title(
    "Personalization Effect by User Type"
)

plt.xticks(rotation=0)

plt.legend(title="Variant")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ctr_by_user_type.png",
    dpi=150,
)

plt.close()


# ============================================================
# 3. Treatment lift by segment
# ============================================================

segment_results = []

for column in [
    "user_type",
    "device",
]:

    for value in df[column].unique():

        subset = df[
            df[column] == value
        ]

        control_ctr = (
            subset[
                subset["treatment"] == 0
            ]["clicked"]
            .mean()
        )

        treatment_ctr = (
            subset[
                subset["treatment"] == 1
            ]["clicked"]
            .mean()
        )

        lift = (
            treatment_ctr
            - control_ctr
        ) * 100

        segment_results.append(
            {
                "segment":
                    f"{column}: {value}",

                "lift_pp":
                    lift,
            }
        )


segment_df = pd.DataFrame(
    segment_results
)

plt.figure(figsize=(9, 6))

plt.barh(
    segment_df["segment"],
    segment_df["lift_pp"],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Treatment lift (percentage points)"
)

plt.ylabel("")

plt.title(
    "Personalization Treatment Effect by Segment"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "segment_lift.png",
    dpi=150,
)

plt.close()


# ============================================================
# 4. Latency guardrail
# ============================================================

latency = (
    df.groupby("variant")[
        "latency_ms"
    ]
    .mean()
)

plt.figure(figsize=(7, 5))

latency.plot(kind="bar")

plt.ylabel(
    "Average recommendation latency (ms)"
)

plt.xlabel("")

plt.title(
    "Recommendation Latency Guardrail"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "latency_guardrail.png",
    dpi=150,
)

plt.close()


print("Charts saved to:")
print(OUTPUT_DIR)
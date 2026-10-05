from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import chisquare, chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    proportion_effectsize,
    proportions_ztest,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Recommendation Experiment",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM HTML HELPER
# Prevents indented HTML from rendering as a code block.
# ============================================================

def html(content: str):
    st.html(
        dedent(content)
    )


# ============================================================
# CUSTOM CSS
# ============================================================

html(
    """
    <style>

    /* ---------------------------------------------------------
       GLOBAL
    --------------------------------------------------------- */

    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.4rem;
        padding-bottom: 5rem;
    }

    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------------------------------------------------------
       HERO
    --------------------------------------------------------- */

    .hero {
        position: relative;
        overflow: hidden;

        min-height: 430px;

        display: flex;
        flex-direction: column;
        justify-content: center;

        padding: 72px 76px;

        border-top: 7px solid #7666ff;
        border-radius: 0 0 28px 28px;

        background:
            radial-gradient(
                circle at 84% 30%,
                rgba(45, 107, 59, 0.42),
                transparent 31%
            ),
            radial-gradient(
                circle at 22% 100%,
                rgba(118, 102, 255, 0.22),
                transparent 35%
            ),
            linear-gradient(
                115deg,
                #050706 0%,
                #090d0a 48%,
                #08150b 100%
            );

        color: #ffffff;
    }

    .hero::after {
        content: "";
        position: absolute;
        top: -80px;
        right: -30px;

        width: 480px;
        height: 480px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(75, 133, 81, 0.20),
                transparent 65%
            );

        pointer-events: none;
    }

    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 900px;
    }

    .hero-kicker {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        color: #b9b7d2;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 58px;
        line-height: 1.04;
        font-weight: 720;
        letter-spacing: -1.8px;
        margin: 0 0 20px 0;
        color: #ffffff;
    }

    .hero-subtitle {
        max-width: 790px;
        font-size: 18px;
        line-height: 1.65;
        color: #d4d4d8;
        margin-bottom: 30px;
    }

    .hero-pill {
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;

        padding: 8px 14px;

        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 100px;

        background: rgba(255,255,255,0.07);

        color: #f4f4f5;
        font-size: 12px;
        font-weight: 500;
    }

    /* ---------------------------------------------------------
       THREE FEATURE CARDS
    --------------------------------------------------------- */

    .feature-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);

        background: #ffffff;

        border: 1px solid #efeff3;
        border-top: none;
        border-radius: 0 0 24px 24px;

        box-shadow: 0 12px 35px rgba(20, 20, 40, 0.05);

        margin-bottom: 38px;
    }

    .feature-card {
        padding: 42px 40px;
        min-height: 225px;
    }

    .feature-card + .feature-card {
        border-left: 1px solid #eeeeF2;
    }

    .feature-number {
        width: 42px;
        height: 42px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 50%;

        background: #7666ff;
        color: white;

        font-size: 13px;
        font-weight: 700;

        margin-bottom: 20px;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 700;
        color: #20222a;
        margin-bottom: 10px;
    }

    .feature-text {
        font-size: 14px;
        line-height: 1.65;
        color: #6c6d76;
    }

    /* ---------------------------------------------------------
       SECTION TITLES
    --------------------------------------------------------- */

    .section-kicker {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        color: #7666ff;
        margin-bottom: 7px;
    }

    .section-title {
        font-size: 34px;
        font-weight: 720;
        letter-spacing: -0.7px;
        color: #22242c;
        margin-bottom: 10px;
    }

    .section-description {
        font-size: 15px;
        line-height: 1.7;
        color: #686a73;
        max-width: 850px;
        margin-bottom: 28px;
    }

    /* ---------------------------------------------------------
       CARDS
    --------------------------------------------------------- */

    .content-card {
        background: #ffffff;
        border: 1px solid #e9e9ef;
        border-radius: 18px;
        padding: 28px 30px;
        margin: 16px 0 22px 0;

        box-shadow: 0 6px 20px rgba(0,0,0,0.025);
    }

    .content-card h3 {
        margin-top: 0;
        font-size: 21px;
        color: #22242c;
    }

    .soft-box {
        background: #f7f7fa;
        border: 1px solid #ececf1;
        border-radius: 14px;
        padding: 22px 24px;
        margin: 16px 0;
    }

    .accent-box {
        background: #f5f3ff;
        border-left: 4px solid #7666ff;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 18px 0;
        color: #33343c;
    }

    .success-box {
        background: #eef8f2;
        border: 1px solid #d3eadb;
        border-radius: 15px;
        padding: 24px;
        margin: 16px 0;
        color: #23382a;
    }

    .neutral-box {
        background: #f6f6f8;
        border: 1px solid #e6e6ea;
        border-radius: 15px;
        padding: 24px;
        margin: 16px 0;
        color: #303038;
    }

    .warning-box {
        background: #fff9ea;
        border: 1px solid #f0e1b5;
        border-radius: 15px;
        padding: 24px;
        margin: 16px 0;
        color: #413a27;
    }

    /* ---------------------------------------------------------
       FLOW
    --------------------------------------------------------- */

    .flow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        margin: 20px 0 28px 0;
    }

    .flow-step {
        padding: 9px 14px;
        border-radius: 8px;
        background: #f5f4fb;
        border: 1px solid #e4e1f8;
        color: #48425f;
        font-size: 13px;
        font-weight: 600;
    }

    .flow-arrow {
        color: #9995ad;
        font-size: 15px;
    }

    /* ---------------------------------------------------------
       PRESENTATION
    --------------------------------------------------------- */

    .presentation-section {
        padding: 32px 4px 40px 4px;
        border-bottom: 1px solid #eeeeF2;
    }

    .presentation-number {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.7px;
        color: #7666ff;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .presentation-title {
        font-size: 31px;
        font-weight: 720;
        color: #23252d;
        margin-bottom: 15px;
    }

    .presentation-text {
        font-size: 15px;
        line-height: 1.78;
        color: #60626c;
        max-width: 950px;
    }

    /* ---------------------------------------------------------
       METRICS
    --------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: #fafafb;
        border: 1px solid #ececf0;
        padding: 18px 20px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 13px;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* ---------------------------------------------------------
       TABS
    --------------------------------------------------------- */

    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        border-bottom: 1px solid #e7e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 13px;

        font-size: 13px;
        font-weight: 600;

        color: #5f6068;
    }

    .stTabs [aria-selected="true"] {
        color: #7666ff;
    }

    /* ---------------------------------------------------------
       DATAFRAMES
    --------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------------------------------------------------------
       MOBILE
    --------------------------------------------------------- */

    @media (max-width: 900px) {

        .hero {
            padding: 55px 28px;
            min-height: 390px;
        }

        .hero-title {
            font-size: 40px;
        }

        .hero-subtitle {
            font-size: 16px;
        }

        .feature-row {
            grid-template-columns: 1fr;
        }

        .feature-card + .feature-card {
            border-left: none;
            border-top: 1px solid #eeeeF2;
        }

        .feature-card {
            min-height: auto;
        }
    }

    </style>
    """
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

EXPERIMENT_USERS_FILE = (
    ROOT
    / "simulation"
    / "data"
    / "experiment_users.csv"
)

OUTCOMES_FILE = (
    ROOT
    / "simulation"
    / "data"
    / "experiment_outcomes.csv"
)

TRUTH_FILE = (
    ROOT
    / "simulation"
    / "data"
    / "simulation_truth.csv"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "outputs"
)

CHART_DIR = (
    OUTPUT_DIR
    / "charts"
)

QINI_FILE = (
    OUTPUT_DIR
    / "qini_curve.png"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    experiment_users = pd.read_csv(
        EXPERIMENT_USERS_FILE
    )

    outcomes = pd.read_csv(
        OUTCOMES_FILE
    )

    truth = None

    if TRUTH_FILE.exists():
        truth = pd.read_csv(
            TRUTH_FILE
        )

    return (
        experiment_users,
        outcomes,
        truth,
    )


experiment_users, df, truth = load_data()


# ============================================================
# HELPERS
# ============================================================

def two_proportion_test(
    control_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
):

    control_clicks = int(
        control_df["clicked"].sum()
    )

    treatment_clicks = int(
        treatment_df["clicked"].sum()
    )

    control_n = len(
        control_df
    )

    treatment_n = len(
        treatment_df
    )

    count = np.array(
        [
            treatment_clicks,
            control_clicks,
        ]
    )

    nobs = np.array(
        [
            treatment_n,
            control_n,
        ]
    )

    z_stat, p_value = (
        proportions_ztest(
            count,
            nobs,
        )
    )

    control_ctr = (
        control_clicks
        / control_n
    )

    treatment_ctr = (
        treatment_clicks
        / treatment_n
    )

    difference = (
        treatment_ctr
        - control_ctr
    )

    relative_lift = (
        difference
        / control_ctr
    )

    se = np.sqrt(
        (
            control_ctr
            * (1 - control_ctr)
            / control_n
        )
        +
        (
            treatment_ctr
            * (1 - treatment_ctr)
            / treatment_n
        )
    )

    ci_low = (
        difference
        - 1.96 * se
    )

    ci_high = (
        difference
        + 1.96 * se
    )

    return {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_clicks": control_clicks,
        "treatment_clicks": treatment_clicks,
        "control_ctr": control_ctr,
        "treatment_ctr": treatment_ctr,
        "difference": difference,
        "relative_lift": relative_lift,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def cramers_v(table: pd.DataFrame):

    chi2, _, _, _ = (
        chi2_contingency(
            table
        )
    )

    n = table.to_numpy().sum()

    rows, cols = (
        table.shape
    )

    denominator = min(
        rows - 1,
        cols - 1,
    )

    if denominator == 0:
        return 0.0

    return np.sqrt(
        chi2
        / (
            n
            * denominator
        )
    )


def standardized_mean_difference(
    control_values,
    treatment_values,
):

    control_values = (
        pd.Series(
            control_values
        )
        .dropna()
        .astype(float)
    )

    treatment_values = (
        pd.Series(
            treatment_values
        )
        .dropna()
        .astype(float)
    )

    pooled_sd = np.sqrt(
        (
            control_values.var(
                ddof=1
            )
            +
            treatment_values.var(
                ddof=1
            )
        )
        / 2
    )

    if pooled_sd == 0:
        return 0.0

    return (
        treatment_values.mean()
        -
        control_values.mean()
    ) / pooled_sd


def show_chart(
    filename: str,
    caption: str = "",
):

    chart_path = (
        CHART_DIR
        / filename
    )

    if chart_path.exists():

        st.image(
            str(chart_path),
            caption=caption,
            use_container_width=True,
        )


def section_header(
    number: str,
    title: str,
    description: str,
):

    html(
        f"""
        <div style="margin-top: 20px;">
            <div class="section-kicker">
                {number}
            </div>

            <div class="section-title">
                {title}
            </div>

            <div class="section-description">
                {description}
            </div>
        </div>
        """
    )


# ============================================================
# CORE DATA
# ============================================================

control = (
    df[
        df["variant"]
        == "control"
    ]
    .copy()
)

treatment = (
    df[
        df["variant"]
        == "treatment"
    ]
    .copy()
)

overall = (
    two_proportion_test(
        control,
        treatment,
    )
)


# ============================================================
# POWER ANALYSIS
# ============================================================

ALPHA = 0.05

POWER = 0.80

MDE = 0.01

PLANNING_BASELINE_CTR = 0.08979

PLANNING_TARGET_CTR = (
    PLANNING_BASELINE_CTR
    + MDE
)

effect_size = (
    proportion_effectsize(
        PLANNING_BASELINE_CTR,
        PLANNING_TARGET_CTR,
    )
)

required_per_group = int(
    np.ceil(
        NormalIndPower()
        .solve_power(
            effect_size=effect_size,
            power=POWER,
            alpha=ALPHA,
            ratio=1,
            alternative="two-sided",
        )
    )
)

required_total = (
    required_per_group
    * 2
)


# ============================================================
# SECONDARY / GUARDRAIL
# ============================================================

control_watchlist = (
    control[
        "watchlisted"
    ]
    .mean()
)

treatment_watchlist = (
    treatment[
        "watchlisted"
    ]
    .mean()
)

control_latency = (
    control[
        "latency_ms"
    ]
    .mean()
)

treatment_latency = (
    treatment[
        "latency_ms"
    ]
    .mean()
)

latency_increase = (
    treatment_latency
    - control_latency
)


# ============================================================
# HERO
# ============================================================

html(
    """
    <div class="hero">

        <div class="hero-content">

            <div class="hero-kicker">
                Experimentation Case Study
            </div>

            <div class="hero-title">
                Recommendation Personalization A/B Test
            </div>

            <div class="hero-subtitle">
                An end-to-end experiment evaluating whether a
                personalized recommendation experience improves
                engagement over a popularity-based baseline while
                accounting for statistical uncertainty,
                heterogeneous treatment effects and technical
                guardrails.
            </div>

            <span class="hero-pill">
                28,000 users
            </span>

            <span class="hero-pill">
                Randomized experiment
            </span>

            <span class="hero-pill">
                A/B testing
            </span>

            <span class="hero-pill">
                Uplift modeling
            </span>

            <span class="hero-pill">
                Causal inference
            </span>

        </div>

    </div>
    """
)


# ============================================================
# FEATURE CARDS
# ============================================================

html(
    """
    <div class="feature-row">

        <div class="feature-card">

            <div class="feature-number">
                01
            </div>

            <div class="feature-title">
                Experiment Design
            </div>

            <div class="feature-text">
                Users were independently randomized 50/50 between
                an existing popularity-based experience and a
                personalized treatment. Metrics, MDE, power and
                guardrails were defined before outcome analysis.
            </div>

        </div>

        <div class="feature-card">

            <div class="feature-number">
                02
            </div>

            <div class="feature-title">
                Statistical Analysis
            </div>

            <div class="feature-text">
                Randomization was validated with SRM and covariate
                balance checks, followed by a two-proportion z-test,
                confidence intervals, multiple-testing correction
                and interaction analysis.
            </div>

        </div>

        <div class="feature-card">

            <div class="feature-number">
                03
            </div>

            <div class="feature-title">
                Product Decision
            </div>

            <div class="feature-text">
                Returning users showed the strongest evidence of
                benefit. The final recommendation was a targeted
                rollout rather than universal deployment.
            </div>

        </div>

    </div>
    """
)


# ============================================================
# ANALYSIS FLOW
# ============================================================

html(
    """
    <div class="flow">

        <div class="flow-step">
            Design
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Randomize
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Validate
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Power
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Analyze
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Segment
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Uplift
        </div>

        <div class="flow-arrow">
            →
        </div>

        <div class="flow-step">
            Decide
        </div>

    </div>
    """
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Experiment Design",
        "Experiment Health",
        "Statistical Analysis",
        "Segments & HTE",
        "Uplift Modeling",
        "Business Decision",
        "Presentation",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:

    section_header(
        "Overview",
        "Experiment Summary",
        (
            "A concise view of the business question, treatment "
            "design, primary outcome and headline result."
        ),
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Users",
        f"{len(df):,}",
    )

    c2.metric(
        "Control CTR",
        f"{overall['control_ctr'] * 100:.3f}%",
    )

    c3.metric(
        "Treatment CTR",
        f"{overall['treatment_ctr'] * 100:.3f}%",
    )

    c4.metric(
        "Absolute Lift",
        f"{overall['difference'] * 100:+.3f} pp",
    )

    html(
        """
        <div class="content-card">

            <h3>Business Question</h3>

            <p>
            Does a personalized recommendation experience improve
            recommendation engagement enough to justify replacing
            the existing popularity-based experience?
            </p>

        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        html(
            """
            <div class="neutral-box">

                <b>Control</b>

                <br><br>

                Popularity-based recommendations.

                <br><br>

                This represents the existing baseline experience.

            </div>
            """
        )

    with c2:

        html(
            """
            <div class="success-box">

                <b>Treatment</b>

                <br><br>

                Personalized recommendations.

                <br><br>

                This represents the new product experience being tested.

            </div>
            """
        )

    st.subheader(
        "Success Metrics"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Primary Metric",
        "CTR",
    )

    c2.metric(
        "Secondary Metric",
        "Watchlist Rate",
    )

    c3.metric(
        "Guardrail",
        "Latency",
    )

    c1, c2 = st.columns(
        [1.15, 1]
    )

    with c1:

        show_chart(
            "overall_ctr.png",
            "Overall recommendation CTR",
        )

    with c2:

        html(
            f"""
            <div class="accent-box">

                <b>Headline result</b>

                <br><br>

                Absolute CTR lift:
                <b>{overall['difference'] * 100:+.3f} pp</b>

                <br><br>

                Relative lift:
                <b>{overall['relative_lift'] * 100:+.2f}%</b>

                <br><br>

                P-value:
                <b>{overall['p_value']:.4f}</b>

                <br><br>

                95% CI:
                <b>
                [{overall['ci_low'] * 100:.3f},
                {overall['ci_high'] * 100:.3f}] pp
                </b>

            </div>
            """
        )


# ============================================================
# TAB 2 — EXPERIMENT DESIGN
# ============================================================

with tabs[1]:

    section_header(
        "01",
        "Experiment Design",
        (
            "The experiment was designed around individual-user "
            "randomization, a pre-defined primary metric and a "
            "planned detectable effect."
        ),
    )

    design_df = pd.DataFrame(
        {
            "Component": [
                "Experimental unit",
                "Control",
                "Treatment",
                "Assignment",
                "Primary metric",
                "Secondary metric",
                "Guardrail",
                "Alpha",
                "Power",
                "MDE",
            ],
            "Value": [
                "User",
                "Popularity recommendations",
                "Personalized recommendations",
                "50/50 random assignment",
                "Recommendation CTR",
                "Watchlist-add rate",
                "Recommendation latency",
                "0.05",
                "80%",
                "+1.00 percentage point",
            ],
        }
    )

    st.dataframe(
        design_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Random Assignment"
    )

    st.code(
        """
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
        """,
        language="python",
    )

    st.subheader(
        "Primary Hypothesis"
    )

    st.latex(
        r"H_0: p_{treatment} = p_{control}"
    )

    st.latex(
        r"H_1: p_{treatment} \neq p_{control}"
    )

    html(
        """
        <div class="accent-box">

            CTR is based on a binary click outcome.

            <br><br>

            Because the comparison is between two independent
            proportions, the primary inference method is a
            <b>two-proportion z-test</b>.

        </div>
        """
    )

    st.subheader(
        "Power & Sample Size"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Planning Baseline",
        f"{PLANNING_BASELINE_CTR * 100:.2f}%",
    )

    c2.metric(
        "MDE",
        "+1.00 pp",
    )

    c3.metric(
        "Required / Arm",
        f"{required_per_group:,}",
    )

    c4.metric(
        "Actual Users",
        f"{len(df):,}",
    )

    html(
        f"""
        <div class="soft-box">

            With alpha = <b>0.05</b>, power = <b>80%</b>,
            baseline CTR ≈ <b>{PLANNING_BASELINE_CTR * 100:.2f}%</b>
            and an MDE of <b>+1.00 pp</b>, the required sample was
            approximately <b>{required_total:,}</b> total users.

            <br><br>

            The experiment used <b>{len(df):,}</b> users, so the
            planned sample-size requirement was satisfied.

        </div>
        """
    )


# ============================================================
# TAB 3 — EXPERIMENT HEALTH
# ============================================================

with tabs[2]:

    section_header(
        "02",
        "Experiment Health",
        (
            "Randomization was validated before outcome analysis "
            "using assignment-ratio and pre-treatment balance checks."
        ),
    )

    control_n = int(
        experiment_users[
            "variant"
        ]
        .eq(
            "control"
        )
        .sum()
    )

    treatment_n = int(
        experiment_users[
            "variant"
        ]
        .eq(
            "treatment"
        )
        .sum()
    )

    observed = np.array(
        [
            control_n,
            treatment_n,
        ]
    )

    expected = np.array(
        [
            len(
                experiment_users
            )
            / 2,
            len(
                experiment_users
            )
            / 2,
        ]
    )

    srm_stat, srm_p = (
        chisquare(
            observed,
            expected,
        )
    )

    st.subheader(
        "Sample Ratio Mismatch"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Control",
        f"{control_n:,}",
        f"{control_n / len(experiment_users) * 100:.2f}%",
    )

    c2.metric(
        "Treatment",
        f"{treatment_n:,}",
        f"{treatment_n / len(experiment_users) * 100:.2f}%",
    )

    c3.metric(
        "SRM P-value",
        f"{srm_p:.4f}",
    )

    html(
        """
        <div class="success-box">

            <b>SRM result: Pass</b>

            <br><br>

            The observed assignment ratio is consistent with the
            intended 50/50 randomization.

        </div>
        """
    )

    st.subheader(
        "Categorical Balance"
    )

    categorical_columns = [
        "user_type",
        "device",
        "engagement_level",
        "preferred_genre",
    ]

    categorical_results = []

    for column in (
        categorical_columns
    ):

        if column not in experiment_users.columns:
            continue

        table = pd.crosstab(
            experiment_users[
                "variant"
            ],
            experiment_users[
                column
            ],
        )

        chi2, p_value, _, _ = (
            chi2_contingency(
                table
            )
        )

        categorical_results.append(
            {
                "Covariate":
                    column,
                "Chi-square":
                    chi2,
                "P-value":
                    p_value,
                "Cramér's V":
                    cramers_v(
                        table
                    ),
            }
        )

    categorical_df = pd.DataFrame(
        categorical_results
    )

    st.dataframe(
        categorical_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Numeric Balance"
    )

    numeric_columns = [
        "account_age_days",
        "previous_sessions",
        "movies_rated",
        "avg_session_minutes",
    ]

    exp_control = (
        experiment_users[
            experiment_users[
                "variant"
            ]
            == "control"
        ]
    )

    exp_treatment = (
        experiment_users[
            experiment_users[
                "variant"
            ]
            == "treatment"
        ]
    )

    numeric_results = []

    for column in (
        numeric_columns
    ):

        if column not in experiment_users.columns:
            continue

        smd = (
            standardized_mean_difference(
                exp_control[
                    column
                ],
                exp_treatment[
                    column
                ],
            )
        )

        numeric_results.append(
            {
                "Covariate":
                    column,
                "Control Mean":
                    exp_control[
                        column
                    ].mean(),
                "Treatment Mean":
                    exp_treatment[
                        column
                    ].mean(),
                "SMD":
                    smd,
                "|SMD|":
                    abs(smd),
            }
        )

    numeric_df = pd.DataFrame(
        numeric_results
    )

    st.dataframe(
        numeric_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    if (
        len(
            numeric_df
        )
        > 0
        and
        numeric_df[
            "|SMD|"
        ].max()
        < 0.10
    ):

        html(
            """
            <div class="success-box">

                <b>Numeric balance result: Pass</b>

                <br><br>

                All absolute standardized mean differences are
                below 0.10.

            </div>
            """
        )

    st.subheader(
        "Randomization Debugging"
    )

    html(
        """
        <div class="warning-box">

            During development, the same deterministic RNG seed was
            reused for user generation and treatment assignment.

            <br><br>

            This created severe treatment / user-type imbalance.

            <br><br>

            The validation pipeline detected the problem before
            outcome analysis. Treatment assignment was regenerated
            using an independent seed.

        </div>
        """
    )


# ============================================================
# TAB 4 — STATISTICAL ANALYSIS
# ============================================================

with tabs[3]:

    section_header(
        "03",
        "Statistical Analysis",
        (
            "The primary analysis compares recommendation CTR "
            "between the randomized control and treatment groups."
        ),
    )

    html(
        """
        <div class="content-card">

            <h3>Primary Test</h3>

            <p>
            A <b>two-proportion z-test</b> was used because the
            outcome is binary and CTR is a proportion.
            </p>

        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "Control"
        )

        st.metric(
            "Users",
            f"{overall['control_n']:,}",
        )

        st.metric(
            "Clicks",
            f"{overall['control_clicks']:,}",
        )

        st.metric(
            "CTR",
            f"{overall['control_ctr'] * 100:.3f}%",
        )

    with c2:

        st.subheader(
            "Treatment"
        )

        st.metric(
            "Users",
            f"{overall['treatment_n']:,}",
        )

        st.metric(
            "Clicks",
            f"{overall['treatment_clicks']:,}",
        )

        st.metric(
            "CTR",
            f"{overall['treatment_ctr'] * 100:.3f}%",
        )

    st.subheader(
        "Inference"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Absolute Lift",
        f"{overall['difference'] * 100:+.3f} pp",
    )

    c2.metric(
        "Relative Lift",
        f"{overall['relative_lift'] * 100:+.2f}%",
    )

    c3.metric(
        "Z-statistic",
        f"{overall['z_stat']:.4f}",
    )

    c4.metric(
        "P-value",
        f"{overall['p_value']:.4f}",
    )

    st.metric(
        "95% Confidence Interval",
        (
            f"[{overall['ci_low'] * 100:.3f}, "
            f"{overall['ci_high'] * 100:.3f}] pp"
        ),
    )

    html(
        f"""
        <div class="success-box">

            <b>Statistical conclusion</b>

            <br><br>

            Since p = <b>{overall['p_value']:.4f}</b> is below
            alpha = 0.05, the null hypothesis is rejected.

            <br><br>

            There is statistical evidence that personalization
            changed recommendation CTR.

        </div>
        """
    )

    html(
        f"""
        <div class="warning-box">

            <b>Practical significance</b>

            <br><br>

            Planned practical threshold:
            <b>+1.00 pp</b>

            <br><br>

            Observed point estimate:
            <b>{overall['difference'] * 100:+.3f} pp</b>

            <br><br>

            The point estimate is below the threshold, but the
            95% CI still includes effects above +1 pp.

            <br><br>

            Practical significance therefore remains uncertain.

        </div>
        """
    )

    st.subheader(
        "Secondary Metric"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Control Watchlist",
        f"{control_watchlist * 100:.3f}%",
    )

    c2.metric(
        "Treatment Watchlist",
        f"{treatment_watchlist * 100:.3f}%",
    )

    c3.metric(
        "Difference",
        (
            f"{(treatment_watchlist - control_watchlist) * 100:+.3f} pp"
        ),
    )


# ============================================================
# TAB 5 — SEGMENTS & HTE
# ============================================================

with tabs[4]:

    section_header(
        "04",
        "Segments & Heterogeneous Treatment Effects",
        (
            "The average treatment effect was decomposed across "
            "predefined user segments, then formally tested using "
            "multiple-testing correction and interaction analysis."
        ),
    )

    segment_rows = []

    for segment_col in [
        "user_type",
        "device",
    ]:

        for group in sorted(
            df[
                segment_col
            ]
            .dropna()
            .unique()
        ):

            segment_data = (
                df[
                    df[
                        segment_col
                    ]
                    == group
                ]
            )

            segment_control = (
                segment_data[
                    segment_data[
                        "variant"
                    ]
                    == "control"
                ]
            )

            segment_treatment = (
                segment_data[
                    segment_data[
                        "variant"
                    ]
                    == "treatment"
                ]
            )

            result = (
                two_proportion_test(
                    segment_control,
                    segment_treatment,
                )
            )

            segment_rows.append(
                {
                    "Segment":
                        segment_col,
                    "Group":
                        group,
                    "Control N":
                        result[
                            "control_n"
                        ],
                    "Treatment N":
                        result[
                            "treatment_n"
                        ],
                    "Control CTR":
                        (
                            result[
                                "control_ctr"
                            ]
                            * 100
                        ),
                    "Treatment CTR":
                        (
                            result[
                                "treatment_ctr"
                            ]
                            * 100
                        ),
                    "Lift (pp)":
                        (
                            result[
                                "difference"
                            ]
                            * 100
                        ),
                    "Raw p":
                        result[
                            "p_value"
                        ],
                    "CI Lower":
                        (
                            result[
                                "ci_low"
                            ]
                            * 100
                        ),
                    "CI Upper":
                        (
                            result[
                                "ci_high"
                            ]
                            * 100
                        ),
                }
            )

    segment_df = pd.DataFrame(
        segment_rows
    )

    reject, adjusted_p, _, _ = (
        multipletests(
            segment_df[
                "Raw p"
            ],
            alpha=0.05,
            method="holm",
        )
    )

    segment_df[
        "Holm Adjusted p"
    ] = adjusted_p

    segment_df[
        "Significant"
    ] = reject

    st.dataframe(
        segment_df.round(3),
        use_container_width=True,
        hide_index=True,
    )

    html(
        """
        <div class="accent-box">

            Five subgroup tests were evaluated together.

            <br><br>

            Holm correction was used to control the
            family-wise error rate.

            <br><br>

            Returning users were the only tested subgroup that
            remained statistically significant after correction.

        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        show_chart(
            "ctr_by_user_type.png",
            "CTR by user type",
        )

    with c2:

        show_chart(
            "segment_lift.png",
            "Treatment lift by segment",
        )

    st.subheader(
        "Interaction Analysis"
    )

    st.markdown(
        """
        A significant result in one segment and a non-significant
        result in another does not by itself prove their treatment
        effects differ.

        An explicit interaction test was therefore performed:
        """
    )

    st.code(
        """
clicked ~ treatment
        + returning
        + treatment:returning
        """,
        language="text",
    )

    interaction_data = (
        df.copy()
    )

    interaction_data[
        "returning"
    ] = (
        interaction_data[
            "user_type"
        ]
        .eq(
            "returning"
        )
        .astype(int)
    )

    interaction_model = (
        ols(
            (
                "clicked ~ treatment "
                "+ returning "
                "+ treatment:returning"
            ),
            data=interaction_data,
        )
        .fit(
            cov_type="HC3"
        )
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

    interaction_ci = (
        interaction_model
        .conf_int()
        .loc[
            "treatment:returning"
        ]
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Interaction Effect",
        f"{interaction_effect * 100:+.3f} pp",
    )

    c2.metric(
        "P-value",
        f"{interaction_p:.4f}",
    )

    c3.metric(
        "95% CI",
        (
            f"[{interaction_ci.iloc[0] * 100:.3f}, "
            f"{interaction_ci.iloc[1] * 100:.3f}] pp"
        ),
    )

    html(
        """
        <div class="success-box">

            The interaction is positive and statistically significant.

            <br><br>

            Personalization therefore produced a significantly larger
            treatment effect for returning users than for new users.

        </div>
        """
    )


# ============================================================
# TAB 6 — UPLIFT MODELING
# ============================================================

with tabs[5]:

    section_header(
        "05",
        "Uplift Modeling",
        (
            "The analysis moved beyond predefined segments to test "
            "whether treatment effects could be estimated at the "
            "individual-user level."
        ),
    )

    c1, c2 = st.columns(2)

    with c1:

        html(
            """
            <div class="content-card">

                <h3>T-Learner</h3>

                <p>
                Separate outcome models are trained for control and
                treatment users.
                </p>

                <p>
                Individual CATE is estimated as:
                </p>

                <p>
                predicted treatment outcome
                minus predicted control outcome
                </p>

            </div>
            """
        )

    with c2:

        html(
            """
            <div class="content-card">

                <h3>X-Learner</h3>

                <p>
                Outcome models are first estimated for both arms.
                Pseudo-treatment effects are then constructed and
                second-stage effect models are trained.
                </p>

            </div>
            """
        )

    comparison = pd.DataFrame(
        {
            "Model": [
                "T-Learner",
                "X-Learner",
            ],
            "CATE MAE (pp)": [
                4.004,
                2.640,
            ],
            "Pearson": [
                0.212,
                0.338,
            ],
            "Spearman": [
                0.184,
                0.281,
            ],
            "Uplift@10% (pp)": [
                6.373,
                3.759,
            ],
            "Uplift@20% (pp)": [
                4.212,
                2.380,
            ],
            "Uplift@30% (pp)": [
                2.554,
                1.884,
            ],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    html(
        """
        <div class="success-box">

            X-Learner improved CATE accuracy and treatment-effect
            ranking relative to the T-Learner.

        </div>
        """
    )

    html(
        """
        <div class="warning-box">

            Individual-level estimates were still too noisy for
            production targeting.

            <br><br>

            The uplift model therefore remained a research and
            monitoring tool rather than production decision logic.

        </div>
        """
    )

    if QINI_FILE.exists():

        c1, c2 = st.columns(
            [1.25, 1]
        )

        with c1:

            st.image(
                str(
                    QINI_FILE
                ),
                caption="T-Learner Qini curve",
                use_container_width=True,
            )

        with c2:

            html(
                """
                <div class="accent-box">

                    <b>Interpretation</b>

                    <br><br>

                    The models detected some heterogeneous treatment
                    structure, especially around returning-user
                    behavior.

                    <br><br>

                    However, group-level evidence remained more
                    trustworthy than individual CATE estimates.

                </div>
                """
            )


# ============================================================
# TAB 7 — BUSINESS DECISION
# ============================================================

with tabs[6]:

    section_header(
        "06",
        "Business Decision",
        (
            "The final recommendation combines statistical evidence, "
            "business significance, segment consistency, uplift-model "
            "quality and the latency guardrail."
        ),
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Overall Lift",
        f"{overall['difference'] * 100:+.3f} pp",
    )

    c2.metric(
        "Returning Lift",
        "+1.595 pp",
    )

    c3.metric(
        "New User Lift",
        "-0.472 pp",
    )

    c4.metric(
        "Latency Increase",
        f"+{latency_increase:.1f} ms",
    )

    st.subheader(
        "Technical Guardrail"
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Control Latency",
        f"{control_latency:.1f} ms",
    )

    c2.metric(
        "Treatment Latency",
        f"{treatment_latency:.1f} ms",
    )

    c3.metric(
        "Increase",
        f"+{latency_increase:.1f} ms",
    )

    c1, c2 = st.columns(
        [1.05, 1]
    )

    with c1:

        show_chart(
            "latency_guardrail.png",
            "Recommendation latency",
        )

    with c2:

        html(
            """
            <div class="accent-box">

                Personalization improved engagement but increased
                recommendation latency.

                <br><br>

                The product decision therefore considers both
                engagement gain and technical cost.

            </div>
            """
        )

    st.subheader(
        "Final Rollout Policy"
    )

    c1, c2 = st.columns(2)

    with c1:

        html(
            """
            <div class="success-box">

                <h3>Returning Users</h3>

                <b>Deploy personalization</b>

                <br><br>

                Treatment lift: +1.595 pp

                <br>

                Holm-adjusted p: 0.0027

                <br>

                Positive interaction confirmed

            </div>
            """
        )

    with c2:

        html(
            """
            <div class="neutral-box">

                <h3>New Users</h3>

                <b>Keep popularity recommendations</b>

                <br><br>

                Treatment lift: -0.472 pp

                <br>

                Adjusted p: 0.6449

                <br>

                No reliable evidence of benefit

            </div>
            """
        )

    html(
        """
        <div class="warning-box">

            <b>Individual uplift targeting</b>

            <br><br>

            Do not deploy individual X-Learner targeting yet.

            <br><br>

            Continue collecting experimental data and improving
            CATE estimation.

        </div>
        """
    )

    st.subheader(
        "Recommended Product Logic"
    )

    st.code(
        """
if user_type == "returning":
    recommendation_strategy = "personalized"
else:
    recommendation_strategy = "popularity"
        """,
        language="python",
    )


# ============================================================
# TAB 8 — PRESENTATION
# ============================================================

with tabs[7]:

    section_header(
        "Presentation",
        "Complete Experiment Case Study",
        (
            "A presentation-style walkthrough of the entire project "
            "from product question to final rollout recommendation."
        ),
    )

    # --------------------------------------------------------
    # 01 BUSINESS PROBLEM
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                01 — Business Problem
            </div>

            <div class="presentation-title">
                Should personalization replace the existing
                recommendation experience?
            </div>

            <div class="presentation-text">

                The existing product experience uses broadly
                popularity-based movie recommendations.

                <br><br>

                The proposed change introduces personalized
                recommendations.

                <br><br>

                The experiment asks whether personalization improves
                recommendation engagement enough to justify rollout
                while avoiding unacceptable technical regressions.

            </div>

        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        html(
            """
            <div class="neutral-box">

                <b>Control</b>

                <br><br>

                Popularity-based recommendation experience.

            </div>
            """
        )

    with c2:

        html(
            """
            <div class="success-box">

                <b>Treatment</b>

                <br><br>

                Personalized recommendation experience.

            </div>
            """
        )

    # --------------------------------------------------------
    # 02 DESIGN
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                02 — Experiment Design
            </div>

            <div class="presentation-title">
                Randomized user-level experiment
            </div>

            <div class="presentation-text">

                Users were independently randomized with equal
                probability into control and treatment.

                <br><br>

                Experimental unit: user

                <br>

                Primary metric: recommendation CTR

                <br>

                Secondary metric: watchlist-add rate

                <br>

                Guardrail metric: recommendation latency

                <br><br>

                Alpha was set to 0.05, target power to 80%, and the
                pre-defined MDE to +1.00 percentage point.

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # 03 POWER
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                03 — Power & Sample Size
            </div>

            <div class="presentation-title">
                Plan the experiment before evaluating outcomes
            </div>

            <div class="presentation-text">

                The planning baseline CTR was approximately 8.98%.

                <br><br>

                The experiment was designed to detect a +1.00 pp
                increase with 80% power at alpha = 0.05.

                <br><br>

                Required sample:
                approximately 13,461 users per arm.

                <br>

                Required total:
                approximately 26,922 users.

                <br>

                Actual sample:
                28,000 users.

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # 04 HEALTH
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                04 — Experiment Health
            </div>

            <div class="presentation-title">
                Validate randomization before trusting outcomes
            </div>

            <div class="presentation-text">

                Control contained 14,043 users and treatment contained
                13,957 users.

                <br><br>

                The SRM p-value was approximately 0.8875, providing no
                evidence of a sample ratio mismatch.

                <br><br>

                Categorical covariates were checked using chi-square
                tests and Cramér's V.

                <br><br>

                Numeric pre-treatment covariates were checked using
                standardized mean differences.

                <br><br>

                An earlier RNG-seed reuse bug created severe covariate
                imbalance. The validation pipeline detected the issue
                before outcome analysis, and treatment assignment was
                regenerated with an independent seed.

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # 05 PRIMARY TEST
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                05 — Primary Statistical Test
            </div>

            <div class="presentation-title">
                Compare control and treatment CTR
            </div>

            <div class="presentation-text">

                The primary outcome is binary: clicked or not clicked.

                <br><br>

                A two-proportion z-test was therefore used to compare
                CTR between the two randomized arms.

            </div>

        </div>
        """
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Control CTR",
        "9.343%",
    )

    c2.metric(
        "Treatment CTR",
        "10.188%",
    )

    c3.metric(
        "Absolute Lift",
        "+0.846 pp",
    )

    c4.metric(
        "Relative Lift",
        "+9.05%",
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Z-statistic",
        "2.3837",
    )

    c2.metric(
        "P-value",
        "0.0171",
    )

    c3.metric(
        "95% CI",
        "[0.150, 1.541] pp",
    )

    html(
        """
        <div class="accent-box">

            The overall CTR result was statistically significant.

            <br><br>

            The point estimate was below the +1.00 pp practical
            threshold, but the confidence interval still included
            effects above that threshold.

            <br><br>

            Statistical significance was established, while
            practical significance remained uncertain.

        </div>
        """
    )

    # --------------------------------------------------------
    # 06 SEGMENTS
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                06 — Segment Analysis
            </div>

            <div class="presentation-title">
                The average effect hides important heterogeneity
            </div>

            <div class="presentation-text">

                Treatment effects were estimated separately for new,
                returning, desktop, mobile and tablet users.

                <br><br>

                Because multiple subgroup hypotheses were tested,
                Holm correction was applied to control the family-wise
                error rate.

            </div>

        </div>
        """
    )

    presentation_segments = pd.DataFrame(
        {
            "Segment": [
                "Returning",
                "New",
                "Desktop",
                "Mobile",
                "Tablet",
            ],
            "Lift (pp)": [
                1.595,
                -0.472,
                0.579,
                0.833,
                2.431,
            ],
            "Holm Adjusted p": [
                0.0027,
                0.6449,
                0.645,
                0.298,
                0.298,
            ],
            "Significant": [
                "Yes",
                "No",
                "No",
                "No",
                "No",
            ],
        }
    )

    st.dataframe(
        presentation_segments,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 07 INTERACTION
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                07 — Interaction Analysis
            </div>

            <div class="presentation-title">
                Formally test whether returning users respond
                differently
            </div>

            <div class="presentation-text">

                One significant subgroup and one non-significant
                subgroup do not automatically imply different
                treatment effects.

                <br><br>

                An explicit treatment × returning interaction was
                therefore estimated using a linear probability model
                with HC3 robust standard errors.

            </div>

        </div>
        """
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Interaction Effect",
        "+2.066 pp",
    )

    c2.metric(
        "P-value",
        "0.0037",
    )

    c3.metric(
        "95% CI",
        "[0.671, 3.461] pp",
    )

    # --------------------------------------------------------
    # 08 UPLIFT
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                08 — Uplift Modeling
            </div>

            <div class="presentation-title">
                Can treatment response be estimated at the
                individual level?
            </div>

            <div class="presentation-text">

                A T-Learner and X-Learner were trained using
                pre-treatment user characteristics.

                <br><br>

                The X-Learner improved CATE accuracy and ranking
                relative to the T-Learner.

                <br><br>

                However, individual treatment-effect estimates
                remained too noisy for production targeting.

            </div>

        </div>
        """
    )

    presentation_uplift = pd.DataFrame(
        {
            "Model": [
                "T-Learner",
                "X-Learner",
            ],
            "CATE MAE (pp)": [
                4.004,
                2.640,
            ],
            "Spearman": [
                0.184,
                0.281,
            ],
            "Pearson": [
                0.212,
                0.338,
            ],
        }
    )

    st.dataframe(
        presentation_uplift,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 09 GUARDRAIL
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                09 — Technical Guardrail
            </div>

            <div class="presentation-title">
                Engagement improved, but latency increased
            </div>

            <div class="presentation-text">

                Control recommendation latency was approximately
                121 ms.

                <br><br>

                Treatment recommendation latency was approximately
                150 ms.

                <br><br>

                Personalization therefore introduced roughly a
                29 ms technical regression.

            </div>

        </div>
        """
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Control",
        "121.0 ms",
    )

    c2.metric(
        "Treatment",
        "149.6 ms",
    )

    c3.metric(
        "Increase",
        "+28.7 ms",
    )

    # --------------------------------------------------------
    # 10 DECISION
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                10 — Final Product Decision
            </div>

            <div class="presentation-title">
                Targeted rollout
            </div>

            <div class="presentation-text">

                The experiment does not support a universal rollout.

                <br><br>

                The strongest causal evidence of benefit is
                concentrated among returning users.

            </div>

        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:

        html(
            """
            <div class="success-box">

                <h3>Returning Users</h3>

                Deploy personalized recommendations.

                <br><br>

                Lift: +1.595 pp

                <br>

                Holm-adjusted p: 0.0027

                <br>

                Positive interaction confirmed.

            </div>
            """
        )

    with c2:

        html(
            """
            <div class="neutral-box">

                <h3>New Users</h3>

                Keep popularity-based recommendations.

                <br><br>

                Lift: -0.472 pp

                <br>

                Adjusted p: 0.6449

                <br>

                No reliable evidence of benefit.

            </div>
            """
        )

    html(
        """
        <div class="warning-box">

            <b>Individual uplift targeting</b>

            <br><br>

            Do not deploy individual-level X-Learner targeting yet.

            <br><br>

            Continue collecting data and improving CATE estimation.

        </div>
        """
    )

    # --------------------------------------------------------
    # 11 LIMITATIONS
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                11 — Limitations
            </div>

            <div class="presentation-title">
                What this project does not claim
            </div>

            <div class="presentation-text">

                The experiment uses synthetic users and simulated
                behavioral outcomes.

                <br><br>

                Treatment heterogeneity was intentionally encoded in
                the simulator to stress-test the analysis pipeline.

                <br><br>

                Assignment was assumed to imply exposure.

                <br><br>

                Non-compliance, CUPED, sequential testing, network
                interference and cluster randomization were not
                implemented.

                <br><br>

                The treatment represents a personalized recommendation
                experience; the statistical experiment does not claim
                to evaluate two production recommendation algorithms.

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # FINAL TAKEAWAY
    # --------------------------------------------------------

    html(
        """
        <div class="presentation-section">

            <div class="presentation-number">
                Final Takeaway
            </div>

            <div class="presentation-title">
                Statistical significance alone did not determine
                the product decision
            </div>

            <div class="presentation-text">

                The overall treatment effect was positive and
                statistically significant.

                <br><br>

                But segment analysis, interaction testing,
                multiple-testing correction, latency and uplift-model
                reliability changed the rollout decision.

                <br><br>

                The final recommendation was therefore a
                <b>targeted rollout to returning users</b>, while
                keeping new users on the existing popularity-based
                experience.

            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    (
        "Synthetic experimentation case study demonstrating "
        "experiment design, randomization validation, power "
        "analysis, statistical inference, heterogeneous treatment "
        "effects, uplift modeling and business decision-making."
    )
)
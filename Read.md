Experiment Validation and Randomization Bug

During the initial experiment setup, I found a serious randomization issue before analyzing any treatment outcomes.

The synthetic user generator and the treatment-assignment script initially used the same random seed (42). Because both scripts initialized NumPy's random-number generator from the same deterministic state, the generated user_type variable became unintentionally correlated with treatment assignment.

The initial balance check exposed the problem:

             New    Returning
Control      71%       29%
Treatment     0%      100%

This also produced major differences in pre-treatment characteristics:

                     Control    Treatment
Account age          233 days    769 days
Previous sessions       5.39        8.55
Movies rated            6.27       11.54

Analyzing outcomes under this assignment would have produced a confounded experiment. Any apparent improvement in the treatment group could have been caused by its much larger share of established users rather than by personalized recommendations.

I corrected the issue by using an independent random seed for treatment assignment and reran the randomization.

After correction:

Group sizes
Control:     9,990
Treatment:  10,010

New users
Control:     35.0%
Treatment:   35.5%

Returning users
Control:     65.0%
Treatment:   64.5%

Numeric pre-treatment characteristics were also closely balanced:

                     Control    Treatment
Account age           502.98      502.49
Previous sessions       6.92        7.04
Movies rated            8.89        8.95
Session duration        14.14       14.26

This validation step demonstrates why random assignment should never simply be assumed to have worked. Treatment and control groups should be checked for sample allocation and pre-treatment covariate balance before treatment effects are interpreted.

A strong interview explanation would be:

“Before analyzing outcomes, I validated the randomization and discovered an accidental dependency caused by deterministic RNG reuse. The treatment group contained only returning users, which would have severely confounded the treatment effect. I corrected the assignment mechanism and revalidated covariate balance before proceeding.”

# Movie Recommendation A/B Testing & Uplift Modeling

An end-to-end product experimentation project evaluating whether personalized movie recommendations should replace a popularity-based recommendation strategy.

The project goes beyond a basic A/B test by covering:

- experiment design and randomization
- Sample Ratio Mismatch (SRM) detection
- covariate balance validation
- power analysis
- statistical and practical significance
- confidence intervals
- segment-level treatment effects
- multiple-testing correction
- treatment × segment interaction analysis
- heterogeneous treatment effect estimation
- T-Learner and X-Learner uplift modeling
- Qini / Uplift@K analysis
- product rollout decision-making

> **Important:** The experimentation pipeline is real, but user traffic and behavioral outcomes are synthetically generated. No claim is made that 28,000 real users participated in this experiment.

---

## Business Question

Should a movie-streaming product replace its popularity-based recommendation system with personalized recommendations?

### Control

Users receive:

> Popular / trending movie recommendations.

### Treatment

Users receive:

> Personalized recommendations based on user history and characteristics.

The core causal question is:

> **Does personalization cause users to engage more with movie recommendations?**

---

# Experiment Design

### Experimental unit

One user.

### Allocation

50% Control / 50% Treatment.

### Primary Metric

**Recommendation Click-Through Rate (CTR)**

```text
CTR = users who clicked / users exposed
Secondary Metric

Watchlist-add rate

Guardrail Metric

Recommendation latency

Personalization may improve engagement while also requiring additional computation.

Synthetic User Population

A synthetic population was generated with realistic pre-treatment characteristics:

new vs returning user
mobile / desktop / tablet
engagement level
preferred genre
account age
previous sessions
movies previously rated
average session duration

These characteristics are generated before treatment assignment.

This allows the project to test whether randomization successfully balances relevant user characteristics across experiment groups.

Randomization Validation

An early version of the experiment exposed an important randomization bug.

The synthetic-user generator and treatment-assignment script originally initialized NumPy with the same random seed.

This unintentionally created a strong relationship between user_type and treatment assignment:

             New    Returning
Control      71%       29%
Treatment     0%      100%

Treatment users also had much greater account history:

                     Control    Treatment
Account age          233 days    769 days
Previous sessions       5.39        8.55
Movies rated            6.27       11.54

Analyzing treatment effects under this assignment would have produced a confounded experiment.

The treatment-assignment RNG was separated from the population-generation RNG and randomization was repeated.

After correction:

Control:     14,043
Treatment:   13,957

Pre-treatment characteristics were well balanced.

Example standardized mean differences:

account_age_days       |SMD| = 0.0010
previous_sessions      |SMD| = 0.0189
movies_rated           |SMD| = 0.0065
avg_session_minutes    |SMD| = 0.0136

All were far below the commonly used |SMD| < 0.10 balance threshold.

Sample Ratio Mismatch

The planned allocation was:

50% Control
50% Treatment

SRM was tested using a chi-square goodness-of-fit test.

Earlier validation produced:

Control:    9,990
Treatment: 10,010

p-value = 0.8875

No evidence of Sample Ratio Mismatch was detected.

Power Analysis

Before interpreting treatment results, the experiment was designed around:

Baseline CTR ≈ 8.98%
Minimum Detectable Effect = +1 percentage point
Alpha = 0.05
Power = 80%

Required sample size:

13,461 users per group
26,922 users total

The first simulated run contained only:

20,000 users

and was therefore underpowered for the pre-specified effect.

The experiment was extended to:

28,000 users

before final analysis.

Overall A/B Test

Final experiment:

Control users:   14,043
Treatment users: 13,957

Results:

Metric	Control	Treatment
CTR	9.343%	10.188%
Absolute lift	—	+0.846 pp
Relative lift	—	+9.05%

Statistical test:

Z-statistic = 2.3837
p-value = 0.0171

95% confidence interval for the treatment effect:

[+0.150, +1.541] percentage points
Statistical significance

Yes.

p = 0.0171 < 0.05
Practical significance

No.

The pre-defined minimum practically meaningful improvement was:

+1.00 percentage point

while the observed effect was:

+0.846 percentage points

Therefore:

The overall experiment is statistically significant, but the observed improvement does not reach the pre-defined business threshold.

Segment Analysis
Returning Users
Control CTR:   10.015%
Treatment CTR: 11.610%

Absolute lift: +1.595 pp
Relative lift: +15.9%

Adjusted p-value: 0.0027

This effect is both:

statistically significant
practically significant
New Users
Control CTR:   8.079%
Treatment CTR: 7.607%

Absolute lift: -0.472 pp
Adjusted p-value: 0.6449

There is no evidence that personalization improves engagement for new users.

This is consistent with a potential cold-start problem: new users have limited behavioral history available for personalization.

Treatment × User-Type Interaction

Separate significant/non-significant segment results are not enough to prove treatment effects differ.

An interaction model was therefore estimated:

clicked ~ treatment
        + returning
        + treatment × returning

Result:

Treatment × Returning interaction:
+2.066 percentage points

95% CI:
[+0.671, +3.461]

p-value = 0.0037

Therefore:

Personalization has a statistically significantly larger effect for returning users than for new users.

Guardrail: Recommendation Latency

Personalization introduces additional computational cost.

Control latency:   121.3 ms
Treatment latency: 149.1 ms

Increase: +27.8 ms

The project therefore evaluates treatment using both:

Engagement benefit
        +
System-performance cost

rather than optimizing CTR alone.

Heterogeneous Treatment Effects

The average experiment effect does not tell us whether every user benefits equally.

Two uplift approaches were tested:

T-Learner
X-Learner

The objective was to estimate:

CATE(x)
=
P(click | treatment, X)
-
P(click | control, X)

where X represents user characteristics.

T-Learner

The T-Learner trains two outcome models:

Model 1:
P(click | Control, X)

Model 2:
P(click | Treatment, X)

Estimated CATE is:

Treatment prediction - Control prediction

Performance:

CATE MAE:             4.004 pp
Pearson correlation:  0.212
Spearman correlation: 0.184

Although the highest-ranked users demonstrated strong observed uplift, individual CATE estimates were poorly calibrated.

X-Learner

The X-Learner first estimates potential outcomes and then explicitly models imputed treatment effects.

Performance:

CATE MAE:             2.640 pp
Pearson correlation:  0.338
Spearman correlation: 0.281

The X-Learner outperformed the T-Learner on both:

CATE estimation error
treatment-benefit ranking

However, ranking quality remained insufficient for production-level individual targeting.

Uplift@K

X-Learner targeting results:

Top 10% predicted beneficiaries:
+3.759 pp observed uplift

Top 20%:
+2.380 pp

Top 30%:
+1.884 pp

The highest-ranked X-Learner decile consisted largely of returning users with substantial historical movie activity:

Returning-user share: 97.4%
Average movies rated: 20.2

True simulated uplift:     +2.47 pp
Observed experiment uplift: +3.76 pp

This indicates that the uplift model identified useful signal.

However, individual-level ranking remained noisy.

Model Comparison
Metric	T-Learner	X-Learner
CATE MAE	4.004 pp	2.640 pp
Pearson correlation	0.212	0.338
Spearman correlation	0.184	0.281
Uplift@10%	6.373 pp	3.759 pp
Uplift@20%	4.212 pp	2.380 pp
Uplift@30%	2.554 pp	1.884 pp

The X-Learner provides better overall treatment-effect estimation and ranking, despite the T-Learner producing larger observed Uplift@K estimates on this noisy test sample.

Final Business Decision
❌ Do not ship personalization to everyone

Overall treatment produced:

+0.846 pp CTR

which was statistically significant but below the pre-defined:

+1.00 pp practical threshold
✅ Ship personalization to returning users

Returning users showed:

+1.595 pp CTR
Adjusted p = 0.0027

The treatment × returning-user interaction was also significant.

❌ Do not personalize new users yet

New users showed:

-0.472 pp

with no statistically significant benefit.

They should remain on the popularity-based recommendation strategy.

❌ Do not use individual ML uplift targeting yet

Although the X-Learner outperformed the T-Learner:

Spearman correlation = 0.281
CATE MAE = 2.640 pp

The model is not sufficiently reliable for individual production decisions.

More experimental data and improved causal models would be required.

Recommended Product Policy
                  User arrives
                       |
                New or returning?
                  /           \
               New           Returning
                |               |
         Popularity-based    Personalized
          recommender        recommender

Individual uplift predictions remain a research/monitoring tool rather than production decision logic.

Visualizations
Overall CTR

Treatment Effect by User Type

Segment Treatment Effects

Latency Guardrail

Qini Curve

Project Structure
netflix/
│
├── src/
├── public/
├── package.json
│
├── simulation/
│   ├── generate_users.py
│   ├── assign_experiment.py
│   ├── validate_experiment.py
│   ├── simulate_behavior.py
│   │
│   └── data/
│       ├── synthetic_users.csv
│       ├── experiment_users.csv
│       ├── experiment_outcomes.csv
│       └── simulation_truth.csv
│
├── analysis/
│   ├── power_analysis.py
│   ├── ab_test_analysis.py
│   ├── segment_analysis.py
│   ├── interaction_analysis.py
│   ├── uplift_modeling.py
│   ├── evaluate_uplift.py
│   ├── x_learner.py
│   ├── model_comparison.py
│   ├── final_decision.py
│   ├── create_charts.py
│   │
│   └── outputs/
│
└── README.md
Running the Experiment

Create and activate the environment:

conda create -n ab-testing python=3.11 -y
conda activate ab-testing

Install analysis dependencies:

pip install numpy pandas scipy statsmodels scikit-learn matplotlib

Run the pipeline:

python simulation/generate_users.py
python simulation/assign_experiment.py
python simulation/validate_experiment.py
python simulation/simulate_behavior.py

python analysis/power_analysis.py
python analysis/ab_test_analysis.py
python analysis/segment_analysis.py
python analysis/interaction_analysis.py
python analysis/uplift_modeling.py
python analysis/evaluate_uplift.py
python analysis/x_learner.py
python analysis/model_comparison.py
python analysis/final_decision.py
python analysis/create_charts.py
Methodological Notes
Synthetic outcomes

User behavior is simulated.

This project demonstrates experiment design, statistical inference, causal reasoning and uplift-model evaluation under controlled synthetic conditions.

It does not claim that the experiment was conducted on real Netflix users or real customers.

Simulation truth

Because the experiment is synthetic, both potential click probabilities can be stored:

P(click | control)
P(click | treatment)

This allows estimated CATE to be compared against known synthetic treatment effects.

This information is used only for model evaluation and is never given to the treatment-effect models during training.

Current Limitation

The treatment currently represents the simulated behavioral effect of a personalized recommender.

A real recommendation model has not yet been connected to the treatment arm.

A stronger production extension would use:

Control
→ popularity-based recommender

Treatment
→ collaborative filtering / embedding-based recommender

Real recommendation quality
→ user interaction
→ experimentation system
Future Work

Possible extensions include:

connect a real MovieLens recommendation model
integrate GrowthBook for experiment assignment
collect real product telemetry
increase sample size for heterogeneous-treatment estimation
compare causal forests / doubly robust learners
calibrate CATE estimates
test sequential experimentation
use CUPED variance reduction
deploy the recommendation application
monitor experiment metrics in production
Frontend Attribution

The movie-streaming frontend was adapted from the open-source:

sijeeshmiziha/netflix

The frontend serves as the product surface for this experimentation project.

The original frontend should retain any attribution and licensing requirements from its source repository.

The primary work in this project focuses on:

experimentation design
synthetic traffic generation
statistical validation
power analysis
causal inference
uplift modeling
product decision-making
Key Takeaway

A statistically significant experiment does not automatically justify a full rollout.

This project found that personalization improved overall CTR, but the effect was below the pre-defined practical threshold.

Deeper analysis showed that the value was concentrated among returning users.

The final recommendation was therefore:

Personalize returning users, keep new users on the popularity baseline, and do not deploy individual-level uplift targeting until treatment-effect estimation becomes more reliable.


One thing I would **not** do yet is call this a “production A/B testing platform.” That would overstate what you built. Right now, **“End-to-End A/B Testing & Uplift Modeling for Movie Recommendations”** is accurate and stronger because every claim is defensible.

Your README should explicitly say:

This project uses synthetic experimental users and simulated heterogeneous treatment effects. The purpose is to demonstrate end-to-end experimentation, causal inference, subgroup analysis, and uplift modeling rather than evaluate a real production recommendation algorithm.

Also say:

Returning/new-user heterogeneity was intentionally encoded in the simulation to stress-test whether the analytical pipeline could recover meaningful treatment-effect differences.

That disclosure is important.

For overall practical significance, use:

The overall +0.846 pp point estimate was below the pre-defined +1.0 pp practical threshold. However, its 95% confidence interval [0.150, 1.541] pp included effects above the threshold, so practical significance remained uncertain.

Do not just write “not practically significant.”

4. Put the charts in README

You already have:

analysis/outputs/charts/overall_ctr.png
analysis/outputs/charts/ctr_by_user_type.png
analysis/outputs/charts/segment_lift.png
analysis/outputs/charts/latency_guardrail.png
analysis/outputs/qini_curve.png

Use Markdown like:

![Overall CTR](analysis/outputs/charts/overall_ctr.png)

![CTR by User Type](analysis/outputs/charts/ctr_by_user_type.png)

![Segment Lift](analysis/outputs/charts/segment_lift.png)

![Latency Guardrail](analysis/outputs/charts/latency_guardrail.png)

![Qini Curve](analysis/outputs/qini_curve.png)
5. Then initialize Git immediately

Because you got burned once already:

git init
git add .
git status

Make sure .env and node_modules are not staged.

Then:

git commit -m "Complete A/B testing and uplift modeling project"

At that point, the project is complete as an A/B testing / causal inference portfolio project. The next useful thing is polishing README.md, not adding another model.
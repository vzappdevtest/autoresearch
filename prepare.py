"""
One-time data preparation for autoresearch experiments.

Simulates a randomized marketing-campaign A/B/n test and freezes it to disk so
that every experiment is scored against the exact same data. This file is the
fixed harness: the data generating process, the logged experiment, and the
ground-truth evaluation metric all live here and are NOT modified by the agent.

Usage:
    python prepare.py                  # generate + cache the campaign dataset
    python prepare.py --force          # regenerate even if cache exists

Data is stored in ~/.cache/autoresearch/.

--------------------------------------------------------------------------------
The problem
--------------------------------------------------------------------------------
A marketing team ran a randomized experiment. Each customer was shown one of
several campaign variants (chosen uniformly at random), including a "no contact"
control. We logged, per customer: their features, which variant they got, and
the outcome (whether they converted and the revenue). Because assignment was
randomized, this log is an unbiased basis for learning a *targeting policy*.

The recommendation task: given a customer's features, decide which campaign
variant to send them (or none) so as to maximize expected **net profit**
(expected revenue from conversion minus the cost of the contact). Campaign
effects are heterogeneous — a variant that lifts conversion for one segment can
annoy another into never converting — so the best policy is personalized, and
beats any single mass campaign.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42          # fixed DGP seed -> the experiment is identical every run
N_TRAIN = 60_000          # number of logged customers in the training experiment
N_VAL = 20_000            # number of held-out customers used to score policies
FIT_TIME_BUDGET = 120     # seconds; a recommender's fit() must finish within this

# Campaign variants. Index 0 is always the "no contact" control.
CAMPAIGN_NAMES = [
    "no_contact",          # 0: control, costs nothing, no lift
    "email_10pct_off",     # 1
    "email_free_ship",     # 2
    "sms_reminder",        # 3
    "retargeting_ad",      # 4
    "loyalty_points",      # 5
]
N_CAMPAIGNS = len(CAMPAIGN_NAMES)

# Per-contact cost in dollars (control is free). These are known to the business,
# so the recommender is allowed to use them for cost-aware net-value targeting.
CAMPAIGN_COSTS = np.array([0.00, 0.08, 0.08, 0.04, 0.55, 0.30], dtype=np.float64)

# ---------------------------------------------------------------------------
# Configuration / cache paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_experiment.parquet")
VAL_USERS_PATH = os.path.join(DATA_DIR, "val_users.parquet")
VAL_LOG_PATH = os.path.join(DATA_DIR, "val_log.parquet")
TRUTH_PATH = os.path.join(DATA_DIR, "val_truth.npz")   # loaded ONLY by the evaluator

# Human-readable customer features. The first block are numeric, the last two
# are low-cardinality categoricals that get one-hot expanded in build_features().
NUMERIC_FEATURES = [
    "age",              # years
    "tenure_months",    # how long they've been a customer
    "recency_days",     # days since last purchase
    "frequency",        # purchases in last year
    "monetary",         # avg spend per purchase ($)
    "email_opens_30d",  # engagement signal
    "web_visits_30d",   # engagement signal
    "discount_affinity",# latent propensity to respond to discounts [0,1]
]
CATEGORICAL_FEATURES = {
    "region": ["north", "south", "east", "west"],
    "device": ["mobile", "desktop", "tablet"],
}

# ---------------------------------------------------------------------------
# Data generating process (the ground truth — never seen by the recommender)
# ---------------------------------------------------------------------------

def _raw_customers(n, rng):
    """Sample a table of raw, human-readable customer features."""
    age = np.clip(rng.normal(42, 14, n), 18, 90)
    tenure = np.clip(rng.exponential(28, n), 0, 240)
    recency = np.clip(rng.exponential(40, n), 0, 365)
    frequency = rng.poisson(4, n).astype(np.float64)
    monetary = np.clip(rng.lognormal(3.6, 0.6, n), 5, 500)
    email_opens = rng.poisson(3, n).astype(np.float64)
    web_visits = rng.poisson(5, n).astype(np.float64)
    discount_affinity = rng.beta(2, 3, n)
    region = rng.choice(CATEGORICAL_FEATURES["region"], n)
    device = rng.choice(CATEGORICAL_FEATURES["device"], n)
    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "recency_days": recency,
        "frequency": frequency,
        "monetary": monetary,
        "email_opens_30d": email_opens,
        "web_visits_30d": web_visits,
        "discount_affinity": discount_affinity,
        "region": region,
        "device": device,
    })


def build_features(df):
    """
    Turn the raw customer table into a numeric feature matrix for models.

    This is the sanctioned featurization: the recommender is free to build its
    own features from the raw columns, but this gives a clean default. Numeric
    columns pass through; categoricals are one-hot encoded with a fixed column
    order so the matrix layout is stable across train/val.

    Returns (X, feature_names).
    """
    parts = [df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)]
    names = list(NUMERIC_FEATURES)
    for col, levels in CATEGORICAL_FEATURES.items():
        for lvl in levels:
            parts.append((df[col].to_numpy() == lvl).astype(np.float64)[:, None])
            names.append(f"{col}={lvl}")
    return np.concatenate(parts, axis=1), names


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return (X - mu) / sd


def _true_outcome_model(df, rng):
    """
    Compute the ground-truth expected net value V_k(x) for every customer x and
    every campaign k. This is the oracle used only for scoring — it is never
    exposed to the recommender.

    Returns:
        V         : (n, K) expected net profit ($) of each campaign per customer
        p         : (n, K) true conversion probability of each campaign per customer
        margin    : (n,)   profit per conversion ($) for each customer
    """
    X, _ = build_features(df)
    Xs = _standardize(X)
    n, d = Xs.shape

    # The true response surface is deliberately NONLINEAR: campaign effects
    # depend on feature interactions and thresholds, not just a weighted sum.
    # A linear model can only capture part of it, so there is real headroom for
    # better recommenders (interaction features, tree ensembles, X-learners, ...).
    # This nonlinear basis is used ONLY by the ground truth and is never exposed.
    def nonlinear_basis(seed):
        r = np.random.default_rng(seed)
        cols = [Xs]
        cols.append(Xs ** 2)                                  # curvature
        # A handful of random pairwise interactions.
        for _ in range(8):
            i, j = r.integers(0, d), r.integers(0, d)
            cols.append((Xs[:, i] * Xs[:, j])[:, None])
        # Threshold / regime features (segments respond in step changes).
        cols.append((Xs > 0.7).astype(np.float64))
        return np.concatenate(cols, axis=1)

    Z = nonlinear_basis(RANDOM_SEED + 7)
    dz = Z.shape[1]

    def unit(v):
        """Zero-mean, unit-std — lets us control effect magnitude independent of the basis."""
        return (v - v.mean()) / (v.std() + 1e-8)

    # Baseline conversion propensity (campaign 0 / no-contact), ~5-25%.
    wb = rng.normal(0, 1.0, dz)
    base_logit = -2.0 + 0.8 * unit(Z @ wb)
    p0 = _sigmoid(base_logit)

    # Profit per conversion scales with a customer's monetary value.
    monetary = df["monetary"].to_numpy()
    margin = 12.0 + 0.35 * monetary   # ~$14-$190 per conversion

    # Heterogeneous, campaign-specific uplift on the conversion probability.
    # Each campaign responds to a different nonlinear mix of features (its shape
    # is hard for a linear model to capture), but the effect magnitude is
    # normalized so no single campaign trivially dominates. Some campaigns hurt
    # certain segments (negative uplift = annoyance / opt-out). This heterogeneity
    # is exactly what a good personalized policy exploits.
    UPLIFT_STD = 0.11
    p = np.zeros((n, N_CAMPAIGNS))
    p[:, 0] = p0
    for k in range(1, N_CAMPAIGNS):
        wk = rng.normal(0, 1.0, dz)
        base_lift = rng.uniform(0.0, 0.03)              # small, similar across campaigns
        uplift = base_lift + UPLIFT_STD * unit(Z @ wk)  # nonlinear shape, controlled size
        # Discount campaigns work better on discount-affine customers.
        if "off" in CAMPAIGN_NAMES[k] or "ship" in CAMPAIGN_NAMES[k]:
            aff = df["discount_affinity"].to_numpy()
            uplift += 0.06 * (aff - 0.4) + 0.05 * (aff > 0.6)
        # Recency matters: lapsed customers respond more to reminders/retargeting,
        # but only past a threshold (a step change, not a smooth ramp).
        if "reminder" in CAMPAIGN_NAMES[k] or "ad" in CAMPAIGN_NAMES[k]:
            rec = df["recency_days"].to_numpy()
            uplift += 0.05 * (rec > 60) - 0.03 * (rec < 15)
        p[:, k] = np.clip(p0 + uplift, 0.001, 0.98)

    # Expected net profit of showing campaign k = margin * P(convert | k) - cost_k.
    V = margin[:, None] * p - CAMPAIGN_COSTS[None, :]
    return V, p, margin


def _simulate_log(df, p, margin, rng):
    """
    Given true per-campaign conversion probabilities, simulate one randomized
    experiment: assign each customer a campaign uniformly at random, then draw
    their observed conversion and revenue. Returns a logged DataFrame with the
    columns a real marketing team would have (NO ground-truth leakage).
    """
    n = len(df)
    assigned = rng.integers(0, N_CAMPAIGNS, n)
    p_assigned = p[np.arange(n), assigned]
    converted = (rng.random(n) < p_assigned).astype(np.int64)
    # Observed revenue: noisy realization of the per-conversion margin.
    rev_noise = rng.normal(1.0, 0.15, n).clip(0.3, 2.0)
    revenue = converted * margin * rev_noise

    log = df.copy()
    log["campaign"] = assigned
    log["propensity"] = 1.0 / N_CAMPAIGNS   # known randomization probability
    log["converted"] = converted
    log["revenue"] = revenue
    return log

# ---------------------------------------------------------------------------
# Generation entry point
# ---------------------------------------------------------------------------

def generate(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    if (not force and os.path.exists(TRAIN_PATH) and os.path.exists(VAL_USERS_PATH)
            and os.path.exists(TRUTH_PATH)):
        print(f"Data: already generated at {DATA_DIR} (use --force to regenerate)")
        return

    print("Generating synthetic marketing-campaign experiment...")
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    # Draw one big population, then split into a logged training experiment and a
    # held-out set of customers used to score targeting policies.
    df = _raw_customers(N_TRAIN + N_VAL, rng)
    V, p, margin = _true_outcome_model(df, rng)

    train_df = df.iloc[:N_TRAIN].reset_index(drop=True)
    val_df = df.iloc[N_TRAIN:].reset_index(drop=True)
    p_train, margin_train = p[:N_TRAIN], margin[:N_TRAIN]
    V_val, p_val, margin_val = V[N_TRAIN:], p[N_TRAIN:], margin[N_TRAIN:]

    # Training experiment (what the recommender learns from).
    train_log = _simulate_log(train_df, p_train, margin_train, rng)
    train_log.to_parquet(TRAIN_PATH, index=False)

    # Held-out customers to be targeted (features only — the recommender picks a
    # campaign for each of them). Also emit a randomized val log for optional
    # off-policy sanity checks.
    val_df.to_parquet(VAL_USERS_PATH, index=False)
    val_log = _simulate_log(val_df, p_val, margin_val, rng)
    val_log.to_parquet(VAL_LOG_PATH, index=False)

    # Ground-truth value table + reference policy values, used ONLY by the
    # evaluator in this file. Not part of the recommender's inputs.
    oracle_action = V_val.argmax(axis=1)
    oracle_value = V_val[np.arange(len(V_val)), oracle_action].mean()
    per_campaign_mean = V_val.mean(axis=0)
    best_single = int(per_campaign_mean.argmax())
    np.savez_compressed(
        TRUTH_PATH,
        V_val=V_val,
        oracle_value=oracle_value,
        no_contact_value=per_campaign_mean[0],
        best_single_value=per_campaign_mean[best_single],
        best_single_campaign=best_single,
        per_campaign_mean=per_campaign_mean,
    )

    t1 = time.time()
    print(f"Data: generated in {t1 - t0:.1f}s")
    print(f"  train experiment : {len(train_log):,} logged customers -> {TRAIN_PATH}")
    print(f"  val customers    : {len(val_df):,} held-out -> {VAL_USERS_PATH}")
    print(f"  campaigns        : {N_CAMPAIGNS} ({', '.join(CAMPAIGN_NAMES)})")
    print( "  reference policy values ($/customer):")
    print(f"    no-contact (control) : {per_campaign_mean[0]:.4f}")
    print(f"    best single campaign : {per_campaign_mean[best_single]:.4f} "
          f"({CAMPAIGN_NAMES[best_single]})")
    print(f"    oracle (personalized): {oracle_value:.4f}")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by recommend.py)
# ---------------------------------------------------------------------------

def load_train_experiment():
    """
    Load the logged randomized experiment the recommender learns from.

    Returns a DataFrame with the raw feature columns plus:
        campaign    : int  which variant this customer was shown (0..K-1)
        propensity  : float probability that variant was assigned (== 1/K)
        converted   : int  observed conversion (0/1)
        revenue     : float observed revenue ($, 0 if no conversion)
    """
    if not os.path.exists(TRAIN_PATH):
        print("No data found. Run `uv run prepare.py` first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_parquet(TRAIN_PATH)


def load_val_customers():
    """Load the held-out customers to be targeted (raw features only)."""
    if not os.path.exists(VAL_USERS_PATH):
        print("No data found. Run `uv run prepare.py` first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_parquet(VAL_USERS_PATH)


def load_val_log():
    """Load the randomized val log (for optional off-policy sanity checks)."""
    return pd.read_parquet(VAL_LOG_PATH)

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_policy(recommender):
    """
    Score a targeting policy against the frozen ground truth.

    The `recommender` must expose `.recommend(df) -> array[int]`, returning one
    campaign index (0..K-1) per held-out customer. Because we hold the true
    per-customer expected net value of every campaign, the policy value is exact
    and perfectly comparable across experiments (no evaluation noise).

    The headline metric is POLICY_VALUE: expected net profit ($) per targeted
    customer under the recommended assignments. HIGHER IS BETTER.

    Returns a dict of metrics; `policy_value` is the one to optimize.
    """
    truth = np.load(TRUTH_PATH)
    V_val = truth["V_val"]                    # (n_val, K) true expected net value
    n = V_val.shape[0]

    val_customers = load_val_customers()
    actions = np.asarray(recommender.recommend(val_customers)).astype(int)
    assert actions.shape == (n,), f"recommend() must return {n} actions, got {actions.shape}"
    assert actions.min() >= 0 and actions.max() < N_CAMPAIGNS, "action out of range"

    policy_value = float(V_val[np.arange(n), actions].mean())
    oracle_value = float(truth["oracle_value"])
    no_contact_value = float(truth["no_contact_value"])
    best_single_value = float(truth["best_single_value"])

    regret = oracle_value - policy_value
    # Fraction of the achievable personalization gain that was captured.
    denom = oracle_value - best_single_value
    gain_captured = (policy_value - best_single_value) / denom if denom > 1e-9 else 0.0
    lift_vs_best_single = (policy_value / best_single_value - 1.0) * 100 if best_single_value != 0 else 0.0

    # Distribution of recommended actions (useful diagnostic).
    action_counts = np.bincount(actions, minlength=N_CAMPAIGNS)

    return {
        "policy_value": policy_value,           # <- optimize this (higher better)
        "regret": regret,                       # oracle - policy (lower better)
        "oracle_value": oracle_value,
        "no_contact_value": no_contact_value,
        "best_single_value": best_single_value,
        "lift_vs_best_single_pct": lift_vs_best_single,
        "gain_captured_frac": gain_captured,
        "action_counts": action_counts.tolist(),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cache exists")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}\n")
    generate(force=args.force)
    print("\nDone! Ready to run experiments with `uv run recommend.py`.")

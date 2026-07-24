"""
Autoresearch marketing-campaign recommender. Single-file, CPU-only.

This is the ONE file the agent edits. It learns a targeting policy from a logged
randomized campaign experiment and, for each held-out customer, recommends which
campaign variant to send (or none) to maximize expected net profit.

Usage: uv run recommend.py

The baseline below is a deliberately simple "T-learner" (direct method): fit one
outcome model per campaign on the logged data, predict each campaign's expected
revenue for a customer, subtract the campaign's cost, and recommend the variant
with the highest expected net value. Everything here is fair game to improve --
the model class, features, how uplift is estimated, cost handling, calibration,
regularization, ensembling, etc. The only rule is that `evaluate_policy` in
prepare.py stays untouched and your fit() finishes within the time budget.
"""

import time

import numpy as np
from sklearn.linear_model import Ridge

from prepare import (
    FIT_TIME_BUDGET,
    N_CAMPAIGNS,
    CAMPAIGN_NAMES,
    CAMPAIGN_COSTS,
    build_features,
    load_train_experiment,
    load_val_customers,
    evaluate_policy,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

OUTCOME = "revenue"     # what each per-campaign model predicts: "revenue" or "converted"
RIDGE_ALPHA = 10.0      # L2 regularization strength for the per-campaign models
STANDARDIZE = True      # z-score features before fitting (helps linear models)
COST_AWARE = True       # subtract per-contact cost when picking the best campaign

# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class Recommender:
    """
    T-learner targeting policy.

    fit(train_df):   learn one expected-outcome model per campaign from the log.
    recommend(df):   for each customer, pick argmax over campaigns of
                     (predicted revenue - campaign cost).
    """

    def __init__(self):
        self.models = {}          # campaign index -> fitted regressor
        self.feature_mean = None
        self.feature_std = None

    def _featurize(self, df):
        X, _ = build_features(df)
        if STANDARDIZE:
            if self.feature_mean is None:
                self.feature_mean = X.mean(axis=0)
                self.feature_std = X.std(axis=0) + 1e-8
            X = (X - self.feature_mean) / self.feature_std
        return X

    def fit(self, train_df):
        X = self._featurize(train_df)
        campaign = train_df["campaign"].to_numpy()
        if OUTCOME == "revenue":
            target = train_df["revenue"].to_numpy()
        else:
            target = train_df["converted"].to_numpy().astype(np.float64)

        # One regressor per campaign, trained on the customers who were shown it.
        for k in range(N_CAMPAIGNS):
            mask = campaign == k
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(X[mask], target[mask])
            self.models[k] = model
        return self

    def recommend(self, df):
        X = self._featurize(df)
        n = X.shape[0]
        # Predicted expected outcome of each campaign for each customer.
        pred = np.zeros((n, N_CAMPAIGNS))
        for k in range(N_CAMPAIGNS):
            pred[:, k] = self.models[k].predict(X)
        if OUTCOME == "converted":
            # Convert predicted conversion prob into expected revenue using the
            # customer's average observed spend as a rough per-conversion margin.
            margin = df["monetary"].to_numpy() * 0.35 + 12.0
            pred = pred * margin[:, None]
        # Net value = expected revenue - cost of contacting.
        net = pred - (CAMPAIGN_COSTS[None, :] if COST_AWARE else 0.0)
        return net.argmax(axis=1)

# ---------------------------------------------------------------------------
# Run one experiment: fit within the time budget, then score on held-out customers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    np.random.seed(42)

    train_df = load_train_experiment()
    print(f"Loaded {len(train_df):,} logged customers, {N_CAMPAIGNS} campaigns")

    rec = Recommender()
    t_fit0 = time.time()
    rec.fit(train_df)
    fit_seconds = time.time() - t_fit0
    print(f"Fit in {fit_seconds:.1f}s (budget {FIT_TIME_BUDGET}s)")
    if fit_seconds > FIT_TIME_BUDGET:
        print("FAIL: fit exceeded time budget")
        raise SystemExit(1)

    metrics = evaluate_policy(rec)
    t_end = time.time()

    # Human-readable action mix.
    counts = metrics["action_counts"]
    total = sum(counts)
    mix = ", ".join(f"{CAMPAIGN_NAMES[k]}:{100*c/total:.0f}%" for k, c in enumerate(counts) if c)

    print("---")
    print(f"policy_value:        {metrics['policy_value']:.6f}")
    print(f"regret:              {metrics['regret']:.6f}")
    print(f"lift_vs_best_single: {metrics['lift_vs_best_single_pct']:.2f}")
    print(f"gain_captured:       {100 * metrics['gain_captured_frac']:.2f}")
    print(f"oracle_value:        {metrics['oracle_value']:.6f}")
    print(f"best_single_value:   {metrics['best_single_value']:.6f}")
    print(f"no_contact_value:    {metrics['no_contact_value']:.6f}")
    print(f"fit_seconds:         {fit_seconds:.1f}")
    print(f"total_seconds:       {t_end - t_start:.1f}")
    print(f"n_campaigns:         {N_CAMPAIGNS}")
    print(f"action_mix:          {mix}")
    print(f"recommender:         T-learner (Ridge, outcome={OUTCOME})")

# autoresearch

This is an experiment to have the LLM do its own research: autonomously improving
the **recommendation algorithm** that decides which marketing campaign to send each
customer, based on the results of a randomized campaign test.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context and the problem being solved.
   - `prepare.py` — fixed data generation, logged experiment, and the evaluation metric. Do not modify.
   - `recommend.py` — the file you modify. The targeting model and how it picks campaigns.
4. **Verify data exists**: Check that `~/.cache/autoresearch/data/` contains the parquet files. If not, tell the human to run `uv run prepare.py` (fast, a couple seconds).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment fits a recommender on the logged campaign experiment (CPU only) and
scores it on a held-out set of customers. You launch it simply as: `uv run recommend.py`.

**What you CAN do:**
- Modify `recommend.py` — this is the only file you edit. Everything is fair game: the model class, feature engineering, how treatment effects / uplift are estimated (T-learner, S-learner, X-learner, direct uplift, ...), propensity handling, cost-aware selection, calibration, ensembling, regularization, hyperparameters, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data generating process, the logged experiment, and the evaluation metric.
- Modify the evaluation harness. The `evaluate_policy` function in `prepare.py` is the ground-truth metric. Do NOT import or use any of prepare.py's internal ground-truth functions (`_true_outcome_model`, the `val_truth.npz` file, etc.) inside `recommend.py` — the recommender may only learn from the logged data returned by `load_train_experiment()` and the held-out customer features. Using the ground truth to make recommendations is cheating and invalidates the result.
- Add dependencies beyond what's in `pyproject.toml` (numpy, pandas, scikit-learn, scipy are available).

**The goal is simple: get the highest `policy_value`.** This is the expected net profit (in $) per targeted customer under your recommender's per-customer campaign choices — HIGHER IS BETTER. For reference, the harness also prints the value of never contacting anyone (`no_contact_value`), the best you can do with a single mass campaign (`best_single_value`), and the unbeatable personalized `oracle_value`. Your job is to climb from `best_single_value` toward `oracle_value`.

**Time budget**: Fitting the recommender must finish within `FIT_TIME_BUDGET` (120s). This keeps experiments fast and comparable — you can't win just by throwing unlimited compute at fitting. The evaluation is exact and deterministic, so there is no evaluation noise: any change in `policy_value` is real signal.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. Weigh the complexity cost against the improvement magnitude. A +0.001 policy_value gain that adds 30 lines of hacky code? Probably not worth it. A +0.001 gain from deleting code? Definitely keep. Equal policy_value but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the recommender as is.

## Ideas to explore (non-exhaustive)

The baseline is a linear T-learner (one Ridge model per campaign, predicting revenue,
picking argmax of predicted revenue minus cost). The true campaign effects are
nonlinear with feature interactions, so there is real headroom. Directions:

- **Better outcome models**: gradient boosting (`HistGradientBoostingRegressor`), richer regularization, calibration.
- **Better features**: interactions, polynomial terms, target/frequency encodings, log transforms of skewed features (monetary, recency).
- **Uplift/meta-learners**: S-learner (single model with campaign as a feature), X-learner, R-learner, or direct uplift modeling instead of the plain T-learner.
- **Predict conversion vs revenue**: model conversion probability and multiply by a learned per-customer margin, rather than regressing revenue directly.
- **Cost-aware & risk-aware selection**: better handling of the "no contact" option, thresholds, shrinking toward control when uplift is uncertain.
- **Variance reduction**: pooling information across campaigns (shared base model + per-campaign deltas) instead of six fully independent models.

## Output format

Once the script finishes it prints a summary like this:

```
---
policy_value:        6.039625
regret:              1.000711
lift_vs_best_single: 29.60
gain_captured:       57.95
oracle_value:        7.040336
best_single_value:   4.660287
no_contact_value:    3.693594
fit_seconds:         0.2
total_seconds:       0.3
n_campaigns:         6
action_mix:          no_contact:8%, email_10pct_off:24%, ...
recommender:         T-learner (Ridge, outcome=revenue)
```

You can extract the key metric from the log file:

```
grep "^policy_value:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	policy_value	regret	status	description
```

1. git commit hash (short, 7 chars)
2. policy_value achieved (e.g. 6.039625) — use 0.000000 for crashes
3. regret (oracle_value - policy_value, e.g. 1.000711) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	policy_value	regret	status	description
a1b2c3d	6.039625	1.000711	keep	baseline: linear T-learner (Ridge)
b2c3d4e	6.265800	0.774500	keep	switch per-campaign models to gradient boosting
c3d4e5f	6.010000	1.030300	discard	S-learner with campaign one-hot
d4e5f6g	0.000000	0.000000	crash	added feature that referenced a missing column
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `recommend.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run recommend.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the result: `grep "^policy_value:\|^regret:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If policy_value improved (higher), you "advance" the branch, keeping the git commit.
9. If policy_value is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment is fast (fit is capped at 120s). If a run somehow hangs past a few minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (a bug, an exceeded fit budget, etc.), use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, revisit the ideas list above, try combining previous near-misses, try more radical modeling changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes well under a minute, so you can run a great many overnight. The user then wakes up to a log of experiments, all completed by you while they slept, and (hopefully) a materially better targeting policy than the baseline.

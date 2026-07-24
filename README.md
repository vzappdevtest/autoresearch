# autoresearch

*An autonomous AI agent that improves the recommendation algorithm behind a
marketing campaign — overnight, on its own. You give it a randomized campaign
test and a baseline targeting model; it edits the code, refits, checks whether
profit-per-customer improved, keeps or discards, and repeats. You wake up to a
log of experiments and (hopefully) a materially better targeting policy.*

The idea (adapted from Karpathy's original [autoresearch](https://github.com/karpathy/autoresearch)):
give an AI agent a small but real optimization problem and let it experiment
autonomously. Here the problem is **campaign targeting**: given the results of a
randomized marketing test, learn a policy that decides which campaign variant to
send each customer to maximize net profit. As with the original, you're not
touching the algorithm file like a normal engineer would. Instead you program the
`program.md` file that instructs the agent, and let it iterate on the algorithm
for you.

## The problem

A marketing team ran a **randomized A/B/n test**. Each customer was shown one of
several campaign variants — chosen uniformly at random — including a "no contact"
control. The log records, for every customer: their features, which variant they
got, and the outcome (did they convert, and how much revenue). Because assignment
was randomized, this log is an unbiased basis for learning who to target with what.

The **recommendation task**: for each new customer, decide which campaign to send
(or none) so as to maximize expected **net profit** = expected revenue from
conversion − cost of the contact. Campaign effects are *heterogeneous*: a discount
that lifts conversion for bargain-hunters is wasted margin on loyal full-price
buyers, and an aggressive retargeting ad that wins back lapsed customers annoys
recently-active ones into opting out. So the best policy is personalized, and it
beats any single mass campaign.

The metric is **`policy_value`** — the expected net profit per targeted customer
under the recommender's choices. Higher is better. It's computed exactly against a
frozen ground-truth simulator, so every experiment is directly and noiselessly
comparable.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — the fixed harness. Simulates the randomized campaign experiment, freezes it to disk, and defines the ground-truth evaluation metric (`evaluate_policy`). **Not modified.**
- **`recommend.py`** — the single file the agent edits. Contains the targeting model: how it learns from the logged experiment and how it picks a campaign for each customer. **This file is edited and iterated on by the agent.**
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human.**

The agent's only job is to raise `policy_value` on held-out customers, climbing from
the best-single-campaign baseline toward the personalized oracle.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). CPU only — no GPU needed.

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Generate the campaign dataset (one-time, ~1s)
uv run prepare.py

# 4. Run a single experiment with the baseline recommender (~1s)
uv run recommend.py
```

A baseline run prints something like:

```
---
policy_value:        6.039625
regret:              1.000711
lift_vs_best_single: 29.60
gain_captured:       57.95
oracle_value:        7.040336
best_single_value:   4.660287
no_contact_value:    3.693594
...
```

Read that as: the baseline earns **$6.04 net profit per customer**, versus $4.66 for
the best mass campaign and $3.69 for not contacting anyone. The unbeatable
personalized oracle earns $7.04, so the baseline has captured ~58% of the available
personalization gain — leaving real headroom for the agent to close.

If those commands work, your setup is good and you can go into autonomous research mode.

## Running the agent

Spin up your Claude/Codex or whatever you want in this repo (and disable all
permissions), then prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill" that turns the
agent into an autonomous targeting-algorithm researcher.

## Project structure

```
prepare.py      — data generation + logged experiment + evaluation metric (do not modify)
recommend.py    — the targeting model (agent modifies this)
program.md      — agent instructions
analysis.ipynb  — plot experiment progress and inspect a policy
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `recommend.py`. This keeps the scope manageable and diffs reviewable.
- **Exact, comparable metric.** Because the data is a frozen, seeded simulation, `evaluate_policy` computes the true expected profit of a policy with zero evaluation noise. Any change in `policy_value` is real signal — no lucky seeds, no variance to chase.
- **Fixed fit budget.** Fitting is capped at 120s, so a recommender can't win just by burning more compute. The interesting constraint is generalization from the logged experiment, not raw horsepower.
- **Honest learning setup.** The recommender only ever sees the logged, randomized experiment (features, assigned campaign, propensity, outcome) — never the ground-truth response surface. That's exactly the position a real marketing data scientist is in.
- **Self-contained & CPU-only.** No GPU, no external data downloads, no network. Standard `numpy` / `pandas` / `scikit-learn` / `scipy`.

## The recommendation problem in more depth

The synthetic experiment is built to reward good modeling, not luck:

- **Heterogeneous treatment effects.** Each campaign's uplift on conversion depends on a *nonlinear* mix of customer features (interactions and thresholds, not a simple weighted sum). A linear model captures part of it; tree ensembles, interaction features, and meta-learners (S-/T-/X-learners) can capture more.
- **Costs and the control arm.** Contacts cost money and the "no contact" option is always available, so the policy must be cost-aware — sometimes the right call is to spend nothing.
- **Uplift ≪ outcome.** As in real life, the treatment effect is small relative to baseline conversion, so naively regressing revenue leaves a lot on the table versus modeling the *incremental* effect directly.

See the "Ideas to explore" section of `program.md` for concrete directions the agent
can take, from gradient-boosted outcome models to X-learners to cost-aware selection.

## Credit

The framework and philosophy — a single editable algorithm file, a fixed harness, a
one-number metric, and an autonomous keep/discard loop — are adapted from
[@karpathy's autoresearch](https://github.com/karpathy/autoresearch), originally a
5-minute LLM pretraining loop. This fork repurposes that loop for marketing-campaign
recommendation.

## License

MIT

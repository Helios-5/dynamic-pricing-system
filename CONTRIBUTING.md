# Contributing to the AI-Driven Dynamic Pricing System

First off — thanks for taking the time to contribute. This document lays out
the engineering standards and workflow expectations so your PR lands cleanly on
the first review cycle instead of bouncing back with style corrections.

---

## Table of Contents

1. [Local Setup](#local-setup)
2. [Running the App](#running-the-app)
3. [Running the Test Suite](#running-the-test-suite)
4. [Project Architecture in 60 Seconds](#project-architecture-in-60-seconds)
5. [The Non-Negotiable: Inline Documentation Standards](#the-non-negotiable-inline-documentation-standards)
6. [Pull Request Checklist](#pull-request-checklist)
7. [Branch and Commit Conventions](#branch-and-commit-conventions)

---

## Local Setup

We use a `Makefile` to abstract away the verbose Docker and pytest incantations.
If you're on a machine without `make`, the underlying commands are in the
Makefile targets — you can always copy-paste them directly.

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/dynamic-pricing-system.git
cd dynamic-pricing-system

# 2. Create a virtual environment and install all dependencies
#    (production deps from requirements.txt + pytest + pytest-cov)
make install

# 3. Activate the venv for your shell session
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

You're ready. The venv now has scikit-learn, PuLP, Streamlit, and the test
tools already installed.

### Available Make Targets

Run `make` (or `make help`) at any time for the full list:

| Command | What it does |
|---|---|
| `make install` | Creates `.venv` and installs all dependencies |
| `make test` | Runs the full pytest suite with coverage |
| `make test-fast` | Runs pytest without the coverage report (faster) |
| `make run-local` | Starts the Streamlit app on `localhost:8501` |
| `make generate-data` | Runs the synthetic data generation script |
| `make build` | Builds the Docker image |
| `make up` | Starts Docker services in detached mode |
| `make down` | Stops and removes Docker containers |
| `make logs` | Tails Docker logs live |
| `make clean` | Removes `__pycache__`, `.pytest_cache`, stale `.joblib` files |

---

## Running the App

### Option A: Locally (fastest for active development)

```bash
make run-local
```

The app will be available at `http://localhost:8501`. Streamlit hot-reloads on
file save, so you'll see your changes reflected without restarting.

### Option B: Docker (closest to production behaviour)

```bash
make build    # Build the image once
make up       # Start the container in detached mode
make logs     # Watch the output
make down     # Shut everything down when done
```

The Dockerfile installs dependencies in a layer-cached step, so re-builds
after a pure Python change take a few seconds, not a minute.

---

## Running the Test Suite

```bash
make test
```

This runs all 114 tests across `test_data_access.py`, `test_features.py`,
`test_model.py`, and `test_optimization.py` with coverage reporting. The
whole suite completes in under 5 seconds on a modern laptop.

If you're in a fast edit-test loop and coverage reporting is slowing you down:

```bash
make test-fast
```

### Writing New Tests

All tests live in `tests/`. If you add a new module in `src/`, add a
corresponding `test_<module>.py` file. The test conventions are:

- **Isolated** — tests must not depend on the state of other tests. Reset any
  `@lru_cache` between tests (see `test_data_access.py::_clear_config_cache()`
  for the pattern).
- **Deterministic** — if the production code uses a random number generator,
  mock it in tests. See the zero-noise RNG mock in `test_features.py` for how
  we handle `np.random.default_rng`.
- **Fast** — tests should complete in milliseconds. If a test takes more than a
  second, it's doing too much and should be split or mocked more aggressively.
- **Named descriptively** — the test name is the failure message a reviewer sees
  in CI. `test_infinite_demand_ratio_is_clipped_to_multiplier_max` is useful.
  `test_feature_1` is not.

---

## Project Architecture in 60 Seconds

The system is strictly decoupled across five modules. Understanding this before
you touch any file will save you a debugging session:

```
generator.py      →  Creates raw ride-request DataFrames (synthetic simulation)
    ↓
features.py       →  Transforms raw rows into ML-ready feature vectors.
                     Also generates the stochastic price_multiplier target.
    ↓
model.py          →  Trains RandomForestRegressor + LinearRegression.
                     Keeps whichever achieves lower RMSE on 80/20 holdout.
    ↓
optimization.py   →  Refines the ML price via a soft-constraint LP (PuLP + CBC).
                     Slack variables ensure the solver NEVER returns Infeasible.
    ↓
app.py            →  Pure Streamlit UI. Zero business logic.
                     Calls the above modules; renders what they return.

data_access.py    →  Cross-cutting concern. Config loading, CSV path resolution,
                     coordinate imputation, DataFrame filtering. Used by both
                     app.py and optimization.py.
```

**The rule:** if you find yourself writing business logic in `app.py`, it
belongs in one of the other modules. `app.py` should read like a list of
function calls and Streamlit render instructions — nothing more.

---

## The Non-Negotiable: Inline Documentation Standards

> **PRs with inadequate comments will be returned for revision before any
> functional review begins. This is not a stylistic preference — it is an
> engineering standard for this codebase.**

This project's inline documentation follows a strict philosophy that's best
summarised as: **explain the Why, not the What**. The code already says
_what_ is happening. A comment that just restates the code in plain English
adds noise, not signal.

### What we reject

These are real patterns that will get a PR sent back:

```python
# Bad: restates the code
# Divide requests by drivers to get demand ratio
df["demand_ratio"] = df["requests"] / df["drivers"]

# Bad: generic, AI-generated-sounding, says nothing useful
# This function trains the model using the provided dataframe.
def train_model(df):
    ...

# Bad: single-line comment on a non-obvious decision
model = RandomForestRegressor(n_estimators=200)  # 200 trees

# Bad: no comment at all on a business-critical parameter
_RET_TARGET: float = 0.80
```

### What we require

Comments must explain:

- **Why this value, not another?** If you pick `n_estimators=200`, say that
  it's the point where training time on 2,000 rows is ~1-2 seconds on a modern
  laptop, and going to 500 adds no meaningful accuracy gain on this feature set.
- **What happens if this assumption breaks?** If `demand_ratio` can be `inf`
  (because `drivers=0`), say so, say why, and explain why the clip is the
  right fix rather than a try/except.
- **What business rule is encoded here?** If `_RET_TARGET = 0.80` comes from an
  A/B test result, say so. If it's a product team decision, cite the decision.
  Future engineers need to know whether they can change it safely or whether
  it's load-bearing.
- **What you tried that didn't work.** If you chose a soft-constraint LP over
  a hard-constraint LP because the hard version produced infeasible regions at
  extreme utilization, say that. That context is irreplaceable once it leaves
  your head.

### The tone we're after

Write like a senior engineer leaving notes for the next person on the team —
not like an API reference document. Conversational, pragmatic, and specific.
The existing files (`model.py`, `optimization.py`, `features.py`) are the
style reference. Read a few hundred lines before you start writing.

**A well-documented code block looks like this:**

```python
# ---------------------------------------------------------------------------
# Soft-constraint design note:
#
# The original version of this used hard constraints on utilization and retention.
# When utilization was already at 0.97 during a city-wide cricket match, the
# constraints became contradictory — the LP returned 'Infeasible' and silently
# fell back to the raw ML prediction, meaning the entire optimisation layer did
# nothing. We only discovered this weeks later when someone noticed the LP metrics
# weren't showing in the logs.
#
# Soft constraints fix this: slack variables absorb any violation, the feasible
# region is always non-empty, and the solver always returns Optimal. The cost of
# a violation is baked into the objective — the LP will only violate a constraint
# when the revenue gain from doing so exceeds the penalty weight. If you need
# stricter enforcement, raise the penalty in config.yaml. Don't go back to hard
# constraints without a very good reason and a test that covers the
# over-utilization case.
# ---------------------------------------------------------------------------
```

That's the bar. Match it.

---

## Pull Request Checklist

Before opening a PR, run through this list. CI will catch most of it, but
fixing it before the CI run wastes less of everyone's time:

- [ ] `make test` passes locally with **zero failures**.
- [ ] New public functions and classes have **docstrings** that explain the Why.
- [ ] New constants or magic numbers have **inline comments** explaining their
      origin (empirical tuning, product decision, A/B result, etc.).
- [ ] If you changed a feature in `features.py`, you updated the `FEATURES`
      list in `model.py` and added/updated a test in `test_features.py`.
- [ ] If you changed an LP constraint in `optimization.py`, you added a test
      in `test_optimization.py` that exercises the edge case you're modifying.
- [ ] The PR description explains **what** changed and **why** — not just the
      diff in prose form.
- [ ] No `xgboost`, `torch`, `tensorflow`, or other heavy ML library imports.
      The algorithm constraint is intentional: RF + LR + PuLP only.

---

## Branch and Commit Conventions

- Branch names: `feature/<short-description>`, `fix/<short-description>`,
  `docs/<short-description>`.
- Commits: imperative mood, present tense. `Add soft constraint for retention`
  not `Added` or `Adding`.
- Keep commits atomic — one logical change per commit makes `git bisect` useful
  when something breaks.
- Squash fixup commits before opening the PR. A clean history is a courtesy
  to the engineer who will blame this code in 18 months.

---

_This document was written with the same philosophy as the codebase: explain
the reasoning, not just the rules. If something in here seems arbitrary or
unclear, open an issue — the engineering standards should be as easy to
understand as the code they govern._

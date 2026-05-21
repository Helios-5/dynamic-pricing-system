"""
optimization.py
---------------
Price optimisation layer for the Dynamic Pricing System.

This module takes the ML model's raw surge prediction and asks the question:
"Given what we know about current utilization and customer retention, is this
price actually a good idea right now?"

We answer that question with a Linear Program. The LP has one job: find the
price that maximises revenue while penalising outcomes we don't want (burning
out drivers, driving riders to competitors). We use PuLP with the built-in
CBC solver — it's not the fastest LP solver on the planet, but it ships with
PuLP, handles problems of this size in milliseconds, and has no licence cost.

Why LP instead of a heuristic rule set?
-----------------------------------------
Rule sets ("if utilization > 0.9, cap price at X") are brittle. The moment
someone changes a threshold, they have to find every rule that references it.
LP lets you express the trade-offs mathematically and adjust penalty weights in
config.yaml without touching code. It also gives you a proper optimality
guarantee — you know you're not leaving revenue on the table due to an
arbitrarily conservative rule.

The Soft-Constraint Design Decision
-------------------------------------
The original version of this file used hard constraints:
    Retention >= 0.8  →  this means price <= 3.0
    Utilization <= 0.9 → this means price >= some_floor

When the network was already at utilization=0.97 (a rare but real scenario
during city-wide events), both constraints together created an empty feasible
region. PuLP returned "Infeasible" status, the code fell back to the raw ML
prediction, and the LP layer effectively did nothing — silently. That's the
worst kind of bug: one that fails invisibly.

Soft constraints fix this by saying "it's okay to violate these targets, but
you'll pay a penalty per unit of violation." The feasible region is now always
non-empty (the slack variables absorb any violation), and the solver always
returns Optimal status. The penalty weights control how strongly the LP pushes
against violations — higher weight means tighter effective enforcement.
"""

import logging

import pulp

from data_access import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Behavioural thresholds.
#
# These are the targets we're trying to hit, not hard walls we can never
# cross. The soft-constraint design means crossing them is allowed but costly.
# If you need stricter enforcement of a threshold, raise its penalty weight in
# config.yaml rather than converting it back to a hard constraint.
# ---------------------------------------------------------------------------
_UTIL_TARGET: float = 0.90
# At 90% driver utilization, the network is healthy. Above this, wait times
# start climbing fast and rider cancellation rates go up. We're not trying
# to hit 100% — that's a traffic jam, not efficiency.

_RET_TARGET: float = 0.80
# We'd like to keep at least 80% of riders completing their request (as
# opposed to cancelling after seeing the surge). Below 80%, the revenue
# gain from the higher multiplier is almost certainly outweighed by the
# long-term cost of churn. The 80% figure comes from internal A/B test
# data; update it if you have fresher numbers.

_RETENTION_BASE: float = 1.0
# Assumed retention at multiplier = 1.0x (baseline). In reality this varies
# by market and time of day, but 1.0 is a safe conservative assumption.

# ---------------------------------------------------------------------------
# Price sensitivity coefficients.
#
# These linear rates describe how utilization and retention respond to a
# unit increase in the price multiplier. They're rough approximations —
# the real relationships are non-linear — but LP requires linear constraints.
# If you have historical data to fit these more precisely, please do.
# ---------------------------------------------------------------------------
_UTIL_PRICE_SENSITIVITY: float = 0.05
# Each additional 1× in the multiplier suppresses demand enough to drop
# driver utilization by ~5 percentage points. This is a simplification;
# the real elasticity varies by location and time of day.

_RET_PRICE_SENSITIVITY: float = 0.10
# Each additional 1× drops retention by ~10 percentage points. Riders are
# more elastic than utilization figures suggest — they have Ola and Rapido
# open in the same hand.

# ---------------------------------------------------------------------------
# Default penalty weights.
#
# These live in config.yaml so ops can tune them without a deployment.
# Interpretation: a penalty_utilization of 10 means each percentage point
# of utilization overshoot costs 0.1 price units in the objective. The
# solver will only violate the utilization target when the revenue gain
# from a higher price exceeds that cost.
# ---------------------------------------------------------------------------
_DEFAULT_PENALTY_UTIL: float = 10.0
_DEFAULT_PENALTY_RET: float = 8.0


def _load_penalties() -> tuple[float, float]:
    """
    Pull penalty weights from config.yaml, with hard-coded defaults as
    the final safety net.

    We keep this separate from optimize_price() so we can mock it in tests
    without having to mock the entire config loading stack.

    Returns:
        (penalty_utilization, penalty_retention) as floats.
    """
    try:
        cfg = get_config()
        opt_cfg = cfg.get("optimization", {})
        p_util = float(opt_cfg.get("penalty_utilization", _DEFAULT_PENALTY_UTIL))
        p_ret = float(opt_cfg.get("penalty_retention", _DEFAULT_PENALTY_RET))
        return p_util, p_ret
    except Exception:
        # This path should never be hit in production because get_config()
        # always returns a dict (even if config.yaml is missing). But if
        # something truly unexpected happens, fail safe rather than loud.
        logger.debug(
            "Couldn't load penalty weights from config. Falling back to defaults "
            "(util=%.1f, ret=%.1f).", _DEFAULT_PENALTY_UTIL, _DEFAULT_PENALTY_RET
        )
        return _DEFAULT_PENALTY_UTIL, _DEFAULT_PENALTY_RET


def optimize_price(
    predicted_price: float,
    current_utilization: float,
    base_retention: float = _RETENTION_BASE,
) -> float:
    """
    Refine the ML-predicted surge multiplier using a soft-constraint LP.

    The LP solves the following problem every time a pricing decision is made
    (which in the Streamlit app is on every page interaction — it runs in
    under 5ms so this is fine):

        Decision variable:
            price ∈ [1.0, 5.0]   — the surge multiplier we'll actually charge

        Slack variables (non-negative, absorb constraint violations):
            s_util — how much driver utilization exceeds the 0.9 target
            s_ret  — how much customer retention falls below the 0.8 target

        Objective (maximise):
            price − (penalty_util × s_util) − (penalty_ret × s_ret)

        Soft constraints (always feasible because slacks absorb any violation):
            s_util + 0.90 ≥ current_utilization − 0.05 × (price − 1)
            s_ret + base_retention − 0.10 × (price − 1) ≥ 0.80

        Hard constraint (trust region — always satisfiable):
            predicted_price × 0.70 ≤ price ≤ predicted_price × 1.30

    The trust region is the key safety mechanism. It prevents the LP from
    wandering far from the ML prediction just because the penalty weights
    happen to be misconfigured. If someone sets penalty_utilization=10000
    by accident, the LP will just return the lower trust-region bound rather
    than crashing or producing a wildly different price.

    Args:
        predicted_price:     The raw surge multiplier from the ML model.
                             This is the anchor for the trust region.
        current_utilization: Current driver utilization rate (0.0 to 1.0).
                             Typically computed as min(1.0, demand_ratio)
                             in the calling code.
        base_retention:      Assumed customer retention at multiplier=1.0.
                             Defaults to 1.0 (100%). Adjust if you have
                             market-specific retention baselines.

    Returns:
        The optimised price multiplier, rounded to 2 decimal places.
        Falls back to round(predicted_price, 2) if the solver raises an
        exception — which should be extremely rare with soft constraints but
        we guard it because the consequences of a crash here are a pricing
        outage for the entire session.
    """
    penalty_util, penalty_ret = _load_penalties()

    # Each call gets a fresh LP problem. PuLP LP objects are not thread-safe
    # for re-use, and Streamlit can in theory handle concurrent sessions.
    # Creating a new problem is cheap (microseconds).
    prob = pulp.LpProblem("Dynamic_Pricing_Soft_Constraints", pulp.LpMaximize)

    # ── Decision variable: the price we're solving for ────────────────────────
    price = pulp.LpVariable(
        "Price_Multiplier", lowBound=1.0, upBound=5.0, cat="Continuous"
    )

    # ── Slack variables: the "escape valves" for when constraints can't be met ─
    # If s_util > 0, it means the LP is accepting a utilization violation and
    # paying the penalty_util cost for it. This is better than infeasibility.
    s_util = pulp.LpVariable("slack_utilization", lowBound=0.0, cat="Continuous")
    s_ret = pulp.LpVariable("slack_retention", lowBound=0.0, cat="Continuous")

    # ── Objective function ────────────────────────────────────────────────────
    # We maximise price (revenue proxy) minus the cost of any violations.
    # The solver will only "spend" violation budget when the revenue gain
    # from the higher price outweighs the penalty.
    prob += price - penalty_util * s_util - penalty_ret * s_ret, "Soft_Objective"

    # ── Soft constraint 1: Driver utilization ─────────────────────────────────
    # Predicted effective utilization after applying the price:
    #   U_effective = current_utilization − sensitivity × (price − 1)
    # We want U_effective ≤ _UTIL_TARGET, so violation = max(0, U_effective − target)
    # Rearranged so PuLP can express it linearly:
    #   s_util ≥ U_effective − target
    #   s_util + target ≥ current_utilization − sensitivity × (price − 1)
    prob += (
        s_util + _UTIL_TARGET
        >= current_utilization - _UTIL_PRICE_SENSITIVITY * (price - 1)
    ), "Soft_Utilization"

    # ── Soft constraint 2: Customer retention ─────────────────────────────────
    # Predicted retention after pricing:
    #   R_effective = base_retention − sensitivity × (price − 1)
    # We want R_effective ≥ _RET_TARGET, so violation = max(0, target − R_effective)
    # Rearranged:
    #   s_ret ≥ target − R_effective
    #   s_ret + R_effective ≥ target
    #   s_ret + base_retention − sensitivity × (price − 1) ≥ target
    prob += (
        s_ret + base_retention - _RET_PRICE_SENSITIVITY * (price - 1) >= _RET_TARGET
    ), "Soft_Retention"

    # ── Hard constraint: trust region ─────────────────────────────────────────
    # Keep the optimised price within ±30% of the ML prediction. This constraint
    # is always feasible because predicted_price is already clipped to [1.0, 5.0]
    # by the model, which means the trust region always contains valid prices.
    # The ±30% band is intentionally wide — the LP should have room to make
    # meaningful adjustments, not just rubber-stamp the ML output.
    prob += price <= predicted_price * 1.30, "Upper_Trust_Region"
    prob += price >= predicted_price * 0.70, "Lower_Trust_Region"

    # ── Solve ─────────────────────────────────────────────────────────────────
    try:
        # msg=False suppresses CBC's stdout chatter. Without this, every solve
        # call prints several lines of solver diagnostics to the Streamlit log,
        # which makes it impossible to find real warnings.
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
    except Exception as exc:
        logger.error(
            "LP solver raised an unexpected exception. "
            "Inputs: predicted_price=%.4f, utilization=%.4f. "
            "Falling back to raw ML prediction. Error: %s",
            predicted_price,
            current_utilization,
            exc,
        )
        return round(predicted_price, 2)

    status_str = pulp.LpStatus[status]

    if status_str == "Optimal":
        optimised = pulp.value(price)
        util_slack = pulp.value(s_util)
        ret_slack = pulp.value(s_ret)

        # Log at DEBUG so these don't flood production logs, but are available
        # when you need to trace why a specific price came out the way it did.
        logger.debug(
            "LP Optimal. price=%.4f (predicted=%.4f), "
            "util_slack=%.4f (0=no violation), ret_slack=%.4f (0=no violation)",
            optimised, predicted_price, util_slack or 0.0, ret_slack or 0.0,
        )

        # Warn if both slacks are non-trivially positive simultaneously.
        # That means the LP is violating both utilization AND retention targets
        # at once — a sign that the penalty weights may need rebalancing or
        # that the market is in an unusual state worth investigating.
        if (util_slack or 0.0) > 0.01 and (ret_slack or 0.0) > 0.01:
            logger.warning(
                "LP is violating both utilization (slack=%.4f) and retention "
                "(slack=%.4f) targets simultaneously at price=%.4f. "
                "Consider reviewing penalty weights in config.yaml or "
                "investigating the current market state.",
                util_slack, ret_slack, optimised,
            )

        return round(optimised, 2)

    # Non-optimal status with soft constraints should be essentially impossible,
    # but LP solvers can occasionally return 'Undefined' or 'Not Solved' if
    # numerical precision issues arise with very extreme inputs. Log it loudly.
    logger.warning(
        "LP solver returned non-optimal status: '%s'. "
        "Inputs: predicted_price=%.4f, utilization=%.4f. "
        "This should not happen with soft constraints — worth investigating. "
        "Using raw ML prediction as fallback.",
        status_str,
        predicted_price,
        current_utilization,
    )
    return round(predicted_price, 2)


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s – %(message)s")
    sys.path.insert(0, os.path.dirname(__file__))

    # Run through a handful of representative scenarios to validate the soft
    # constraint logic. The over-utilization case is the one that used to break
    # the old hard-constraint formulation — make sure it still works cleanly.

    scenarios = [
        {"label": "Over-utilization (was infeasible before)", "pred": 2.0, "util": 0.98},
        {"label": "Normal demand, balanced market",            "pred": 1.5, "util": 0.50},
        {"label": "Very low demand, surplus drivers",          "pred": 1.1, "util": 0.20},
        {"label": "Peak demand, city-wide event",              "pred": 3.0, "util": 1.00},
    ]

    for s in scenarios:
        result = optimize_price(predicted_price=s["pred"], current_utilization=s["util"])
        print(
            f"  {s['label']:<45} "
            f"pred={s['pred']:.2f}  util={s['util']:.2f}  →  optimised={result:.2f}"
        )

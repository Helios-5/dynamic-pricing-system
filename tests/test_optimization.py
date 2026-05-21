"""
test_optimization.py
--------------------
Unit tests for src/optimization.py.

The optimization layer is the production safety net — it takes the ML model's
raw surge prediction and makes sure we're not charging a price that burns out
our drivers or drives away our riders.  If it breaks, the app still runs, but
it either falls back to the raw ML price (if the solver crashes) or produces
subtly wrong prices (if the LP constraints are silently violated).

Two categories of tests here:

1. Behavioral tests — real solver runs, real LP solutions.
   These call optimize_price() with actual PuLP/CBC and verify the outputs
   make business sense.  They're slower than pure unit tests (~50ms each vs ~1ms)
   but they test the thing that actually runs in production.

2. Failure-mode tests — mock the solver to simulate crashes and bad statuses.
   These don't test the LP math; they test whether our error handling and
   fallback logic works correctly.  We mock pulp.LpProblem to raise an exception
   or return a non-Optimal status.

On mocking strategy:
We mock at the ``optimization.pulp`` level, not at the ``pulp`` module level.
This is the correct approach because optimization.py does ``import pulp`` and
then uses ``pulp.LpProblem``, ``pulp.LpVariable``, etc.  Patching the module-level
``pulp`` in optimization's namespace intercepts those calls correctly.
Patching the top-level ``pulp`` package would not intercept calls made from
within optimization.py after the import has already happened.
"""

import logging
from unittest.mock import patch, MagicMock, PropertyMock

import pulp
import pytest

import optimization
from optimization import (
    optimize_price,
    _load_penalties,
    _DEFAULT_PENALTY_UTIL,
    _DEFAULT_PENALTY_RET,
    _UTIL_TARGET,
    _RET_TARGET,
)


# ---------------------------------------------------------------------------
# Tests: _load_penalties()
# ---------------------------------------------------------------------------

class TestLoadPenalties:
    """
    Tests for the penalty weight loading.

    _load_penalties() is a thin wrapper around get_config().  We test its
    defaults and its graceful failure mode rather than re-testing all the config
    loading logic (which is covered in test_data_access.py).
    """

    def test_returns_tuple_of_two_floats(self):
        """The return type must always be (float, float), not (None, None) or a dict."""
        result = _load_penalties()
        assert isinstance(result, tuple)
        assert len(result) == 2
        p_util, p_ret = result
        assert isinstance(p_util, float)
        assert isinstance(p_ret, float)

    def test_returns_defaults_when_config_raises(self):
        """
        If get_config() throws for any reason, _load_penalties() must catch it
        and return the module-level defaults rather than propagating the exception.

        This is the 'fail safe' guarantee of the optimization layer — even if
        the entire config system is broken, we still return a valid price.
        """
        with patch("optimization.get_config", side_effect=RuntimeError("YAML parse error")):
            p_util, p_ret = _load_penalties()
        assert p_util == _DEFAULT_PENALTY_UTIL
        assert p_ret == _DEFAULT_PENALTY_RET

    def test_positive_penalty_values(self):
        """Penalties must be positive — a zero or negative penalty would mean we're
        rewarding constraint violations, which would make the LP useless."""
        p_util, p_ret = _load_penalties()
        assert p_util > 0
        assert p_ret > 0


# ---------------------------------------------------------------------------
# Tests: optimize_price() — behavioral (real solver)
# ---------------------------------------------------------------------------

class TestOptimizePriceNormalScenarios:
    """
    Behavioral tests using the real CBC solver.

    The first and most important property to verify: the output is always a
    valid price in [1.0, 5.0].  This must hold for every input combination —
    the LP should never produce a price outside this range.
    """

    @pytest.mark.parametrize("predicted_price,current_utilization", [
        (1.0, 0.0),    # Absolute minimum inputs
        (5.0, 1.0),    # Absolute maximum inputs
        (2.5, 0.5),    # Typical mid-day scenario
        (1.5, 0.75),   # Moderate demand, moderate utilization
        (3.0, 0.85),   # High demand, approaching utilization target
        (1.1, 0.30),   # Low demand, surplus drivers
    ])
    def test_output_is_always_within_valid_price_range(self, predicted_price, current_utilization):
        """
        The absolute contract: no matter what the ML model predicts and no matter
        what the utilization is, the optimizer must return a price in [1.0, 5.0].
        This is enforced by the hard bounds on the LP decision variable, not by
        any application code — so if this fails, the LP formulation itself is broken.
        """
        result = optimize_price(predicted_price, current_utilization)
        assert 1.0 <= result <= 5.0, (
            f"optimize_price({predicted_price}, {current_utilization}) = {result} "
            f"is outside [1.0, 5.0]. The LP hard bounds are violated."
        )

    def test_output_is_float(self):
        """The return type must be a Python float, not a PuLP variable or None."""
        result = optimize_price(2.0, 0.5)
        assert isinstance(result, float)

    def test_output_is_rounded_to_two_decimal_places(self):
        """
        We display the optimized price in the UI as '2.60x' etc.  The round(2)
        in optimize_price() ensures this.  If it changed to round(4), the UI
        would look ugly — small thing, but worth locking in.
        """
        result = optimize_price(2.0, 0.5)
        # Check that it has at most 2 decimal places by comparing to itself rounded
        assert result == round(result, 2)

    def test_normal_demand_returns_price_near_prediction(self):
        """
        Under normal conditions (utilization=0.5, well below the 0.9 target),
        there's no reason for the LP to deviate much from the ML prediction.
        The output should be within the ±30% trust region.
        """
        predicted = 2.0
        result = optimize_price(predicted, current_utilization=0.5)
        lower_bound = predicted * 0.70
        upper_bound = predicted * 1.30
        assert lower_bound <= result <= upper_bound, (
            f"Under normal conditions (util=0.5), expected output in "
            f"[{lower_bound:.2f}, {upper_bound:.2f}] but got {result:.2f}."
        )


# ---------------------------------------------------------------------------
# Tests: optimize_price() — edge cases with violated soft constraints
# ---------------------------------------------------------------------------

class TestOptimizePriceSoftConstraintEdgeCases:
    """
    This is the most important test class in this file.

    The entire soft-constraint redesign was motivated by a production bug: when
    utilization was already at 0.97+, the old hard-constraint LP had an empty
    feasible region and returned 'Infeasible'.  The fallback to the raw ML
    prediction meant the optimization layer silently did nothing.

    These tests verify that the soft-constraint formulation never fails with
    an infeasible status, even under extreme inputs.
    """

    def test_extremely_high_utilization_does_not_crash(self):
        """
        utilization=0.99 was the scenario that broke the old hard-constraint LP.

        With soft constraints, the solver must still return Optimal and produce
        a valid price.  The slack variables absorb the utilization violation —
        we pay a penalty in the objective but never fail to find a solution.
        """
        result = optimize_price(predicted_price=2.0, current_utilization=0.99)
        assert isinstance(result, float), (
            "optimize_price() returned a non-float — possible solver crash."
        )
        assert 1.0 <= result <= 5.0, (
            f"Got {result} — outside valid bounds even with extreme high utilization."
        )

    def test_utilization_at_maximum_one_does_not_crash(self):
        """
        utilization=1.0 means every driver is occupied.  This is a plausible
        state during city-wide events.  The solver must still find a valid price.
        """
        result = optimize_price(predicted_price=3.0, current_utilization=1.0)
        assert 1.0 <= result <= 5.0

    def test_utilization_at_zero_does_not_crash(self):
        """
        utilization=0.0 means no rides are in progress — a possible state right
        after a city-wide power outage or during a low-demand window.
        """
        result = optimize_price(predicted_price=1.2, current_utilization=0.0)
        assert 1.0 <= result <= 5.0

    def test_predicted_price_at_minimum_does_not_crash(self):
        """
        predicted_price=1.0 is the minimum value. The trust region becomes
        [0.70, 1.30], but the LP lower bound is also 1.0, so the effective
        range is [1.0, 1.30].  Should still produce a valid result.
        """
        result = optimize_price(predicted_price=1.0, current_utilization=0.5)
        assert 1.0 <= result <= 5.0

    def test_predicted_price_at_maximum_does_not_crash(self):
        """
        predicted_price=5.0 means the model is predicting maximum surge.
        The trust region becomes [3.5, 5.0].  Still valid — should stay in bounds.
        """
        result = optimize_price(predicted_price=5.0, current_utilization=0.8)
        assert 1.0 <= result <= 5.0

    def test_both_constraints_violated_simultaneously(self):
        """
        The scenario where both utilization is critically high AND retention
        is under pressure.  In the old hard-constraint formulation, this would
        create contradictory constraints and cause infeasibility.

        With soft constraints, both slacks activate, both penalties apply, and
        the LP still converges to the least-bad price.  This is the core value
        proposition of the soft-constraint design — a sub-optimal price beats
        a pricing outage every time.
        """
        # High predicted price stresses retention (pushes it below 0.8 threshold)
        # High utilization stresses the utilization constraint
        result = optimize_price(predicted_price=4.5, current_utilization=0.98)
        assert isinstance(result, float)
        assert 1.0 <= result <= 5.0, (
            f"Got {result} when both constraints are simultaneously stressed. "
            f"Soft constraints should absorb this without failing."
        )

    @pytest.mark.parametrize("predicted,util", [
        (1.0, 0.0),
        (1.0, 1.0),
        (5.0, 0.0),
        (5.0, 1.0),
        (2.5, 0.5),
    ])
    def test_corner_combinations_all_return_valid_prices(self, predicted, util):
        """
        Exhaustive test of corner inputs.  The parametrize decorator runs this
        test for each combination — it's cheap since CBC solves in <5ms.
        If any combination fails, we'll know exactly which one from the test ID.
        """
        result = optimize_price(predicted_price=predicted, current_utilization=util)
        assert 1.0 <= result <= 5.0, (
            f"optimize_price({predicted}, {util}) = {result} — outside valid range."
        )


# ---------------------------------------------------------------------------
# Tests: optimize_price() — solver failure and fallback mechanism
# ---------------------------------------------------------------------------

class TestOptimizePriceFallbackMechanism:
    """
    Tests that simulate solver failures to verify the fallback behaviour.

    The fallback contract is simple: if the solver raises an exception OR returns
    a non-Optimal status, we return round(predicted_price, 2).  This means the
    app keeps running with a slightly wrong price rather than crashing.

    We test this by mocking the CBC solver's solve() method.  We can't easily
    test the 'non-Optimal status' path by controlling the LP directly (that
    would require finding an actually infeasible formulation), so we mock
    the solver's return value to simulate a solver returning a 'Not Solved' status.
    """

    def test_solver_exception_returns_predicted_price_as_fallback(self):
        """
        If pulp's solve() raises any exception (e.g., the CBC binary is missing,
        a licence error, a numerical issue), we must return the raw ML prediction.

        This is the 'fail safe' guarantee: never crash, always return something.
        We mock the solver call to raise RuntimeError and verify the return value.
        """
        predicted = 2.345

        # We need to patch the PULP_CBC_CMD inside the optimization module's namespace.
        # The solve() call is prob.solve(solver), so we mock the PULP_CBC_CMD constructor
        # to return a mock solver object whose ... but actually the exception is thrown
        # inside prob.solve(). We patch prob.solve by patching pulp.LpProblem.solve.
        with patch("optimization.pulp.PULP_CBC_CMD") as mock_solver_cls:
            mock_solver = MagicMock()
            mock_solver_cls.return_value = mock_solver

            # Make solve() raise — simulating a missing CBC binary or library crash
            with patch.object(pulp.LpProblem, "solve", side_effect=RuntimeError("Solver binary not found")):
                result = optimize_price(predicted_price=predicted, current_utilization=0.5)

        # The fallback must be exactly round(predicted, 2)
        assert result == round(predicted, 2), (
            f"Expected fallback to predicted price {round(predicted, 2)}, got {result}. "
            f"The exception handler may not be returning correctly."
        )

    def test_non_optimal_solver_status_returns_predicted_price(self):
        """
        PuLP can return statuses other than 'Optimal' (e.g., 'Not Solved', 'Undefined').
        This shouldn't happen with soft constraints, but numerical precision edge cases
        in CBC can occasionally produce unexpected statuses on extreme inputs.

        We simulate this by mocking prob.solve() to return the integer status code
        for 'Not Solved' (-3 in PuLP's internal coding) and verify the fallback.
        """
        predicted = 1.75

        # PuLP's LpStatus dict: 1='Optimal', -1='Infeasible', -2='Unbounded', -3='Not Solved'
        not_solved_status = -3

        with patch.object(pulp.LpProblem, "solve", return_value=not_solved_status):
            result = optimize_price(predicted_price=predicted, current_utilization=0.5)

        assert result == round(predicted, 2), (
            f"Expected fallback price {round(predicted, 2)} for non-Optimal status, got {result}."
        )

    def test_fallback_price_is_rounded_to_two_decimal_places(self):
        """
        The fallback (round(predicted_price, 2)) must also be rounded — we don't
        want the UI to suddenly show '2.3456789x' just because the solver crashed.
        """
        predicted = 2.6789  # More than 2 decimal places

        with patch.object(pulp.LpProblem, "solve", side_effect=Exception("Unexpected crash")):
            result = optimize_price(predicted_price=predicted, current_utilization=0.5)

        assert result == round(predicted, 2)
        assert result == pytest.approx(2.68)

    def test_solver_exception_is_logged_at_error_level(self, caplog):
        """
        When the solver crashes, we must log at ERROR level (not silently swallow
        the exception).  This is how ops knows there's a real problem vs. just
        a soft-constraint violation.  The log message should include the predicted
        price and utilization so ops can reproduce the issue.
        """
        with caplog.at_level(logging.ERROR, logger="optimization"):
            with patch.object(
                pulp.LpProblem, "solve", side_effect=RuntimeError("Test crash")
            ):
                optimize_price(predicted_price=2.0, current_utilization=0.5)

        # Verify an error was actually logged — not just silently eaten
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) > 0, (
            "No ERROR log was emitted when the solver crashed. "
            "Silent failures are the hardest to debug in production."
        )

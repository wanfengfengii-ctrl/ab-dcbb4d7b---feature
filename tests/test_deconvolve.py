"""Unit tests for the joint integer deconvolution solver."""
from __future__ import annotations

import itertools
import random

import pytest

from app.deconvolve import solve


def convolve(amplitudes, kernel):
    """First len(amplitudes) outputs of the zero-padded discrete convolution."""
    n = len(amplitudes)
    return [
        sum(kernel[j] * amplitudes[i - j] for j in range(len(kernel)) if i - j >= 0)
        for i in range(n)
    ]


def kernel_box(kernel, tolerance):
    """All admissible kernels: each tap independently within +/- its tolerance."""
    return [
        list(k)
        for k in itertools.product(
            *[range(kernel[j] - tolerance[j], kernel[j] + tolerance[j] + 1)
              for j in range(len(kernel))]
        )
    ]


def worst_case_residuals(samples, amplitudes, kernels):
    """Pointwise worst absolute residual and per-point worst over the kernel box."""
    n = len(samples)
    preds = [convolve(amplitudes, k) for k in kernels]
    pointwise = [
        max(abs(samples[i] - p[i]) for p in preds) for i in range(n)
    ]
    return max(pointwise), pointwise



def brute_force(samples, kernel, max_residual, max_amplitude, max_events):
    """Exhaustive reference: optimum key + first unexplainable position.

    Returns ((max_abs, sum_abs, events, amplitudes) or None, first_unexplainable).
    """
    n = len(samples)
    m = len(kernel)
    best_key = None
    explainable = [False] * n
    for x in itertools.product(range(max_amplitude + 1), repeat=n):
        events = sum(1 for v in x if v)
        if events > max_events:
            continue
        residuals = []
        for i in range(n):
            pred = sum(kernel[j] * x[i - j] for j in range(m) if i - j >= 0)
            residuals.append(samples[i] - pred)
        prefix_ok = True
        for i in range(n):
            if abs(residuals[i]) > max_residual:
                prefix_ok = False
            if prefix_ok:
                explainable[i] = True
        if all(abs(r) <= max_residual for r in residuals):
            key = (
                max(abs(r) for r in residuals),
                sum(abs(r) for r in residuals),
                events,
                x,
            )
            if best_key is None or key < best_key:
                best_key = key
    first_unexplainable = next(
        (i for i in range(n) if not explainable[i]), None
    )
    return best_key, first_unexplainable


# Overlapping ground truth used by several tests: events at 4 (amp 5) and 6 (amp 4)
# with kernel [3, 2, 1]; the tails overlap because 6 - 4 < len(kernel).
KERNEL = [3, 2, 1]
EXACT_SAMPLES = [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0]
NOISY_SAMPLES = [0, 0, 0, 0, 15, 11, 17, 7, 4, 0, 0, 0, 0, 0, 0, 0]


class TestExactRecovery:
    def test_overlapping_pulses_recovered_exactly(self):
        result = solve(EXACT_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=4)
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert result.prediction == EXACT_SAMPLES
        assert result.residuals == [0] * 16
        assert result.max_abs_residual == 0
        assert result.sum_abs_residual == 0
        assert result.event_count == 2

    def test_zero_samples_give_zero_events(self):
        result = solve([0] * 12, [2, 1, 1], max_residual=0, max_amplitude=5, max_events=3)
        assert result.feasible
        assert result.event_count == 0
        assert result.amplitudes == [0] * 12
        assert result.prediction == [0] * 12

    def test_single_event_at_last_position(self):
        samples = convolve([0, 0, 0, 0, 0, 3], [2, 1, 1])
        result = solve(samples, [2, 1, 1], max_residual=0, max_amplitude=9, max_events=1)
        assert result.feasible
        assert result.amplitudes == [0, 0, 0, 0, 0, 3]
        assert result.prediction == samples


class TestObjectiveOrder:
    def test_max_residual_is_minimised(self):
        # Exact reconstruction is impossible for the noisy samples, so the
        # smallest achievable max |residual| is 1 and must be reported.
        result = solve(NOISY_SAMPLES, KERNEL, max_residual=1, max_amplitude=10, max_events=6)
        assert result.feasible
        assert result.max_abs_residual == 1
        assert not solve(NOISY_SAMPLES, KERNEL, 0, 10, 6).feasible

    def test_lexicographic_tie_break_on_amplitudes(self):
        # kernel [1,1,1], samples [0,2,0,0,0], radius 1: both [0,1,0,0,0] and
        # [1,0,0,0,0] give (max=1, sum=3, events=1); the lexicographically
        # smaller amplitude sequence must win.
        result = solve([0, 2, 0, 0, 0], [1, 1, 1], max_residual=1, max_amplitude=5, max_events=5)
        assert result.feasible
        assert result.amplitudes == [0, 1, 0, 0, 0]
        assert (result.max_abs_residual, result.sum_abs_residual, result.event_count) == (1, 3, 1)

    def test_sum_of_residuals_minimised_before_event_count(self):
        # samples [3]*6 with kernel [2,1,1], radius 1: a 3-event fit exists
        # ([2,0,0,1,1,0], residual sum 5) but the 4-event fit
        # [1,1,0,1,1,0] has residual sum 3.  The residual sum is minimised
        # before the event count, so the 4-event fit must win.
        result = solve([3] * 6, [2, 1, 1], max_residual=1, max_amplitude=9, max_events=6)
        assert result.feasible
        assert (result.max_abs_residual, result.sum_abs_residual) == (1, 3)
        assert result.event_count == 4
        assert result.amplitudes == [1, 1, 0, 1, 1, 0]


class TestInfeasible:
    def test_residual_limit_reports_first_unexplainable_position(self):
        # NOISY_SAMPLES[5] = 11 cannot be matched exactly: position 4 forces
        # amplitude 5 (15/3), leaving tail 10 at position 5, and 11-10 is not
        # divisible by kernel[0]=3.
        result = solve(NOISY_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=6)
        assert not result.feasible
        assert result.first_unexplainable_position == 5

    def test_event_limit_reports_first_unexplainable_position(self):
        # One event explains samples up to position 5; at position 6 a second
        # event becomes unavoidable, so with max_events=1 position 6 is the
        # first point no complete candidate can explain.
        result = solve(EXACT_SAMPLES, KERNEL, max_residual=0, max_amplitude=10, max_events=1)
        assert not result.feasible
        assert result.first_unexplainable_position == 6

    def test_amplitude_bound_violation_at_first_sample(self):
        result = solve([9, 0, 0, 0, 0, 0], KERNEL, max_residual=0, max_amplitude=2, max_events=6)
        assert not result.feasible
        assert result.first_unexplainable_position == 0

    def test_negative_sample_beyond_tolerance(self):
        samples = [0] * 12
        samples[3] = -5
        result = solve(samples, KERNEL, max_residual=2, max_amplitude=5, max_events=3)
        assert not result.feasible
        assert result.first_unexplainable_position == 3

    def test_zero_events_allowed(self):
        result = solve([1, -1, 0, 0, 0, 0], KERNEL, max_residual=1, max_amplitude=5, max_events=0)
        assert result.feasible
        assert result.event_count == 0
        assert result.amplitudes == [0] * 6


class TestBruteForceCrossCheck:
    @pytest.mark.parametrize("seed", range(60))
    def test_matches_exhaustive_search(self, seed):
        rng = random.Random(seed)
        n, m = 7, 3
        kernel = [rng.randint(1, 3) for _ in range(m)]
        max_amplitude = 2
        max_events = rng.randint(0, 4)
        max_residual = rng.randint(0, 3)
        if seed % 2 == 0:
            true_x = [rng.randint(0, max_amplitude) for _ in range(n)]
            samples = [v + rng.randint(-1, 1) for v in convolve(true_x, kernel)]
        else:
            samples = [rng.randint(-2, 15) for _ in range(n)]

        result = solve(samples, kernel, max_residual, max_amplitude, max_events)
        best_key, first_unexplainable = brute_force(
            samples, kernel, max_residual, max_amplitude, max_events
        )

        if best_key is None:
            assert not result.feasible
            assert result.first_unexplainable_position == first_unexplainable
            return

        assert result.feasible
        exp_max, exp_sum, exp_events, exp_x = best_key
        assert result.max_abs_residual == exp_max
        assert result.sum_abs_residual == exp_sum
        assert result.event_count == exp_events
        assert tuple(result.amplitudes) == exp_x
        # Response consistency: prediction and residuals match the amplitudes.
        assert result.prediction == convolve(result.amplitudes, kernel)
        assert result.residuals == [samples[i] - result.prediction[i] for i in range(n)]
        assert result.event_count == sum(1 for v in result.amplitudes if v)


class TestRobustKernel:
    """Kernel uncertainty: the event train must cover every admissible kernel."""

    def test_zero_tolerance_matches_nominal_solver(self):
        result = solve(NOISY_SAMPLES, KERNEL, max_residual=1, max_amplitude=10, max_events=6)
        robust = solve(
            NOISY_SAMPLES, KERNEL, max_residual=1, max_amplitude=10, max_events=6,
            kernel_tolerance=[0, 0, 0],
        )
        assert robust.feasible
        assert robust.amplitudes == result.amplitudes
        assert (
            robust.max_abs_residual, robust.sum_abs_residual, robust.event_count
        ) == (result.max_abs_residual, result.sum_abs_residual, result.event_count)
        assert robust.prediction == result.prediction
        assert robust.residuals == result.residuals
        assert robust.prediction_intervals == [[v, v] for v in result.prediction]
        assert robust.residual_intervals == [[v, v] for v in result.residuals]

    def test_robust_success_recovers_true_train_and_reports_intervals(self):
        # Nominal calibration says kernel [3,2,1] with first tap in {2,3,4};
        # the samples match the nominal kernel, and the true train must keep
        # covering the two extreme tap values within the residual limit.
        true_x = [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        samples = convolve(true_x, [3, 2, 1])
        result = solve(
            samples, [3, 2, 1], max_residual=5, max_amplitude=10, max_events=4,
            kernel_tolerance=[1, 0, 0],
        )
        assert result.feasible
        assert result.amplitudes == true_x
        kernels = kernel_box([3, 2, 1], [1, 0, 0])
        # Every admissible kernel must keep every residual within the limit.
        worst_max, pointwise = worst_case_residuals(samples, result.amplitudes, kernels)
        assert worst_max <= 5
        assert result.max_abs_residual == worst_max
        assert result.sum_abs_residual == sum(pointwise)
        # Nominal prediction/residuals are preserved alongside the intervals.
        assert result.prediction == convolve(true_x, [3, 2, 1])
        assert result.residuals == [
            samples[i] - result.prediction[i] for i in range(len(samples))
        ]
        n = len(samples)
        for i in range(n):
            preds = [convolve(true_x, k)[i] for k in kernels]
            assert result.prediction_intervals[i] == [min(preds), max(preds)]
            resids = [samples[i] - p for p in preds]
            assert result.residual_intervals[i] == [min(resids), max(resids)]
            lo, hi = result.prediction_intervals[i]
            assert lo <= result.prediction[i] <= hi
            assert max(abs(result.residual_intervals[i][0]),
                       abs(result.residual_intervals[i][1])) == pointwise[i]

    def test_robust_residual_limit_reports_first_unexplainable_position(self):
        # With the first tap in {2,3,4} a 15-count sample at position 4 cannot
        # be matched exactly by any single amplitude, so position 4 is first.
        result = solve(
            EXACT_SAMPLES, [3, 2, 1], max_residual=0, max_amplitude=10,
            max_events=6, kernel_tolerance=[1, 0, 0],
        )
        assert not result.feasible
        assert result.first_unexplainable_position == 4

    def test_robust_event_limit_reports_first_unexplainable_position(self):
        # Data produced by the low-tap kernel; covering the tails robustly
        # forces a second event at position 6.
        samples = convolve(
            [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 1]
        )
        result = solve(
            samples, [3, 2, 1], max_residual=5, max_amplitude=10,
            max_events=1, kernel_tolerance=[1, 0, 0],
        )
        assert not result.feasible
        assert result.first_unexplainable_position == 6

    def test_robust_solution_covers_every_admissible_kernel(self):
        kernel = [3, 2, 2, 1]
        tolerance = [1, 0, 1, 0]
        true_x = [0, 0, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0]
        rng = random.Random(23)
        samples = [
            v + rng.choice((-1, 0, 1)) for v in convolve(true_x, kernel)
        ]
        # A feasible limit: the worst-case radius the ground-truth train needs.
        worst_max, _ = worst_case_residuals(
            samples, true_x, kernel_box(kernel, tolerance)
        )
        result = solve(
            samples, kernel, max_residual=worst_max, max_amplitude=5,
            max_events=8, kernel_tolerance=tolerance,
        )
        assert result.feasible
        assert result.max_abs_residual <= worst_max
        for k in kernel_box(kernel, tolerance):
            pred = convolve(result.amplitudes, k)
            assert all(abs(samples[i] - pred[i]) <= worst_max
                       for i in range(len(samples)))


class TestRobustBruteForceCrossCheck:
    @pytest.mark.parametrize("seed", range(40))
    def test_matches_exhaustive_search(self, seed):
        rng = random.Random(1000 + seed)
        n, m = 6, 3
        kernel = [rng.randint(2, 3) for _ in range(m)]
        tolerance = [rng.randint(0, 1) for _ in range(m)]
        max_amplitude = 2
        max_events = rng.randint(0, 4)
        max_residual = rng.randint(0, 3)
        if seed % 2 == 0:
            true_x = [rng.randint(0, max_amplitude) for _ in range(n)]
            true_kernel = [
                kernel[j] + rng.randint(-tolerance[j], tolerance[j]) for j in range(m)
            ]
            samples = [v + rng.randint(-1, 1) for v in convolve(true_x, true_kernel)]
        else:
            samples = [rng.randint(-2, 15) for _ in range(n)]

        result = solve(
            samples, kernel, max_residual, max_amplitude, max_events,
            kernel_tolerance=tolerance,
        )
        kernels = kernel_box(kernel, tolerance)
        best_key = None
        explainable = [False] * n
        for x in itertools.product(range(max_amplitude + 1), repeat=n):
            events = sum(1 for v in x if v)
            if events > max_events:
                continue
            pointwise = [
                max(abs(samples[i] - convolve(x, k)[i]) for k in kernels)
                for i in range(n)
            ]
            prefix_ok = True
            for i, worst in enumerate(pointwise):
                if worst > max_residual:
                    prefix_ok = False
                if prefix_ok:
                    explainable[i] = True
            if max(pointwise) <= max_residual:
                key = (max(pointwise), sum(pointwise), events, x)
                if best_key is None or key < best_key:
                    best_key = key
        first_unexplainable = next(
            (i for i in range(n) if not explainable[i]), None
        )

        if best_key is None:
            assert not result.feasible
            assert result.first_unexplainable_position == first_unexplainable
            return

        assert result.feasible
        exp_max, exp_sum, exp_events, exp_x = best_key
        assert result.max_abs_residual == exp_max
        assert result.sum_abs_residual == exp_sum
        assert result.event_count == exp_events
        assert tuple(result.amplitudes) == exp_x
        for i in range(n):
            preds = [convolve(exp_x, k)[i] for k in kernels]
            assert result.prediction_intervals[i] == [min(preds), max(preds)]
            resids = [samples[i] - p for p in preds]
            assert result.residual_intervals[i] == [min(resids), max(resids)]
        assert result.prediction == convolve(list(exp_x), kernel)


class TestScaling:
    def test_full_size_exact_instance(self):
        rng = random.Random(7)
        n, m = 36, 6
        kernel = [rng.randint(1, 4) for _ in range(m)]
        true_x = [rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(n)]
        samples = convolve(true_x, kernel)
        result = solve(samples, kernel, max_residual=0, max_amplitude=9, max_events=n)
        assert result.feasible
        assert result.max_abs_residual == 0
        assert result.prediction == samples
        # Zero-residual solutions of a lower-triangular system are unique.
        assert result.amplitudes == true_x

    def test_full_size_noisy_instance(self):
        rng = random.Random(11)
        n, m = 36, 6
        kernel = [rng.randint(2, 4) for _ in range(m)]
        true_x = [rng.choice([0, 0, 1, 2]) for _ in range(n)]
        samples = [v + rng.randint(-1, 1) for v in convolve(true_x, kernel)]
        result = solve(samples, kernel, max_residual=2, max_amplitude=9, max_events=n)
        assert result.feasible
        assert result.max_abs_residual <= 2
        assert len(result.prediction) == n
        assert result.residuals == [samples[i] - result.prediction[i] for i in range(n)]

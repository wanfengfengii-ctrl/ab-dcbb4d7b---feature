"""Joint non-negative integer deconvolution of overlapping scintillator pulses.

Model
-----
Observed samples ``s[0..n-1]`` are modelled as the first ``n`` outputs of the
zero-padded discrete convolution of an event amplitude train ``x[0..n-1]``
(non-negative integers, each at most ``max_amplitude``) with a positive
integer response kernel ``k[0..m-1]``::

    pred[i] = sum_{j=0}^{m-1} k[j] * x[i-j]        (x[t] = 0 for t < 0)

An *event* is a position whose amplitude is non-zero.  Residuals are
``residuals[i] = samples[i] - prediction[i]``.

Finite-precision calibration: kernel uncertainty
-------------------------------------------------
``kernel_tolerance`` (when given) assigns each kernel tap a non-negative
integer tolerance ``tol[j]``; the true tap is then any integer in
``[kernel[j]-tol[j], kernel[j]+tol[j]]`` (the validated lowest possible value
is still positive).  A recovered event train must keep the pointwise residual
within ``max_residual`` for *every* admissible kernel -- robust feasibility is
a box constraint on the linear prediction, since its minimum and maximum over
the kernel box are attained (componentwise) at the extreme tap values.  The
admissible values of ``x[p]`` still form a contiguous integer interval, so the
same banded-lower-triangular dynamic program applies.

Objectives (minimised lexicographically, in this order)
-------------------------------------------------------
1. worst-case ``max |residual[i]|`` over all admissible kernels -- hard limit
   ``max_residual``; otherwise infeasible
2. pointwise worst-case ``sum_i max_kernel |residual[i]|``
3. number of events          -- hard limit ``max_events``; otherwise infeasible
4. the amplitude sequence ``x`` itself, in lexicographic order

Algorithm
---------
The convolution matrix is banded lower-triangular: once ``x[0..p-1]`` are
fixed, sample ``p`` depends only on the last ``m-1`` amplitudes.  For a given
per-point residual radius the admissible values of ``x[p]`` form a contiguous
integer interval even under kernel uncertainty, so the joint optimum is found
by an exact dynamic program over "tail" states (the last ``m-1`` amplitudes
plus the events used so far) instead of greedy peak-by-peak tail subtraction:

* a feasibility scan finds, for a given radius, whether any complete candidate
  satisfies the limits robustly, and the first sample position no complete
  candidate can explain;
* doubling from radius 0 then binary search over the (integer) radius yields
  the minimal achievable worst-case max |residual|;
* a final dynamic program minimises (pointwise worst-case |residual| sum,
  events, amplitude sequence) under that radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class SolveResult:
    feasible: bool
    amplitudes: Optional[list[int]] = None
    prediction: Optional[list[int]] = None
    residuals: Optional[list[int]] = None
    max_abs_residual: Optional[int] = None
    sum_abs_residual: Optional[int] = None
    event_count: Optional[int] = None
    first_unexplainable_position: Optional[int] = None
    # Populated only when kernel uncertainty is enabled.
    prediction_intervals: Optional[list[list[int]]] = None
    residual_intervals: Optional[list[list[int]]] = None


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for integers with b > 0 (handles negative a)."""
    return -((-a) // b)


class _KernelModel:
    """Prediction bounds for a single sample over the admissible kernel box.

    Given fixed non-negative amplitudes, ``P = K0*v + T`` where ``v`` is the
    newest amplitude and ``T`` collects the contributions of the preceding
    ``m-1`` amplitudes.  Every tap ranges independently in
    ``[kl[j], ku[j]]``; as all amplitude factors are non-negative, ``K0*v`` and
    each tail term attain their extrema at tap box corners, hence:

        P in [kl0*v + tl, ku0*v + tu]

    where ``tl`` (resp. ``tu``) uses the low (resp. high) tap bounds.
    """

    __slots__ = ("m", "nominal", "kl", "ku", "wl", "wu")

    def __init__(self, kernel: Sequence[int], tolerance: Optional[Sequence[int]]):
        self.m = len(kernel)
        self.nominal = list(kernel)
        if tolerance is None:
            self.kl = list(kernel)
            self.ku = list(kernel)
        else:
            self.kl = [kernel[j] - tolerance[j] for j in range(self.m)]
            self.ku = [kernel[j] + tolerance[j] for j in range(self.m)]
        # Tail state tail[i] holds x[p-m+1+i], contributing tap m-1-i.
        self.wl = [self.kl[self.m - 1 - i] for i in range(self.m - 1)]
        self.wu = [self.ku[self.m - 1 - i] for i in range(self.m - 1)]

    def tail_bounds(self, tail: Sequence[int]) -> tuple[int, int]:
        tl = tu = 0
        for i in range(self.m - 1):
            ti = tail[i]
            if ti:
                tl += self.wl[i] * ti
                tu += self.wu[i] * ti
        return tl, tu

    def pred_bounds(self, v: int, tl: int, tu: int) -> tuple[int, int]:
        return self.kl[0] * v + tl, self.ku[0] * v + tu

    def robust_residual(self, sample: int, plo: int, phi: int) -> int:
        """max_{P in [plo, phi]} |sample - P|."""
        if sample < plo:
            return plo - sample
        if sample > phi:
            return sample - phi
        return max(sample - plo, phi - sample)


def _allowed_range(
    sample: int,
    radius: int,
    tl: int,
    tu: int,
    kl0: int,
    ku0: int,
    max_amplitude: int,
) -> tuple[int, int]:
    """Integer v in [0, max_amplitude] with every residual of ``sample`` over
    the prediction box ``[kl0*v+tl, ku0*v+tu]`` within ``+/-radius``."""
    # For every prediction P in [kl0*v+tl, ku0*v+tu] we need
    # |sample - P| <= radius, equivalent to
    #   sample-radius <= kl0*v+tl  ->  v >= (sample-radius-tl) / kl0
    #   ku0*v+tu       <= sample+radius ->  v <= (sample+radius-tu) / ku0
    lo = _ceil_div(sample - radius - tl, kl0)
    hi = (sample + radius - tu) // ku0
    if lo < 0:
        lo = 0
    if hi > max_amplitude:
        hi = max_amplitude
    return lo, hi


def _scan(
    samples: Sequence[int],
    model: _KernelModel,
    radius: int,
    max_amplitude: int,
    max_events: int,
) -> tuple[bool, Optional[int]]:
    """Robust feasibility and first unexplainable position for a radius.

    Returns ``(feasible, first_fail)``.  ``first_fail`` is the smallest sample
    index ``p`` for which no complete candidate (amplitudes within
    ``[0, max_amplitude]``, at most ``max_events`` non-zero amplitudes) can
    keep every residual of samples ``0..p`` within ``+/-radius`` for *every*
    admissible kernel; ``None`` when all samples can be covered.
    """
    n = len(samples)
    m = model.m
    kl0, ku0 = model.kl[0], model.ku[0]
    # states: (tail tuple of the last m-1 amplitudes, events used so far)
    states = {((0,) * (m - 1), 0)}
    for p in range(n):
        sp = samples[p]
        nxt: set[tuple[tuple[int, ...], int]] = set()
        for tail, used in states:
            tl, tu = model.tail_bounds(tail)
            lo, hi = _allowed_range(sp, radius, tl, tu, kl0, ku0, max_amplitude)
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                nxt.add((tail[1:] + (v,), used2))
        if not nxt:
            return False, p
        states = nxt
    return True, None


def _optimize(
    samples: Sequence[int],
    model: _KernelModel,
    radius: int,
    max_amplitude: int,
    max_events: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Minimise (pointwise worst-case |residual| sum, events, amplitudes) with
    the worst-case per-point |residual| at most ``radius``."""
    n = len(samples)
    m = model.m
    kl0, ku0 = model.kl[0], model.ku[0]
    # (tail, events used) -> (worst-case |residual| sum so far, amplitude path)
    states: dict[tuple[tuple[int, ...], int], tuple[int, tuple[int, ...]]] = {
        ((0,) * (m - 1), 0): (0, ())
    }
    for p in range(n):
        sp = samples[p]
        nxt: dict[tuple[tuple[int, ...], int], tuple[int, tuple[int, ...]]] = {}
        for (tail, used), (cost, path) in states.items():
            tl, tu = model.tail_bounds(tail)
            lo, hi = _allowed_range(sp, radius, tl, tu, kl0, ku0, max_amplitude)
            for v in range(lo, hi + 1):
                used2 = used + (1 if v else 0)
                if used2 > max_events:
                    continue
                plo, phi = model.pred_bounds(v, tl, tu)
                step = model.robust_residual(sp, plo, phi)
                key = (tail[1:] + (v,), used2)
                cand = (cost + step, path + (v,))
                cur = nxt.get(key)
                if cur is None or cand < cur:
                    nxt[key] = cand
        if not nxt:  # radius below the feasible minimum; solve() prevents this
            raise RuntimeError("no candidate within the given residual radius")
        states = nxt
    return min(
        (cost, used, path) for (tail, used), (cost, path) in states.items()
    )


def _convolve_bounds(
    amplitudes: Sequence[int], model: _KernelModel
) -> tuple[list[int], list[int], list[int]]:
    """Pointwise prediction box [lo, hi] and the nominal-kernel prediction."""
    n = len(amplitudes)
    m = model.m
    pred_lo: list[int] = []
    pred_hi: list[int] = []
    nominal: list[int] = []
    for i in range(n):
        lo = hi = nom = 0
        for j in range(m):
            t = i - j
            if t >= 0 and amplitudes[t]:
                a = amplitudes[t]
                lo += model.kl[j] * a
                hi += model.ku[j] * a
                nom += model.nominal[j] * a
        pred_lo.append(lo)
        pred_hi.append(hi)
        nominal.append(nom)
    return pred_lo, pred_hi, nominal


def solve(
    samples: Sequence[int],
    kernel: Sequence[int],
    max_residual: int,
    max_amplitude: int,
    max_events: int,
    kernel_tolerance: Optional[Sequence[int]] = None,
) -> SolveResult:
    """Jointly deconvolve ``samples`` under the given limits.

    With ``kernel_tolerance`` given, every admissible kernel (each tap within
    its nominal value +/- tolerance) must satisfy the residual limit; the
    objectives and the reported residual figures then use the per-kernel
    worst case.  When infeasible, only ``feasible`` and
    ``first_unexplainable_position`` are populated.
    """
    model = _KernelModel(kernel, kernel_tolerance)
    robust = kernel_tolerance is not None

    # Find the smallest feasible radius.  Scans at a tight radius are cheap
    # (few amplitude choices per state) while scans at a loose radius can fan
    # out over many tails, so double from 0 until a feasible radius brackets
    # the optimum within a factor of two instead of probing the (possibly
    # huge) user-supplied limit first.
    lo_r, hi_r = -1, 0
    while True:
        ok, fail_at_hi = _scan(samples, model, hi_r, max_amplitude, max_events)
        if ok:
            break
        if hi_r == max_residual:
            return SolveResult(
                feasible=False, first_unexplainable_position=fail_at_hi
            )
        lo_r = hi_r
        hi_r = min(max_residual, max(1, hi_r * 2))

    # Samples, kernel and amplitudes are integers, so residuals are integers
    # and the optimal worst-case max |residual| is an integer; binary-search
    # between the last infeasible probe (lo_r) and the first feasible one.
    while hi_r - lo_r > 1:
        mid = (lo_r + hi_r) // 2
        ok, _ = _scan(samples, model, mid, max_amplitude, max_events)
        if ok:
            hi_r = mid
        else:
            lo_r = mid
    radius = hi_r

    sum_abs, event_count, path = _optimize(
        samples, model, radius, max_amplitude, max_events
    )
    amplitudes = list(path)
    pred_lo, pred_hi, nominal_pred = _convolve_bounds(amplitudes, model)

    if robust:
        n = len(samples)
        prediction = nominal_pred
        residuals = [samples[i] - prediction[i] for i in range(n)]
        residual_intervals = [
            [samples[i] - pred_hi[i], samples[i] - pred_lo[i]] for i in range(n)
        ]
        max_abs = max(
            max(abs(rlo), abs(rhi)) for rlo, rhi in residual_intervals
        )
        return SolveResult(
            feasible=True,
            amplitudes=amplitudes,
            prediction=prediction,
            residuals=residuals,
            max_abs_residual=max_abs,
            sum_abs_residual=sum_abs,
            event_count=event_count,
            prediction_intervals=[
                [pred_lo[i], pred_hi[i]] for i in range(n)
            ],
            residual_intervals=residual_intervals,
        )

    residuals = [samples[i] - nominal_pred[i] for i in range(len(samples))]
    max_abs = max(abs(r) for r in residuals)
    return SolveResult(
        feasible=True,
        amplitudes=amplitudes,
        prediction=nominal_pred,
        residuals=residuals,
        max_abs_residual=max_abs,
        sum_abs_residual=sum_abs,
        event_count=event_count,
    )

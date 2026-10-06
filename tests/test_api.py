"""API tests for POST /api/pulses/deconvolve and the health endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

KERNEL = [3, 2, 1]
EXACT_SAMPLES = [0, 0, 0, 0, 15, 10, 17, 8, 4, 0, 0, 0, 0, 0, 0, 0]
NOISY_SAMPLES = [0, 0, 0, 0, 15, 11, 17, 7, 4, 0, 0, 0, 0, 0, 0, 0]


def make_payload(**overrides):
    payload = {
        "samples": EXACT_SAMPLES,
        "kernel": KERNEL,
        "maxResidual": 0,
        "maxAmplitude": 10,
        "maxEvents": 4,
    }
    payload.update(overrides)
    return payload


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_exact_overlapping_pulses():
    resp = client.post("/api/pulses/deconvolve", json=make_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["events"] == [
        {"position": 4, "amplitude": 5},
        {"position": 6, "amplitude": 4},
    ]
    assert body["amplitudes"] == [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert body["prediction"] == EXACT_SAMPLES
    assert body["residuals"] == [0] * 16
    assert body["objectives"] == {
        "maxAbsResidual": 0,
        "sumAbsResidual": 0,
        "eventCount": 2,
    }


def test_noisy_pulses_within_residual_limit():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(samples=NOISY_SAMPLES, maxResidual=1),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["objectives"]["maxAbsResidual"] <= 1
    assert len(body["prediction"]) == len(NOISY_SAMPLES)
    assert body["residuals"] == [
        NOISY_SAMPLES[i] - body["prediction"][i] for i in range(len(NOISY_SAMPLES))
    ]


def test_infeasible_residual_limit_is_identifiable():
    resp = client.post(
        "/api/pulses/deconvolve",
        json=make_payload(samples=NOISY_SAMPLES, maxResidual=0),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["firstUnexplainablePosition"] == 5


def test_infeasible_event_limit_is_identifiable():
    resp = client.post("/api/pulses/deconvolve", json=make_payload(maxEvents=1))
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["firstUnexplainablePosition"] == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"samples": EXACT_SAMPLES[:11]},            # too few samples
        {"samples": EXACT_SAMPLES + [0] * 21},      # too many samples (37)
        {"kernel": [2, 1]},                         # kernel too short
        {"kernel": [1, 1, 1, 1, 1, 1, 1]},          # kernel too long
        {"kernel": [3, 0, 1]},                      # non-positive kernel entry
        {"kernel": [3, -2, 1]},                     # negative kernel entry
        {"maxResidual": -1},                        # negative residual limit
        {"maxAmplitude": 0},                        # amplitude limit must be >= 1
        {"maxEvents": -1},                          # negative event limit
        {"samples": ["a"] + EXACT_SAMPLES[1:]},     # non-integer sample
    ],
)
def test_invalid_requests_return_400(overrides):
    resp = client.post("/api/pulses/deconvolve", json=make_payload(**overrides))
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid_request"


def test_missing_field_returns_400():
    payload = make_payload()
    del payload["kernel"]
    resp = client.post("/api/pulses/deconvolve", json=payload)
    assert resp.status_code == 400


def test_deterministic_responses():
    first = client.post("/api/pulses/deconvolve", json=make_payload(maxResidual=1, samples=NOISY_SAMPLES))
    second = client.post("/api/pulses/deconvolve", json=make_payload(maxResidual=1, samples=NOISY_SAMPLES))
    assert first.status_code == 200
    assert first.json() == second.json()


class TestKernelTolerance:
    def test_nominal_response_shape_is_unchanged(self):
        resp = client.post("/api/pulses/deconvolve", json=make_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "status", "events", "amplitudes", "prediction",
            "residuals", "objectives",
        }

    def test_nominal_infeasible_shape_is_unchanged(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(samples=NOISY_SAMPLES, maxResidual=0),
        )
        assert resp.status_code == 422
        assert set(resp.json()) == {"status", "detail", "firstUnexplainablePosition"}

    def test_robust_success(self):
        tolerance = [1, 1, 0]
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(maxResidual=30, maxEvents=6,
                              kernelTolerance=tolerance),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert body["robust"] is True
        n = len(EXACT_SAMPLES)
        assert len(body["predictionIntervals"]) == n
        assert len(body["residualIntervals"]) == n
        # Nominal figures are retained and lie within the closed intervals.
        for i in range(n):
            plo, phi = body["predictionIntervals"][i]
            assert plo <= body["prediction"][i] <= phi
            rlo, rhi = body["residualIntervals"][i]
            assert rlo <= body["residuals"][i] <= rhi
            assert [EXACT_SAMPLES[i] - phi, EXACT_SAMPLES[i] - plo] == [rlo, rhi]
        # Worst-case objectives are consistent with the residual intervals.
        worst = [max(abs(lo), abs(hi)) for lo, hi in body["residualIntervals"]]
        assert body["objectives"]["maxAbsResidual"] == max(worst)
        assert body["objectives"]["sumAbsResidual"] == sum(worst)
        # The returned train covers every allowed kernel within the radius.
        offsets = [(d0, d1, d2)
                   for d0 in (-1, 0, 1)
                   for d1 in (-1, 0, 1)
                   for d2 in (0,)]
        x = body["amplitudes"]
        for delta in offsets:
            k = [KERNEL[j] + delta[j] for j in range(3)]
            for i in range(n):
                pred = sum(
                    k[j] * x[i - j] for j in range(3) if i - j >= 0
                )
                assert abs(EXACT_SAMPLES[i] - pred) <= 30
        # The interval extrema are actually attained by allowed kernels.
        for i in range(n):
            preds = [
                sum((KERNEL[j] + delta[j]) * x[i - j]
                    for j in range(3) if i - j >= 0)
                for delta in offsets
            ]
            assert body["predictionIntervals"][i] == [min(preds), max(preds)]

    def test_robust_infeasible_is_identifiable(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(maxResidual=0, maxEvents=6,
                              kernelTolerance=[1, 1, 0]),
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "infeasible"
        assert body["robust"] is True
        assert body["firstUnexplainablePosition"] == 4

    def test_zero_tolerance_gives_nominal_intervals(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(maxResidual=1, samples=NOISY_SAMPLES,
                              kernelTolerance=[0, 0, 0]),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["robust"] is True
        assert body["predictionIntervals"] == [[p, p] for p in body["prediction"]]
        assert body["residualIntervals"] == [[r, r] for r in body["residuals"]]

    @pytest.mark.parametrize(
        "tolerance",
        [
            [1, 1],                 # shorter than kernel
            [1, 1, 1, 1],           # longer than kernel
            [-1, 0, 0],             # negative entry
            [3, 0, 0],              # minimum tap would be non-positive
            [1, 2, 1],              # middle tap 2 - 2 = 0
            ["a", 0, 0],            # non-integer entry
        ],
    )
    def test_invalid_kernel_tolerance_returns_400(self, tolerance):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(kernelTolerance=tolerance),
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["status"] == "invalid_request"
        assert any(
            "kernelTolerance" in error["loc"] for error in body["detail"]
        )

    def test_empty_kernel_tolerance_rejected_per_field(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(kernelTolerance=[]),
        )
        assert resp.status_code == 400
        assert any(
            "kernelTolerance" in error["loc"]
            for error in resp.json()["detail"]
        )

    def test_kernel_tolerance_with_invalid_kernel_still_400(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(kernel=[3, 0, 1], kernelTolerance=[0, 0, 0]),
        )
        assert resp.status_code == 400
        assert resp.json()["status"] == "invalid_request"

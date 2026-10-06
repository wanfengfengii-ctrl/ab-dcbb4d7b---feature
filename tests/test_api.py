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


class TestKernelToleranceCompatibility:
    def test_omitted_tolerance_response_has_no_interval_fields(self):
        resp = client.post("/api/pulses/deconvolve", json=make_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert "predictionIntervals" not in body
        assert "residualIntervals" not in body
        assert body["prediction"] == EXACT_SAMPLES
        assert body["residuals"] == [0] * 16
        assert body["objectives"] == {
            "maxAbsResidual": 0, "sumAbsResidual": 0, "eventCount": 2,
        }

    def test_omitted_tolerance_infeasible_semantics_unchanged(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(samples=NOISY_SAMPLES, maxResidual=0),
        )
        assert resp.status_code == 422
        assert resp.json() == {
            "status": "infeasible",
            "detail": (
                "no event train within the amplitude and event-count limits can "
                "reproduce the samples within the residual limit"
            ),
            "firstUnexplainablePosition": 5,
        }


class TestKernelToleranceRobust:
    def test_robust_success_returns_nominal_and_intervals(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(kernelTolerance=[1, 0, 0], maxResidual=5),
        )
        assert resp.status_code == 200, resp.json()
        body = resp.json()
        n = len(EXACT_SAMPLES)
        assert body["events"] == [
            {"position": 4, "amplitude": 5},
            {"position": 6, "amplitude": 4},
        ]
        # Nominal prediction/residuals are preserved.
        assert body["prediction"] == EXACT_SAMPLES
        assert body["residuals"] == [0] * n
        intervals = body["predictionIntervals"]
        res_intervals = body["residualIntervals"]
        assert len(intervals) == n and len(res_intervals) == n
        for pair in intervals + res_intervals:
            assert len(pair) == 2 and pair[0] <= pair[1]
        # Uncertain points: position 4 spans 10..20, position 6 spans 13..21;
        # all other points stay pinned at the sample value.
        assert intervals[4] == [10, 20]
        assert intervals[6] == [13, 21]
        assert res_intervals[4] == [-5, 5]
        assert res_intervals[6] == [-4, 4]
        assert all(intervals[i] == [EXACT_SAMPLES[i], EXACT_SAMPLES[i]]
                   for i in range(n) if i not in (4, 6))
        objectives = body["objectives"]
        assert objectives == {
            "maxAbsResidual": 5, "sumAbsResidual": 9, "eventCount": 2,
        }
        # Each interval contains the nominal value.
        assert all(intervals[i][0] <= body["prediction"][i] <= intervals[i][1]
                   for i in range(n))

    def test_robust_infeasible_reports_first_unexplainable_position(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(kernelTolerance=[1, 0, 0], maxResidual=0),
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["status"] == "infeasible"
        assert body["firstUnexplainablePosition"] == 4

    def test_zero_tolerance_keeps_nominal_success(self):
        resp = client.post(
            "/api/pulses/deconvolve",
            json=make_payload(kernelTolerance=[0, 0, 0]),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["amplitudes"] == [0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert body["predictionIntervals"] == [[v, v] for v in EXACT_SAMPLES]
        assert body["residualIntervals"] == [[0, 0] for _ in EXACT_SAMPLES]


@pytest.mark.parametrize(
    "tolerance",
    [
        [0, 0],            # shorter than kernel
        [0, 0, 0, 0],      # longer than kernel
        [-1, 0, 0],        # negative entry
        [3, 0, 0],         # kernel[0] - 3 == 0: minimum must stay positive
        [0, 3, 0],         # kernel[1] - 3 < 0
        [0, 0, "x"],       # non-integer entry
        [0, 0, 0, 0, 0, 0, 0],  # too many entries for the allowed kernel length
    ],
)
def test_invalid_kernel_tolerance_returns_400(tolerance):
    resp = client.post(
        "/api/pulses/deconvolve", json=make_payload(kernelTolerance=tolerance)
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "invalid_request"
    # The error is attributed to the kernelTolerance field itself.
    locs = [tuple(err["loc"]) for err in body["detail"]]
    assert any("kernelTolerance" in loc for loc in locs), body

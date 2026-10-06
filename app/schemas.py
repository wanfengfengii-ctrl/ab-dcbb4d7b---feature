"""Request/response schemas for the deconvolution API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class DeconvolveRequest(BaseModel):
    """Input for POST /api/pulses/deconvolve."""

    samples: List[int] = Field(
        ..., min_length=12, max_length=36,
        description="Observed integer samples (12..36 values).",
    )
    kernel: List[int] = Field(
        ..., min_length=3, max_length=6,
        description="Positive integer impulse-response kernel (3..6 values).",
    )
    maxResidual: int = Field(
        ..., ge=0, le=1_000_000_000,
        description="Uniform per-sample absolute residual limit.",
    )
    maxAmplitude: int = Field(
        ..., ge=1, le=1_000_000,
        description="Upper bound for each event's integer amplitude.",
    )
    maxEvents: int = Field(
        ..., ge=0, le=10_000,
        description="Maximum number of events (non-zero amplitudes).",
    )
    kernelTolerance: Optional[List[int]] = Field(
        default=None,
        min_length=3,
        max_length=6,
        description=(
            "Optional non-negative integer per-tap calibration tolerance. "
            "Each kernel tap may then be any integer in [kernel[j]-tol[j], "
            "kernel[j]+tol[j]]; the lowest possible value of every tap must "
            "stay positive and the list length must equal the kernel length. "
            "When given, the recovered event train must satisfy the residual "
            "limit for every admissible kernel."
        ),
    )

    @field_validator("kernel")
    @classmethod
    def _kernel_positive(cls, value: List[int]) -> List[int]:
        for k in value:
            if k < 1:
                raise ValueError("kernel entries must be positive integers")
            if k > 1_000_000:
                raise ValueError("kernel entries must be <= 1000000")
        return value

    @field_validator("kernelTolerance")
    @classmethod
    def _kernel_tolerance_valid(cls, value: Optional[List[int]], info: ValidationInfo) -> Optional[List[int]]:
        if value is None:
            return None
        for t in value:
            if t < 0:
                raise ValueError("kernelTolerance entries must be non-negative integers")
        # ``kernel`` is validated first (field declaration order), so it is
        # available here when it passed validation.
        kernel = info.data.get("kernel")
        if kernel is not None:
            if len(value) != len(kernel):
                raise ValueError(
                    "kernelTolerance length must equal the kernel length"
                )
            for j, (k, t) in enumerate(zip(kernel, value)):
                if k - t < 1:
                    raise ValueError(
                        f"kernel tap {j} minus its tolerance must remain positive"
                    )
        return value


class Event(BaseModel):
    position: int = Field(..., description="Sample index where the event starts.")
    amplitude: int = Field(..., description="Positive integer amplitude of the event.")


class Objectives(BaseModel):
    maxAbsResidual: int = Field(
        description=(
            "Worst-case maximum absolute residual; with kernelTolerance this "
            "is the maximum over every admissible kernel."
        )
    )
    sumAbsResidual: int = Field(
        description=(
            "Pointwise worst-case absolute-residual sum; with kernelTolerance "
            "each point contributes the largest |residual| attained by any "
            "admissible kernel."
        )
    )
    eventCount: int


class DeconvolveSuccessResponse(BaseModel):
    status: str = "ok"
    events: List[Event] = Field(..., description="Recovered events (position/amplitude).")
    amplitudes: List[int] = Field(..., description="Full amplitude sequence (one entry per sample).")
    prediction: List[int] = Field(
        ...,
        description=(
            "Predicted waveform for the nominal kernel (midpoint of each tap's "
            "admissible box under kernelTolerance), aligned with the samples."
        ),
    )
    residuals: List[int] = Field(
        ...,
        description="Pointwise residuals for the nominal prediction: samples - prediction.",
    )
    objectives: Objectives
    predictionIntervals: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "Present only with kernelTolerance: for each sample the closed "
            "interval [low, high] of predictions attained over all admissible "
            "kernels."
        ),
    )
    residualIntervals: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "Present only with kernelTolerance: for each sample the closed "
            "interval [low, high] of residuals (samples - prediction) attained "
            "over all admissible kernels."
        ),
    )


class InfeasibleResponse(BaseModel):
    status: str = "infeasible"
    detail: str = (
        "no event train within the amplitude and event-count limits can "
        "reproduce the samples within the residual limit"
    )
    firstUnexplainablePosition: int = Field(
        ...,
        description=(
            "First sample index p for which no complete candidate (amplitudes "
            "within [0, maxAmplitude], at most maxEvents events) can keep every "
            "residual of samples[0..p] within +/-maxResidual; with "
            "kernelTolerance the residuals must stay within the limit for "
            "every admissible kernel."
        ),
    )

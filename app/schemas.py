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
        description=(
            "Optional per-tap non-negative integer calibration tolerance: tap "
            "j may take any integer in [kernel[j]-kernelTolerance[j], "
            "kernel[j]+kernelTolerance[j]]. Must have the same length as "
            "kernel, and every minimum tap value kernel[j]-tolerance[j] must "
            "stay positive. When omitted the nominal semantics are used."
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
    def _kernel_tolerance_valid(
        cls, value: Optional[List[int]], info: ValidationInfo
    ) -> Optional[List[int]]:
        if value is None:
            return None
        for tol in value:
            if tol < 0:
                raise ValueError("kernelTolerance entries must be non-negative integers")
            if tol > 1_000_000:
                raise ValueError("kernelTolerance entries must be <= 1000000")
        kernel = info.data.get("kernel")
        if kernel is not None:
            if len(value) != len(kernel):
                raise ValueError(
                    "kernelTolerance must have the same length as kernel"
                )
            for j, tol in enumerate(value):
                if kernel[j] - tol < 1:
                    raise ValueError(
                        "each minimum kernel value "
                        "kernel[j] - kernelTolerance[j] must stay positive"
                    )
        return value


class Event(BaseModel):
    position: int = Field(..., description="Sample index where the event starts.")
    amplitude: int = Field(..., description="Positive integer amplitude of the event.")


class Objectives(BaseModel):
    maxAbsResidual: int
    sumAbsResidual: int
    eventCount: int


class DeconvolveSuccessResponse(BaseModel):
    status: str = "ok"
    events: List[Event] = Field(..., description="Recovered events (position/amplitude).")
    amplitudes: List[int] = Field(..., description="Full amplitude sequence (one entry per sample).")
    prediction: List[int] = Field(..., description="Nominal predicted waveform, aligned with the samples.")
    residuals: List[int] = Field(..., description="Nominal pointwise residuals: samples - prediction.")
    objectives: Objectives
    # Present only when kernelTolerance was supplied: per-sample closed
    # intervals over all allowed kernels.
    robust: bool = Field(False, description="Whether calibration tolerance was applied.")
    predictionIntervals: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "Per-sample [min, max] closed prediction intervals attained over "
            "all allowed response kernels (robust mode only)."
        ),
    )
    residualIntervals: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "Per-sample [min, max] closed residual intervals (samples minus "
            "prediction) attained over all allowed response kernels (robust "
            "mode only)."
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
            "residual of samples[0..p] within +/-maxResidual; under "
            "kernelTolerance the residuals must hold for every allowed kernel."
        ),
    )
    robust: bool = Field(False, description="Whether calibration tolerance was applied.")

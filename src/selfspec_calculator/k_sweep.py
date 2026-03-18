from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .config import InputPaths
from .estimator import estimate_point
from .report import Metrics
from .stats import SpeculationStats


class KSweepCandidate(BaseModel):
    k: int = Field(..., ge=0)
    expected_accepted_tokens: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _validate_mean(self) -> "KSweepCandidate":
        if self.expected_accepted_tokens > self.k:
            raise ValueError(
                f"expected_accepted_tokens ({self.expected_accepted_tokens}) must be <= k ({self.k})"
            )
        return self


class KSweepInput(BaseModel):
    candidates: list[KSweepCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_k(self) -> "KSweepInput":
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        ks = [candidate.k for candidate in self.candidates]
        if len(set(ks)) != len(ks):
            raise ValueError("candidate k values must be unique")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "KSweepInput":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(path))

        suffix = p.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported k-sweep format: {p.suffix} (expected .json/.yaml/.yml)")

        if isinstance(raw, dict) and "candidates" not in raw:
            raw = {
                "candidates": [
                    {"k": int(k), "expected_accepted_tokens": float(v)}
                    for k, v in raw.items()
                ]
            }

        try:
            return cls.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid k-sweep input: {p}") from exc


class KSweepPoint(BaseModel):
    l_prompt: int = Field(..., ge=0)
    k: int = Field(..., ge=0)
    expected_accepted_tokens: float = Field(..., ge=0.0)
    expected_committed_tokens: float = Field(..., ge=0.0)
    synthetic_histogram: dict[int, float]
    speculative: Metrics
    baseline: Metrics
    throughput_speedup_vs_baseline: float | None = None
    tokens_per_joule_speedup_vs_baseline: float | None = None


class KSweepBest(BaseModel):
    l_prompt: int = Field(..., ge=0)
    best_throughput: KSweepPoint
    best_tokens_per_joule: KSweepPoint


class KSweepReport(BaseModel):
    generated_at: str
    prompt_lengths: list[int]
    points: list[KSweepPoint]
    best_by_prompt_length: list[KSweepBest]
    paths: InputPaths | None = None


def mean_acceptance_to_stats(*, k: int, expected_accepted_tokens: float) -> SpeculationStats:
    if expected_accepted_tokens < 0.0:
        raise ValueError("expected_accepted_tokens must be >= 0")
    if expected_accepted_tokens > k:
        raise ValueError(f"expected_accepted_tokens ({expected_accepted_tokens}) must be <= k ({k})")

    lower = int(expected_accepted_tokens)
    upper = lower if expected_accepted_tokens == float(lower) else lower + 1
    if upper > k:
        upper = k
        lower = k

    if lower == upper:
        histogram = {lower: 1.0}
    else:
        p_upper = expected_accepted_tokens - float(lower)
        histogram = {lower: 1.0 - p_upper, upper: p_upper}

    return SpeculationStats(k=k, histogram=histogram)


def evaluate_k_sweep(
    *,
    model,  # noqa: ANN001
    hardware,  # noqa: ANN001
    sweep_input: KSweepInput,
    prompt_lengths: list[int],
    paths: dict[str, str] | None = None,
) -> KSweepReport:
    from datetime import datetime, timezone

    points: list[KSweepPoint] = []
    for l_prompt in prompt_lengths:
        baseline, _ = estimate_point(model, hardware, SpeculationStats(k=0, histogram={0: 1.0}), l_prompt)
        for candidate in sorted(sweep_input.candidates, key=lambda item: item.k):
            stats = mean_acceptance_to_stats(
                k=candidate.k,
                expected_accepted_tokens=candidate.expected_accepted_tokens,
            )
            speculative, _ = estimate_point(model, hardware, stats, l_prompt)
            points.append(
                KSweepPoint(
                    l_prompt=l_prompt,
                    k=candidate.k,
                    expected_accepted_tokens=candidate.expected_accepted_tokens,
                    expected_committed_tokens=candidate.expected_accepted_tokens + 1.0,
                    synthetic_histogram=stats.histogram,
                    speculative=speculative,
                    baseline=baseline,
                    throughput_speedup_vs_baseline=(
                        speculative.throughput_tokens_per_s / baseline.throughput_tokens_per_s
                        if baseline.throughput_tokens_per_s > 0
                        else None
                    ),
                    tokens_per_joule_speedup_vs_baseline=(
                        speculative.tokens_per_joule / baseline.tokens_per_joule
                        if baseline.tokens_per_joule > 0
                        else None
                    ),
                )
            )

    best_by_prompt_length: list[KSweepBest] = []
    for l_prompt in sorted(set(prompt_lengths)):
        rows = [point for point in points if point.l_prompt == l_prompt]
        best_throughput = sorted(
            rows,
            key=lambda point: (
                -(point.speculative.throughput_tokens_per_s),
                point.k,
            ),
        )[0]
        best_tokens_per_joule = sorted(
            rows,
            key=lambda point: (
                -(point.speculative.tokens_per_joule),
                point.k,
            ),
        )[0]
        best_by_prompt_length.append(
            KSweepBest(
                l_prompt=l_prompt,
                best_throughput=best_throughput,
                best_tokens_per_joule=best_tokens_per_joule,
            )
        )

    return KSweepReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_lengths=prompt_lengths,
        points=points,
        best_by_prompt_length=best_by_prompt_length,
        paths=InputPaths(**paths) if paths is not None else None,
    )

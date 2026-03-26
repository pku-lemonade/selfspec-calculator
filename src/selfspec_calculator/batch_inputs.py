from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import HardwareConfig, ModelConfig
from .estimator import estimate_sweep
from .k_sweep import KSweepInput, KSweepReport, evaluate_k_sweep
from .stats import SpeculationStats


MODEL_CONFIG_MAP = {
    "Qwen3-0.6B": "examples/model_qwen3_0p6b.yaml",
    "Qwen3-1.7B": "examples/model_qwen3_1p7b.yaml",
    "Llama-3.2-1B": "examples/model_llama3_2_1b.yaml",
}


class SimulatorSweepResult(BaseModel):
    counts: list[int] | None = None
    total_bursts: int | None = None
    mean_accepted: float = Field(..., ge=0.0)
    acceptance_ratio: float | None = None
    expected_committed_tokens_per_burst: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class SimulatorSweepInput(BaseModel):
    run_id: str | None = None
    model: dict[str, Any]
    dataset: dict[str, Any]
    knobs: dict[str, Any]
    results_by_k: dict[str, SimulatorSweepResult]
    best_by_mean_accepted: dict[str, Any] | None = None
    best_by_acceptance_ratio: dict[str, Any] | None = None

    @property
    def prompt_length(self) -> int:
        value = self.dataset.get("prompt_length")
        if value is None:
            raise ValueError("dataset.prompt_length is required")
        return int(value)

    @property
    def draft_adc_bits(self) -> int:
        value = self.knobs.get("draft_adc_bits")
        if value is None:
            raise ValueError("knobs.draft_adc_bits is required")
        return int(value)

    @property
    def verify_adc_bits(self) -> int:
        value = self.knobs.get("verify_adc_bits")
        if value is None:
            raise ValueError("knobs.verify_adc_bits is required")
        return int(value)

    @property
    def model_label(self) -> str:
        checkpoint_path = str(self.model.get("checkpoint_path", ""))
        if "Qwen3-0.6B" in checkpoint_path or "qwen0p6b" in checkpoint_path.lower():
            return "Qwen3-0.6B"
        if "Qwen3-1.7B" in checkpoint_path or "qwen3_1p7b" in checkpoint_path.lower():
            return "Qwen3-1.7B"
        if "Llama-3.2-1B" in checkpoint_path or "llama3p2_1b" in checkpoint_path.lower():
            return "Llama-3.2-1B"
        raise ValueError(f"Cannot infer model from checkpoint path: {checkpoint_path}")

    @classmethod
    def from_path(cls, path: str | Path) -> "SimulatorSweepInput":
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        try:
            return cls.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid simulator sweep input: {p}") from exc

    def to_k_sweep_input(self) -> KSweepInput:
        candidates = []
        for k_str, result in sorted(self.results_by_k.items(), key=lambda item: int(item[0])):
            k = int(k_str)
            if result.mean_accepted > k:
                raise ValueError(
                    f"Input mean_accepted ({result.mean_accepted}) exceeds k ({k}) for results_by_k[{k}]"
                )
            candidates.append({"k": k, "expected_accepted_tokens": result.mean_accepted})
        return KSweepInput.model_validate({"candidates": candidates})


class BatchSummaryRow(BaseModel):
    input_file: str
    model: str
    adc_bits: str
    k: int
    acceptance_rate: float
    final_ppa: str
    output_file: str


class BatchSummary(BaseModel):
    rows: list[BatchSummaryRow]


def _hardware_with_adc_override(template: HardwareConfig, *, draft_bits: int, verify_bits: int) -> HardwareConfig:
    hardware = template.model_copy(deep=True)
    if hardware.analog is None:
        raise ValueError("Batch runner requires knob-based analog hardware template")
    hardware.analog.adc.draft_bits = draft_bits
    hardware.analog.adc.residual_bits = verify_bits
    # Validate the override against library tables
    hardware.resolve_knob_specs()
    return hardware


def _model_from_label(repo_root: Path, model_label: str) -> tuple[ModelConfig, str]:
    rel = MODEL_CONFIG_MAP.get(model_label)
    if rel is None:
        raise ValueError(f"No model config mapping for {model_label}")
    path = repo_root / rel
    return ModelConfig.from_yaml(path), rel


def _final_ppa_string(*, metrics, on_chip_area: float) -> str:  # noqa: ANN001
    return (
        f"lat={metrics.latency_ns_per_token/1000:.2f}us/tok; "
        f"thr={metrics.throughput_tokens_per_s:.2f} tok/s; "
        f"tok/J={metrics.tokens_per_joule:.2f}; "
        f"area={on_chip_area:.2f}mm^2"
    )


def _summary_markdown(summary: BatchSummary) -> str:
    lines = [
        "| model | adc bits | K value | acceptance rate | final PPA |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary.rows:
        lines.append(
            f"| {row.model} | {row.adc_bits} | {row.k} | {row.acceptance_rate:.4f} | {row.final_ppa} |"
        )
    return "\n".join(lines) + "\n"


def run_batch_inputs(
    *,
    inputs_dir: Path,
    output_dir: Path,
    hardware_template_path: Path,
    repo_root: Path,
) -> BatchSummary:
    template = HardwareConfig.from_yaml(hardware_template_path)
    rows: list[BatchSummaryRow] = []
    best_rows: list[BatchSummaryRow] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(inputs_dir.glob("*.json")):
        sim_input = SimulatorSweepInput.from_path(input_path)
        model, model_rel_path = _model_from_label(repo_root, sim_input.model_label)
        hardware = _hardware_with_adc_override(
            template,
            draft_bits=sim_input.draft_adc_bits,
            verify_bits=sim_input.verify_adc_bits,
        )
        sweep_input = sim_input.to_k_sweep_input()
        report = evaluate_k_sweep(
            model=model,
            hardware=hardware,
            sweep_input=sweep_input,
            prompt_lengths=[sim_input.prompt_length],
            paths={
                "model": str(repo_root / model_rel_path),
                "hardware": str(hardware_template_path),
                "stats": str(input_path),
            },
        )

        output_path = output_dir / f"{input_path.stem}.json"
        output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        area_report = estimate_sweep(
            model=model,
            hardware=hardware,
            stats=SpeculationStats(k=0, histogram={0: 1.0}),
            prompt_lengths=[sim_input.prompt_length],
        )
        on_chip_area = area_report.area_breakdown_mm2.on_chip_mm2

        point_rows: list[BatchSummaryRow] = []
        for point in sorted(report.points, key=lambda item: item.k):
            point_rows.append(
                BatchSummaryRow(
                    input_file=input_path.name,
                    model=sim_input.model_label,
                    adc_bits=f"{sim_input.draft_adc_bits}/{sim_input.verify_adc_bits}",
                    k=point.k,
                    acceptance_rate=(point.expected_accepted_tokens / point.k) if point.k > 0 else 0.0,
                    final_ppa=_final_ppa_string(
                        metrics=point.speculative,
                        on_chip_area=on_chip_area,
                    ),
                    output_file=output_path.name,
                )
            )
        rows.extend(point_rows)

        best = report.best_by_prompt_length[0].best_throughput
        best_rows.append(
            BatchSummaryRow(
                input_file=input_path.name,
                model=sim_input.model_label,
                adc_bits=f"{sim_input.draft_adc_bits}/{sim_input.verify_adc_bits}",
                k=best.k,
                acceptance_rate=(best.expected_accepted_tokens / best.k) if best.k > 0 else 0.0,
                final_ppa=_final_ppa_string(
                    metrics=best.speculative,
                    on_chip_area=on_chip_area,
                ),
                output_file=output_path.name,
            )
        )

    summary = BatchSummary(rows=rows)
    best_summary = BatchSummary(rows=best_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_dir / "summary_best.json").write_text(
        json.dumps(best_summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary_best.md").write_text(_summary_markdown(best_summary), encoding="utf-8")
    return summary

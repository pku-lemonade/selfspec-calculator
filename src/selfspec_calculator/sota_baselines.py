from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import FfnType, ModelConfig


DEFAULT_MODEL_PATHS = (
    Path("examples/model_qwen3_0p6b.yaml"),
    Path("examples/model_llama3_2_1b.yaml"),
    Path("examples/model_qwen3_1p7b.yaml"),
)


class WorkloadOps(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    model_path: str | None = None
    prompt_len: int = Field(..., ge=0)
    generated_tokens: int = Field(..., ge=1)
    linear_macs: float = Field(..., ge=0.0)
    attention_qk_macs: float = Field(..., ge=0.0)
    attention_pv_macs: float = Field(..., ge=0.0)
    attention_softmax_ops: float = Field(..., ge=0.0)
    attention_ops: float = Field(..., ge=0.0)
    transformer_block_ops: float = Field(..., ge=0.0)


class BaselineRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    baseline: str
    scope: str
    status: str
    prompt_len: int = Field(..., ge=0)
    generated_tokens: int = Field(..., ge=1)
    operations: float = Field(..., ge=0.0)
    latency_s: float | None = Field(default=None, ge=0.0)
    energy_j: float | None = Field(default=None, ge=0.0)
    throughput_tokens_per_s: float | None = Field(default=None, ge=0.0)
    tokens_per_joule: float | None = Field(default=None, ge=0.0)
    area_mm2: float | None = Field(default=None, ge=0.0)
    effective_power_w: float | None = Field(default=None, ge=0.0)
    basis: str
    caveats: list[str] = Field(default_factory=list)


class SotaBaselineReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    generated_at: str
    prompt_len: int = Field(..., ge=0)
    generated_tokens: int = Field(..., ge=1)
    model_paths: list[str]
    workloads: list[WorkloadOps]
    rows: list[BaselineRow]
    notes: list[str] = Field(default_factory=list)


def _model_name(model: ModelConfig, fallback: Path | None = None) -> str:
    if model.name:
        return model.name
    if fallback is not None:
        return fallback.stem
    return "unnamed-model"


def estimate_workload_ops(
    model: ModelConfig,
    *,
    prompt_len: int,
    generated_tokens: int,
    model_path: str | None = None,
) -> WorkloadOps:
    d_model = model.d_model
    d_head = model.d_head
    d_ff = model.effective_d_ff

    qkv_macs = 3 * d_model * d_model
    wo_macs = d_model * d_model
    if model.ffn_type == FfnType.mlp:
        ffn_macs = 2 * d_model * d_ff
    else:
        ffn_macs = 3 * d_model * d_ff

    linear_per_token_per_layer = qkv_macs + wo_macs + ffn_macs
    linear_macs = float(model.n_layers * generated_tokens * linear_per_token_per_layer)

    context_sum = sum(prompt_len + i for i in range(generated_tokens))
    attention_qk_macs = float(model.n_layers * model.n_heads * d_head * context_sum)
    attention_pv_macs = float(model.n_layers * model.n_heads * d_head * context_sum)
    attention_softmax_ops = float(model.n_layers * model.n_heads * context_sum)
    attention_ops = attention_qk_macs + attention_pv_macs + attention_softmax_ops

    return WorkloadOps(
        model=_model_name(model),
        model_path=model_path,
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        linear_macs=linear_macs,
        attention_qk_macs=attention_qk_macs,
        attention_pv_macs=attention_pv_macs,
        attention_softmax_ops=attention_softmax_ops,
        attention_ops=attention_ops,
        transformer_block_ops=linear_macs + attention_ops,
    )


def project_sota_baselines(workload: WorkloadOps) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    t = float(workload.generated_tokens)

    # HARDSEA TVLSI 2024 Table V reports the self-attention comparison with
    # QK, softmax, and PV mapped to the accelerator.
    hardsea_tput_ops_per_s = 802.1e9
    hardsea_eff_ops_per_j = 821.3e9
    hardsea_latency = workload.attention_ops / hardsea_tput_ops_per_s
    hardsea_energy = workload.attention_ops / hardsea_eff_ops_per_j
    rows.append(
        BaselineRow(
            model=workload.model,
            baseline="HARDSEA",
            scope="attention_qk_softmax_pv",
            status="scoped_projection",
            prompt_len=workload.prompt_len,
            generated_tokens=workload.generated_tokens,
            operations=workload.attention_ops,
            latency_s=hardsea_latency,
            energy_j=hardsea_energy,
            throughput_tokens_per_s=t / hardsea_latency if hardsea_latency > 0 else None,
            tokens_per_joule=t / hardsea_energy if hardsea_energy > 0 else None,
            area_mm2=4.95,
            effective_power_w=hardsea_tput_ops_per_s / hardsea_eff_ops_per_j,
            basis=(
                "HARDSEA TVLSI 2024 Table V average throughput 802.1 GOPS and "
                "average energy efficiency 821.3 GOPS/W at 300 MHz, 0.9 V."
            ),
            caveats=[
                "Attention-only row; linear layers, layernorm, residual paths, and LM head are outside this scope.",
                "Uses the calculator workload dimensions rather than HARDSEA's BERT/GPT-2 benchmark shapes.",
            ],
        )
    )

    # PIM-GPT places a MAC unit at every GDDR6 bank. The paper states 8
    # channels, 16 banks/channel, 16 multipliers per bank MAC unit, and 1 GHz.
    # This peak model is intentionally labeled as a lower bound until the DRAM
    # command/state machine and ASIC-side operations are reproduced.
    pimgpt_peak_ops_per_s = 8 * 16 * 16 * 1e9
    pimgpt_power_w = 8 * 149.29e-3 + 304.59e-3
    pimgpt_latency = workload.transformer_block_ops / pimgpt_peak_ops_per_s
    pimgpt_energy = pimgpt_latency * pimgpt_power_w
    rows.append(
        BaselineRow(
            model=workload.model,
            baseline="PIM-GPT",
            scope="transformer_block_peak_no_lm_head",
            status="lower_bound_projection",
            prompt_len=workload.prompt_len,
            generated_tokens=workload.generated_tokens,
            operations=workload.transformer_block_ops,
            latency_s=pimgpt_latency,
            energy_j=pimgpt_energy,
            throughput_tokens_per_s=t / pimgpt_latency if pimgpt_latency > 0 else None,
            tokens_per_joule=t / pimgpt_energy if pimgpt_energy > 0 else None,
            area_mm2=0.64,
            effective_power_w=pimgpt_power_w,
            basis=(
                "PIM-GPT hardware configuration: 8 channels, 16 banks/channel, "
                "one MAC unit per bank, 16 BF16 multipliers per MAC unit at 1 GHz, "
                "plus 8*149.29 mW PIM MAC power and 304.59 mW ASIC power."
            ),
            caveats=[
                "Peak compute lower bound; excludes DRAM row timing, refresh, row-buffer effects, I/O, and scheduling stalls.",
                "Includes calculator transformer-block work only; LM head is excluded unless modeled separately.",
                "Area is the published ASIC core area only; DRAM/PIM die area is not included.",
            ],
        )
    )

    # HyFlexPIM Table 2 gives the analog RRAM module organization and total
    # analog power across 24 processing units. Use this only for the analog
    # linear-layer lower bound because the full method also needs SVD/rank
    # mapping, digital RRAM attention, SFU, and movement modeling.
    hyflex_peak_ops_per_s = 24 * 512 * 64 * 128 / 100e-9
    hyflex_power_w = 22_336.59e-3
    hyflex_latency = workload.linear_macs / hyflex_peak_ops_per_s
    hyflex_energy = hyflex_latency * hyflex_power_w
    rows.append(
        BaselineRow(
            model=workload.model,
            baseline="HyFlexPIM",
            scope="linear_layers_analog_peak",
            status="lower_bound_projection",
            prompt_len=workload.prompt_len,
            generated_tokens=workload.generated_tokens,
            operations=workload.linear_macs,
            latency_s=hyflex_latency,
            energy_j=hyflex_energy,
            throughput_tokens_per_s=t / hyflex_latency if hyflex_latency > 0 else None,
            tokens_per_joule=t / hyflex_energy if hyflex_energy > 0 else None,
            area_mm2=11.24,
            effective_power_w=hyflex_power_w,
            basis=(
                "HyFlexPIM ISCA 2025 Table 2 analog RRAM module: 24 PUs, "
                "512 arrays per PU, 64x128 array size, 100 ns conversion cycle, "
                "and 22,336.59 mW total analog-module power."
            ),
            caveats=[
                "Analog linear-layer lower bound; excludes digital RRAM attention, SFU, SVD/rank mapping, and movement.",
                "Does not reproduce HyFlexPIM's accuracy recovery, SLC/MLC assignment, or process scaling flow.",
                "Area is the published analog RRAM module total only; digital RRAM module area is not included.",
            ],
        )
    )

    return rows


def build_report(
    *,
    model_paths: list[Path],
    prompt_len: int,
    generated_tokens: int,
) -> SotaBaselineReport:
    workloads: list[WorkloadOps] = []
    rows: list[BaselineRow] = []

    for model_path in model_paths:
        model = ModelConfig.from_yaml(model_path)
        workload = estimate_workload_ops(
            model,
            prompt_len=prompt_len,
            generated_tokens=generated_tokens,
            model_path=str(model_path),
        )
        workloads.append(workload)
        rows.extend(project_sota_baselines(workload))

    return SotaBaselineReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        model_paths=[str(path) for path in model_paths],
        workloads=workloads,
        rows=rows,
        notes=[
            "Operation counts are derived from selfspec-calculator ModelConfig YAMLs so they match the calculator workload abstraction.",
            "Rows marked scoped_projection have a deliberately limited scope that matches a published baseline metric.",
            "Rows marked lower_bound_projection are peak analytical runs and need event-model validation before final paper claims.",
        ],
    )


def _float_for_csv(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_report_outputs(report: SotaBaselineReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    (output_dir / "sota_baseline_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_csv(
        output_dir / "workload_ops.csv",
        [workload.model_dump(mode="json") for workload in report.workloads],
    )
    _write_csv(
        output_dir / "sota_baseline_rows.csv",
        [row.model_dump(mode="json") for row in report.rows],
    )
    (output_dir / "summary.md").write_text(report_to_markdown(report), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _float_for_csv(value) for key, value in row.items()})


def _fmt(value: float | None, *, unit_scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return ""
    scaled = value * unit_scale
    if scaled == 0:
        return "0"
    if abs(scaled) >= 100:
        return f"{scaled:.2f}"
    if abs(scaled) >= 1:
        return f"{scaled:.{digits}f}"
    return f"{scaled:.3e}"


def report_to_markdown(report: SotaBaselineReport) -> str:
    lines: list[str] = []
    lines.append("# Matched SOTA Baseline Runs")
    lines.append("")
    lines.append(f"Prompt length: `{report.prompt_len}`")
    lines.append(f"Generated tokens: `{report.generated_tokens}`")
    lines.append("")
    lines.append("## Workload Operation Counts")
    lines.append("")
    lines.append("| Model | Linear MACs | Attention QK MACs | Attention PV MACs | Softmax ops | Block ops |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for workload in report.workloads:
        lines.append(
            f"| {workload.model} | {workload.linear_macs:.4e} | "
            f"{workload.attention_qk_macs:.4e} | {workload.attention_pv_macs:.4e} | "
            f"{workload.attention_softmax_ops:.4e} | {workload.transformer_block_ops:.4e} |"
        )

    lines.append("")
    lines.append("## Baseline Rows")
    lines.append("")
    lines.append("| Model | Baseline | Scope | Status | Latency (ms) | tok/s | Energy (mJ) | tok/J | Area (mm2) |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in report.rows:
        lines.append(
            f"| {row.model} | {row.baseline} | {row.scope} | {row.status} | "
            f"{_fmt(row.latency_s, unit_scale=1e3)} | {_fmt(row.throughput_tokens_per_s)} | "
            f"{_fmt(row.energy_j, unit_scale=1e3)} | {_fmt(row.tokens_per_joule)} | "
            f"{_fmt(row.area_mm2)} |"
        )

    lines.append("")
    lines.append("## Basis And Caveats")
    lines.append("")
    seen: set[tuple[str, str]] = set()
    for row in report.rows:
        key = (row.baseline, row.scope)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{row.baseline}` / `{row.scope}`: {row.basis}")
        for caveat in row.caveats:
            lines.append(f"  - {caveat}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)

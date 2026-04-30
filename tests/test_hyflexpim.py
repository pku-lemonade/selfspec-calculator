import json
from pathlib import Path

import pytest

from selfspec_calculator.config import ModelConfig
from selfspec_calculator.hyflexpim import (
    HyFlexPIMConfig,
    build_report,
    estimate_hyflexpim,
    estimate_linear_layer_profile,
    write_report_outputs,
)
from selfspec_calculator.hyflexpim_cli import main


def _toy_model() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "name": "toy",
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 4,
            "activation_bits": 8,
            "ffn_type": "mlp",
            "d_ff": 64,
        }
    )


def test_mlc_mapping_reduces_bit_serial_linear_latency() -> None:
    model = _toy_model()
    base = {
        "analog_rows": 16,
        "analog_columns": 16,
        "analog_arrays_per_module": 1,
        "analog_modules_per_pu": 1,
    }

    slc_profile = estimate_linear_layer_profile(
        model,
        config=HyFlexPIMConfig(slc_rate=1.0, **base),
        tensor_pus_per_layer=1,
    )
    mlc_profile = estimate_linear_layer_profile(
        model,
        config=HyFlexPIMConfig(slc_rate=0.0, **base),
        tensor_pus_per_layer=1,
    )

    assert mlc_profile.latency_s < slc_profile.latency_s
    assert mlc_profile.svd_macs == pytest.approx(slc_profile.svd_macs)


def test_hyflexpim_preserves_hard_threshold_macs_for_square_toy_model() -> None:
    model = _toy_model()
    workload, row = estimate_hyflexpim(
        model,
        prompt_len=4,
        generated_tokens=2,
        config=HyFlexPIMConfig(slc_rate=0.20),
    )

    assert row.hyflex_svd_linear_macs == pytest.approx(workload.linear_macs)
    assert row.capacity_pus_per_layer >= 1
    assert row.tensor_pus_per_layer >= row.capacity_pus_per_layer
    assert row.total_latency_s > 0
    assert row.total_energy_j > 0
    assert row.throughput_tokens_per_s == pytest.approx(row.generated_tokens / row.total_latency_s)
    assert row.reported_throughput_mode == "single_stream"
    assert row.single_stream_total_latency_s == pytest.approx(row.total_latency_s)
    assert row.paper_pipeline_throughput_tokens_per_s > row.single_stream_throughput_tokens_per_s
    assert row.tokens_per_joule == pytest.approx(row.generated_tokens / row.total_energy_j)


def test_paper_pipeline_mode_reports_stage_period() -> None:
    model = _toy_model()
    _workload, row = estimate_hyflexpim(
        model,
        prompt_len=4,
        generated_tokens=2,
        config=HyFlexPIMConfig(slc_rate=0.20, throughput_mode="paper_pipeline"),
    )

    assert row.reported_throughput_mode == "paper_pipeline"
    assert row.total_latency_s == pytest.approx(row.paper_pipeline_total_latency_s)
    assert row.throughput_tokens_per_s == pytest.approx(row.paper_pipeline_throughput_tokens_per_s)
    assert row.paper_pipeline_fill_latency_s > row.paper_pipeline_latency_s_per_token


def test_hyflexpim_cli_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "hyflex"

    rc = main(
        [
            "--models",
            str(repo_root / "examples" / "model_qwen3_0p6b.yaml"),
            "--prompt-length",
            "64",
            "--generated-tokens",
            "32",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    payload = json.loads((output_dir / "hyflexpim_report.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["model"] == "qwen3-0.6b"
    assert (output_dir / "hyflexpim_rows.csv").exists()
    assert (output_dir / "hyflexpim_layer_components.csv").exists()
    assert (output_dir / "summary.md").exists()


def test_write_report_outputs_accepts_multiple_models(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = build_report(
        model_paths=[
            repo_root / "examples" / "model_qwen3_0p6b.yaml",
            repo_root / "examples" / "model_llama3_2_1b.yaml",
        ],
        prompt_len=64,
        generated_tokens=32,
        config=HyFlexPIMConfig(slc_rate=0.20),
    )

    write_report_outputs(report, tmp_path)

    rows_csv = (tmp_path / "hyflexpim_rows.csv").read_text(encoding="utf-8")
    assert "qwen3-0.6b" in rows_csv
    assert "llama-3.2-1b" in rows_csv

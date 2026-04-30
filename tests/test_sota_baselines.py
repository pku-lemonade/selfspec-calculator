import json
from pathlib import Path

import pytest

from selfspec_calculator.config import ModelConfig
from selfspec_calculator.sota_baselines import (
    build_report,
    estimate_workload_ops,
    project_sota_baselines,
    write_report_outputs,
)
from selfspec_calculator.sota_baselines_cli import main


def test_workload_ops_match_calculator_model_shape() -> None:
    model = ModelConfig.model_validate(
        {
            "name": "toy",
            "n_layers": 2,
            "d_model": 4,
            "n_heads": 2,
            "activation_bits": 8,
            "ffn_type": "swiglu",
            "d_ff": 8,
        }
    )

    ops = estimate_workload_ops(model, prompt_len=2, generated_tokens=3)

    assert ops.linear_macs == pytest.approx(960.0)
    assert ops.attention_qk_macs == pytest.approx(72.0)
    assert ops.attention_pv_macs == pytest.approx(72.0)
    assert ops.attention_softmax_ops == pytest.approx(36.0)
    assert ops.attention_ops == pytest.approx(180.0)
    assert ops.transformer_block_ops == pytest.approx(1140.0)


def test_project_sota_baselines_has_required_rows() -> None:
    model = ModelConfig.model_validate(
        {
            "name": "toy",
            "n_layers": 2,
            "d_model": 4,
            "n_heads": 2,
            "activation_bits": 8,
            "ffn_type": "mlp",
            "d_ff": 8,
        }
    )
    workload = estimate_workload_ops(model, prompt_len=4, generated_tokens=2)

    rows = project_sota_baselines(workload)

    assert [row.baseline for row in rows] == ["HARDSEA", "PIM-GPT", "HyFlexPIM"]
    assert rows[0].status == "scoped_projection"
    assert rows[1].status == "lower_bound_projection"
    assert rows[2].status == "lower_bound_projection"
    assert rows[0].latency_s == pytest.approx(workload.attention_ops / 802.1e9)
    assert rows[0].energy_j == pytest.approx(workload.attention_ops / 821.3e9)


def test_sota_baselines_cli_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "sota"

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
    payload = json.loads((output_dir / "sota_baseline_report.json").read_text(encoding="utf-8"))
    assert len(payload["workloads"]) == 1
    assert len(payload["rows"]) == 3
    assert (output_dir / "workload_ops.csv").exists()
    assert (output_dir / "sota_baseline_rows.csv").exists()
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
    )

    write_report_outputs(report, tmp_path)

    rows_csv = (tmp_path / "sota_baseline_rows.csv").read_text(encoding="utf-8")
    assert "HARDSEA" in rows_csv
    assert "PIM-GPT" in rows_csv
    assert "HyFlexPIM" in rows_csv

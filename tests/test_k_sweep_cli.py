import json
from pathlib import Path

import pytest

from selfspec_calculator.k_sweep import KSweepInput, evaluate_k_sweep, mean_acceptance_to_stats
from selfspec_calculator.k_sweep_cli import main
from selfspec_calculator.config import HardwareConfig, ModelConfig


def test_mean_acceptance_to_stats_uses_two_bin_histogram() -> None:
    stats = mean_acceptance_to_stats(k=5, expected_accepted_tokens=4.85)
    assert stats.k == 5
    assert stats.histogram == {4: pytest.approx(0.15), 5: pytest.approx(0.85)}


def test_k_sweep_input_rejects_mean_above_k(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text("candidates:\n  - k: 5\n    expected_accepted_tokens: 5.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid k-sweep input"):
        KSweepInput.from_path(path)


def test_evaluate_k_sweep_selects_best_k_for_throughput() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model = ModelConfig.from_yaml(repo_root / "examples" / "model_qwen3_0p6b.yaml")
    hardware = HardwareConfig.from_yaml(repo_root / "examples" / "hardware_soc_area.yaml")
    sweep_input = KSweepInput.model_validate(
        {
            "candidates": [
                {"k": 1, "expected_accepted_tokens": 0.8},
                {"k": 5, "expected_accepted_tokens": 4.85},
            ]
        }
    )

    report = evaluate_k_sweep(
        model=model,
        hardware=hardware,
        sweep_input=sweep_input,
        prompt_lengths=[128],
    )
    assert len(report.points) == 2
    expected_best_throughput = max(report.points, key=lambda point: point.speculative.throughput_tokens_per_s)
    expected_best_tokens = max(report.points, key=lambda point: point.speculative.tokens_per_joule)
    assert report.best_by_prompt_length[0].best_throughput.k == expected_best_throughput.k
    assert report.best_by_prompt_length[0].best_tokens_per_joule.k == expected_best_tokens.k


def test_k_sweep_cli_runs_and_reports_best_k(capsys, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sweep_path = tmp_path / "acceptance_sweep.yaml"
    sweep_path.write_text(
        "\n".join(
            [
                "candidates:",
                "  - k: 1",
                "    expected_accepted_tokens: 0.8",
                "  - k: 5",
                "    expected_accepted_tokens: 4.85",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--model",
            str(repo_root / "examples" / "model_qwen3_0p6b.yaml"),
            "--hardware",
            str(repo_root / "examples" / "hardware_soc_area.yaml"),
            "--acceptance-sweep",
            str(sweep_path),
            "--prompt-lengths",
            "128",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["points"]) == 2
    best_throughput = max(payload["points"], key=lambda point: point["speculative"]["throughput_tokens_per_s"])
    best_tokens = max(payload["points"], key=lambda point: point["speculative"]["tokens_per_joule"])
    assert payload["best_by_prompt_length"][0]["best_throughput"]["k"] == best_throughput["k"]
    assert payload["best_by_prompt_length"][0]["best_tokens_per_joule"]["k"] == best_tokens["k"]

import json
from pathlib import Path

from selfspec_calculator.batch_inputs import SimulatorSweepInput, run_batch_inputs
from selfspec_calculator.batch_inputs_cli import main


def _write_input(path: Path) -> None:
    payload = {
        "run_id": None,
        "model": {
            "checkpoint_path": "checkpoints/Qwen/Qwen3-0.6B/model_demo.pth",
            "draft_checkpoint_path": "checkpoints/Qwen/Qwen3-0.6B/model_demo.pth",
            "tokenizer_path": "checkpoints/Qwen/Qwen3-0.6B/tokenizer.json",
        },
        "dataset": {
            "prompts_path": "out/demo_prompts.txt",
            "prompt_field": "prompt",
            "limit": 10,
            "shuffle": False,
            "loaded_prompts": 10,
            "prompt_length": 128,
        },
        "knobs": {
            "draft_adc_bits": 8,
            "verify_adc_bits": 12,
        },
        "results_by_k": {
            "2": {
                "counts": [1, 2, 3],
                "total_bursts": 6,
                "mean_accepted": 1.6,
                "acceptance_ratio": 0.8,
                "expected_committed_tokens_per_burst": 2.6,
                "meta": {},
            },
            "5": {
                "counts": [0, 0, 0, 0, 1, 9],
                "total_bursts": 10,
                "mean_accepted": 4.9,
                "acceptance_ratio": 0.98,
                "expected_committed_tokens_per_burst": 5.9,
                "meta": {},
            },
        },
        "best_by_mean_accepted": {"k": 5, "mean_accepted": 4.9},
        "best_by_acceptance_ratio": {"k": 5, "acceptance_ratio": 0.98},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_simulator_sweep_input_parses_properties(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    _write_input(path)
    parsed = SimulatorSweepInput.from_path(path)
    assert parsed.model_label == "Qwen3-0.6B"
    assert parsed.prompt_length == 128
    assert parsed.draft_adc_bits == 8
    assert parsed.verify_adc_bits == 12
    assert parsed.draft_delta_readout is False
    assert parsed.verify_delta_readout is False
    assert parsed.to_k_sweep_input().candidates[1].expected_accepted_tokens == 4.9


def test_run_batch_inputs_writes_per_run_outputs_and_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    _write_input(inputs_dir / "k_sweep_qwen0p6b_best_adc8_12_demo.json")

    summary = run_batch_inputs(
        inputs_dir=inputs_dir,
        output_dir=outputs_dir,
        hardware_template_path=repo_root / "examples" / "hardware_soc_area.yaml",
        repo_root=repo_root,
    )

    assert len(summary.rows) == 2
    row = summary.rows[0]
    assert row.model == "Qwen3-0.6B"
    assert row.adc_bits == "8/12"
    assert (outputs_dir / row.output_file).exists()
    assert (outputs_dir / "summary.json").exists()
    assert (outputs_dir / "summary.md").exists()
    assert (outputs_dir / "summary_best.json").exists()
    assert (outputs_dir / "summary_best.md").exists()

    payload = json.loads((outputs_dir / row.output_file).read_text(encoding="utf-8"))
    assert {point["k"] for point in payload["points"]} == {2, 5}
    md = (outputs_dir / "summary.md").read_text(encoding="utf-8")
    assert "| model | adc bits | K value | acceptance rate | final PPA |" in md
    assert md.count("| Qwen3-0.6B | 8/12 |") == 2
    best_md = (outputs_dir / "summary_best.md").read_text(encoding="utf-8")
    assert best_md.count("| Qwen3-0.6B | 8/12 |") == 1



def test_simulator_sweep_input_parses_delta_knobs(tmp_path: Path) -> None:
    path = tmp_path / "delta_input.json"
    payload = {
        "run_id": None,
        "model": {
            "checkpoint_path": "checkpoints/Qwen/Qwen3-1.7B/model_demo.pth",
            "draft_checkpoint_path": "checkpoints/Qwen/Qwen3-1.7B/model_demo.pth",
            "tokenizer_path": "checkpoints/Qwen/Qwen3-1.7B/tokenizer.json",
        },
        "dataset": {
            "prompts_path": "out/demo_prompts.txt",
            "prompt_field": "prompt",
            "limit": 10,
            "shuffle": False,
            "loaded_prompts": 10,
            "prompt_length": 64,
        },
        "knobs": {
            "draft_adc_bits": 6,
            "verify_adc_bits": 8,
            "draft_delta_readout": True,
            "verify_delta_readout": True,
            "draft_delta_dac_bits": 8,
            "verify_delta_dac_bits": 8,
        },
        "results_by_k": {
            "2": {
                "mean_accepted": 1.8,
                "acceptance_ratio": 0.9,
                "expected_committed_tokens_per_burst": 2.8,
                "meta": {},
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    parsed = SimulatorSweepInput.from_path(path)
    assert parsed.model_label == "Qwen3-1.7B"
    assert parsed.draft_delta_readout is True
    assert parsed.verify_delta_readout is True
    assert parsed.draft_delta_dac_bits == 8
    assert parsed.verify_delta_dac_bits == 8

def test_batch_inputs_cli_runs(capsys, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    _write_input(inputs_dir / "k_sweep_qwen0p6b_best_adc8_12_demo.json")

    rc = main(
        [
            "--inputs-dir",
            str(inputs_dir),
            "--output-dir",
            str(outputs_dir),
            "--hardware-template",
            str(repo_root / "examples" / "hardware_soc_area.yaml"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "processed 1 input files" in out
    assert (outputs_dir / "summary.md").exists()
    assert (outputs_dir / "summary_best.md").exists()

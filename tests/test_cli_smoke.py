import json
from pathlib import Path

from selfspec_calculator.cli import main


def test_cli_runs_on_knob_examples(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rc = main(
        [
            "--model",
            str(repo_root / "examples" / "model.yaml"),
            "--hardware",
            str(repo_root / "examples" / "hardware.yaml"),
            "--stats",
            str(repo_root / "examples" / "stats.json"),
            "--prompt-lengths",
            "64",
            "128",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    expected_k = json.loads((repo_root / "examples" / "stats.json").read_text(encoding="utf-8"))["k"]
    assert payload["k"] == expected_k
    assert payload["reuse_policy"] in {"reuse", "reread"}
    assert payload["hardware_mode"] == "knob-based"
    assert payload["resolved_library"] is not None
    assert isinstance(payload["points"], list)
    assert len(payload["points"]) == 2
    assert "area" in payload


def test_cli_runs_on_legacy_example(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rc = main(
        [
            "--model",
            str(repo_root / "examples" / "model.yaml"),
            "--hardware",
            str(repo_root / "examples" / "hardware_legacy.yaml"),
            "--stats",
            str(repo_root / "examples" / "stats.json"),
            "--prompt-lengths",
            "64",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["hardware_mode"] == "legacy"
    assert payload["resolved_library"] is None

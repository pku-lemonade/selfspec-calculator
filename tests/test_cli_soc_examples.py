import json
from pathlib import Path

from selfspec_calculator.cli import main


def test_cli_runs_on_soc_memory_example(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rc = main(
        [
            "--model",
            str(repo_root / "examples" / "model.yaml"),
            "--hardware",
            str(repo_root / "examples" / "hardware_soc_memory.yaml"),
            "--stats",
            str(repo_root / "examples" / "stats.json"),
            "--prompt-lengths",
            "64",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    point = payload["points"][0]
    total = point["breakdown"]["total"]
    assert total["memory_traffic"] is not None
    assert total["memory_traffic"]["hbm_read_bytes"] > 0.0
    assert total["memory_traffic"]["sram_write_bytes"] > 0.0

    stages = total["stages"]
    assert stages["kv_cache_energy_pj"] > 0.0

    components = total["components"]
    assert components["hbm_energy_pj"] > 0.0
    assert components["sram_energy_pj"] > 0.0
    assert components["fabric_energy_pj"] > 0.0


def test_cli_runs_on_analog_periphery_example(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rc = main(
        [
            "--model",
            str(repo_root / "examples" / "model.yaml"),
            "--hardware",
            str(repo_root / "examples" / "hardware_analog_periphery.yaml"),
            "--stats",
            str(repo_root / "examples" / "stats.json"),
            "--prompt-lengths",
            "64",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    point = payload["points"][0]
    total = point["breakdown"]["total"]
    assert total["memory_traffic"] is None

    stages = total["stages"]
    assert stages["buffers_add_energy_pj"] > 0.0
    assert stages["control_energy_pj"] > 0.0

    components = total["components"]
    assert components["tia_energy_pj"] > 0.0
    assert components["snh_energy_pj"] > 0.0


def test_cli_runs_on_custom_library_example(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rc = main(
        [
            "--model",
            str(repo_root / "examples" / "model.yaml"),
            "--hardware",
            str(repo_root / "examples" / "hardware_custom_library.yaml"),
            "--stats",
            str(repo_root / "examples" / "stats.json"),
            "--prompt-lengths",
            "64",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    resolved = payload["resolved_library"]
    assert resolved["name"] == "puma_like_v1"
    assert resolved["adc_draft"]["energy_pj_per_conversion"] == 0.123
    assert resolved["adc_residual"]["energy_pj_per_conversion"] == 0.456

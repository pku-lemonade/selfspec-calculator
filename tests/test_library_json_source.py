from pathlib import Path

import pytest

from selfspec_calculator.config import HardwareConfig


def _write_hardware(path: Path, *, library_file: str | None, library_name: str = "puma_like_v1") -> None:
    library_file_block = f"library_file: {library_file}\n" if library_file is not None else ""
    path.write_text(
        (
            "reuse_policy: reuse\n"
            f"library: {library_name}\n"
            f"{library_file_block}"
            "analog:\n"
            "  xbar_size: 128\n"
            "  num_columns_per_adc: 16\n"
            "  dac_bits: 4\n"
            "  adc:\n"
            "    draft_bits: 4\n"
            "    residual_bits: 12\n"
        ),
        encoding="utf-8",
    )


def _write_custom_library(path: Path, *, include_adc12: bool = True, include_required_sections: bool = True) -> None:
    adc = {
        "4": {
            "energy_pj_per_conversion": 0.123,
            "latency_ns_per_conversion": 0.040,
            "area_mm2_per_unit": 0.0012,
        }
    }
    if include_adc12:
        adc["12"] = {
            "energy_pj_per_conversion": 0.456,
            "latency_ns_per_conversion": 0.120,
            "area_mm2_per_unit": 0.0019,
        }

    payload = {
        "puma_like_v1": {
            "adc": adc,
            "dac": {
                "4": {
                    "energy_pj_per_conversion": 0.004,
                    "latency_ns_per_conversion": 0.010,
                    "area_mm2_per_unit": 1.67e-7,
                }
            },
            "array": {
                "energy_pj_per_activation": 0.0022,
                "latency_ns_per_activation": 0.015,
                "area_mm2_per_weight": 1.0e-9,
            },
            "digital": {
                "attention": {"energy_pj_per_mac": 0.0004, "latency_ns_per_mac": 0.0007},
                "softmax": {"energy_pj_per_mac": 0.00005, "latency_ns_per_mac": 0.00005},
                "elementwise": {"energy_pj_per_mac": 0.00002, "latency_ns_per_mac": 0.00002},
                "kv_cache": {"energy_pj_per_mac": 0.0001, "latency_ns_per_mac": 0.0001},
                "digital_overhead_area_mm2_per_layer": 0.01,
            },
        }
    }

    if not include_required_sections:
        del payload["puma_like_v1"]["array"]
        del payload["puma_like_v1"]["digital"]

    import json

    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_json_source_preserves_existing_library_behavior() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hardware = HardwareConfig.from_yaml(repo_root / "examples" / "hardware.yaml")
    specs = hardware.resolve_knob_specs()

    assert specs.library == "puma_like_v1"
    assert specs.adc_draft.energy_pj_per_conversion == pytest.approx(0.09)
    assert specs.adc_residual.energy_pj_per_conversion == pytest.approx(0.37)


def test_custom_library_file_switches_source_and_resolves_relative_path(tmp_path: Path) -> None:
    lib_path = tmp_path / "custom_lib.json"
    hw_path = tmp_path / "hardware.yaml"
    _write_custom_library(lib_path)
    _write_hardware(hw_path, library_file="custom_lib.json")

    hardware = HardwareConfig.from_yaml(hw_path)
    specs = hardware.resolve_knob_specs()

    assert hardware.library_file == str(lib_path.resolve())
    assert specs.adc_draft.energy_pj_per_conversion == pytest.approx(0.123)
    assert specs.adc_residual.energy_pj_per_conversion == pytest.approx(0.456)


def test_missing_library_source_file_is_rejected(tmp_path: Path) -> None:
    hw_path = tmp_path / "hardware.yaml"
    _write_hardware(hw_path, library_file="missing_library.json")

    with pytest.raises(ValueError, match="Library source file not found"):
        HardwareConfig.from_yaml(hw_path)


def test_malformed_library_json_is_rejected(tmp_path: Path) -> None:
    lib_path = tmp_path / "bad.json"
    hw_path = tmp_path / "hardware.yaml"
    lib_path.write_text("{", encoding="utf-8")
    _write_hardware(hw_path, library_file="bad.json")

    with pytest.raises(ValueError, match="Invalid library JSON"):
        HardwareConfig.from_yaml(hw_path)


def test_incomplete_library_json_missing_required_sections_is_rejected(tmp_path: Path) -> None:
    lib_path = tmp_path / "incomplete.json"
    hw_path = tmp_path / "hardware.yaml"
    _write_custom_library(lib_path, include_required_sections=False)
    _write_hardware(hw_path, library_file="incomplete.json")

    with pytest.raises(ValueError, match="missing required sections"):
        HardwareConfig.from_yaml(hw_path)


def test_custom_library_missing_required_bit_entry_is_rejected(tmp_path: Path) -> None:
    lib_path = tmp_path / "missing_bit.json"
    hw_path = tmp_path / "hardware.yaml"
    _write_custom_library(lib_path, include_adc12=False)
    _write_hardware(hw_path, library_file="missing_bit.json")

    with pytest.raises(ValueError, match="Available ADC bits"):
        HardwareConfig.from_yaml(hw_path)

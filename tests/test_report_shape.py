import json

import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig
from selfspec_calculator.estimator import estimate_sweep
from selfspec_calculator.stats import SpeculationStats


def test_knob_report_includes_stage_component_and_library_metadata() -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 8,
            "activation_bits": 12,
            "ffn_type": "mlp",
            "ffn_expansion": 4.0,
        }
    )
    hardware = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
        }
    )
    stats = SpeculationStats(k=2, histogram={0: 0.5, 2: 0.5})

    report = estimate_sweep(model=model, hardware=hardware, stats=stats, prompt_lengths=[64])
    payload = report.model_dump(mode="json")

    assert payload["hardware_mode"] == "knob-based"
    assert payload["resolved_library"] is not None
    assert payload["resolved_library"]["name"] == "puma_like_v1"
    assert payload["resolved_library"]["dac"]["bits"] == 4
    assert payload["resolved_library"]["adc_draft"]["bits"] == 4
    assert payload["resolved_library"]["adc_residual"]["bits"] == 12

    point = payload["points"][0]
    assert "delta" in point
    assert "baseline_breakdown" in point

    for phase in ["draft", "verify_drafted", "verify_bonus", "total"]:
        phase_payload = point["breakdown"][phase]
        assert "stages" in phase_payload
        assert "components" in phase_payload
        assert "activation_counts" in phase_payload

        stages = phase_payload["stages"]
        assert "qkv_energy_pj" in stages
        assert "ffn_latency_ns" in stages

        components = phase_payload["components"]
        assert "arrays_energy_pj" in components
        assert "dac_energy_pj" in components
        assert "adc_draft_latency_ns" in components
        assert "attention_engine_energy_pj" in components

        counts = phase_payload["activation_counts"]
        assert "dac_conversions" in counts
        assert "adc_draft_conversions" in counts
        assert "adc_residual_conversions" in counts

    verify_bonus_counts = point["breakdown"]["verify_bonus"]["activation_counts"]
    assert verify_bonus_counts["dac_conversions"] > 0
    assert verify_bonus_counts["adc_draft_conversions"] > 0
    assert verify_bonus_counts["adc_residual_conversions"] > 0

    assert "area_breakdown_mm2" in payload
    assert payload["area_breakdown_mm2"]["on_chip_mm2"] > 0.0
    assert payload["area_breakdown_mm2"]["off_chip_hbm_mm2"] == 0.0
    assert payload["area_breakdown_mm2"]["on_chip_components"]["arrays_mm2"] > 0.0


def test_report_exposes_layer_pipelined_verify_wavefront_policy_metadata() -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 8,
            "activation_bits": 12,
            "ffn_type": "mlp",
            "ffn_expansion": 4.0,
        }
    )
    hardware = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "puma_like_v1",
            "soc": {"schedule": "layer-pipelined"},
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
        }
    )
    stats = SpeculationStats(k=4, histogram={0: 0.5, 4: 0.5})

    report = estimate_sweep(model=model, hardware=hardware, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    policy = report["pipeline_policy"]
    assert policy is not None
    assert policy["schedule"] == "layer-pipelined"
    assert policy["draft_stage"] == "serialized"
    assert policy["verify_stage"] == "wavefront"
    assert policy["mismatch_policy"] == "stop-at-first-mismatch"
    assert policy["bonus_in_verify_wavefront"] is True


def test_report_includes_leakage_summary_and_leakage_power_inputs() -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 8,
            "activation_bits": 12,
            "ffn_type": "mlp",
            "ffn_expansion": 4.0,
        }
    )
    hardware = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
            "leakage_power": {
                "arrays_nw": 2_000_000.0,
                "control_nw": 1_500_000.0,
                "sram_nw": 500_000.0,
            },
        }
    )
    stats = SpeculationStats(k=2, histogram={0: 0.5, 2: 0.5})

    report = estimate_sweep(model=model, hardware=hardware, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    point = report["points"][0]
    leakage = point["leakage"]
    assert leakage is not None
    assert leakage["total_power_nw"] == pytest.approx(4_000_000.0)
    assert leakage["energy_pj"] == pytest.approx(leakage["total_power_nw"] * leakage["burst_latency_ns"] * 1e-6)

    assert "hardware_knobs" in report
    assert "leakage_power" in report["hardware_knobs"]
    assert report["hardware_knobs"]["leakage_power"]["arrays_nw"] == pytest.approx(2_000_000.0)
    assert report["hardware_knobs"]["leakage_power"]["control_nw"] == pytest.approx(1_500_000.0)
    assert report["hardware_knobs"]["leakage_power"]["sram_nw"] == pytest.approx(500_000.0)


def test_area_breakdown_reports_memory_and_periphery_area_and_excludes_hbm_from_on_chip_total() -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 8,
            "activation_bits": 12,
            "ffn_type": "mlp",
            "ffn_expansion": 4.0,
        }
    )
    stats = SpeculationStats(k=2, histogram={0: 1.0})

    base_hw = {
        "reuse_policy": "reuse",
        "library": "puma_like_v1",
        "analog": {
            "xbar_size": 128,
            "num_columns_per_adc": 16,
            "dac_bits": 4,
            "adc": {"draft_bits": 4, "residual_bits": 12},
            "periphery": {"tia": {"area_mm2_per_unit": 0.001}},
        },
        "memory": {
            "sram": {"area_mm2": 2.0},
            "hbm": {"area_mm2": 10.0},
            "fabric": {"area_mm2": 1.0},
        },
    }

    hw0 = HardwareConfig.model_validate(base_hw)
    r0 = estimate_sweep(model=model, hardware=hw0, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    a0 = r0["area_breakdown_mm2"]
    assert a0["off_chip_hbm_mm2"] == 10.0
    assert a0["on_chip_components"]["sram_mm2"] == 2.0
    assert a0["on_chip_components"]["fabric_mm2"] == 1.0
    assert a0["on_chip_components"]["tia_mm2"] > 0.0

    hw1 = HardwareConfig.model_validate(
        {**base_hw, "memory": {**base_hw["memory"], "hbm": {"area_mm2": 999.0}}}
    )
    r1 = estimate_sweep(model=model, hardware=hw1, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    a1 = r1["area_breakdown_mm2"]
    assert a1["off_chip_hbm_mm2"] == 999.0
    assert a1["on_chip_mm2"] == a0["on_chip_mm2"]


def test_array_area_uses_model_derived_array_count_when_area_per_array_is_provided(tmp_path) -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 1,
            "d_model": 64,
            "n_heads": 8,
            "activation_bits": 12,
            "ffn_type": "mlp",
            "ffn_expansion": 4.0,
        }
    )
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    # For d_model=64, ffn=256, xbar=32:
    # qkv: ceil(192/32)*ceil(64/32)=12
    # wo: ceil(64/32)*ceil(64/32)=4
    # ffn: ceil(256/32)*ceil(64/32) + ceil(64/32)*ceil(256/32)=16+16=32
    # total logical arrays per layer = 48
    # with arrays_per_weight=4 => total physical arrays per layer = 192
    area_per_array_mm2 = 0.5
    expected_arrays_mm2 = 48.0 * 4.0 * area_per_array_mm2

    lib_path = tmp_path / "lib.json"
    lib_payload = {
        "array_area_per_array_test": {
            "adc": {
                "4": {
                    "energy_pj_per_conversion": 0.1,
                    "latency_ns_per_conversion": 0.1,
                    "area_mm2_per_unit": 0.001,
                }
            },
            "dac": {
                "4": {
                    "energy_pj_per_conversion": 0.01,
                    "latency_ns_per_conversion": 0.01,
                    "area_mm2_per_unit": 1e-6,
                }
            },
            "array": {
                "energy_pj_per_activation": 0.001,
                "latency_ns_per_activation": 0.01,
                "area_mm2_per_weight": 123.0,
                "area_mm2_per_array": area_per_array_mm2,
                "arrays_per_weight": 4,
            },
            "digital": {
                "attention": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "softmax": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "elementwise": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "kv_cache": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "digital_overhead_area_mm2_per_layer": 0.0,
            },
        }
    }
    lib_path.write_text(json.dumps(lib_payload), encoding="utf-8")

    hardware = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "array_area_per_array_test",
            "library_file": str(lib_path),
            "analog": {
                "xbar_size": 32,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 4},
            },
        }
    )

    report = estimate_sweep(model=model, hardware=hardware, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    arrays_mm2 = report["area_breakdown_mm2"]["on_chip_components"]["arrays_mm2"]
    assert arrays_mm2 == expected_arrays_mm2


def test_periphery_area_counts_both_adc_paths_in_split_architecture(tmp_path) -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 1,
            "d_model": 64,
            "n_heads": 8,
            "activation_bits": 12,
            "ffn_type": "mlp",
            "ffn_expansion": 4.0,
        }
    )
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    # For d_model=64, xbar=32: logical arrays per layer = 48 (see test above).
    # ADC units per path = 48 * (32/16) = 96.
    # With 1+3 split, both ADC paths exist physically, so TIA units = 2 * 96.
    # Shared-DAC model: DAC units = logical arrays * xbar_size = 48 * 32 = 1536.
    expected_adc_units_per_path = 96.0
    expected_tia_units = 192.0
    expected_dac_units = 1536.0

    lib_path = tmp_path / "lib.json"
    lib_payload = {
        "split_area_units_test": {
            "adc": {
                "4": {
                    "energy_pj_per_conversion": 0.0,
                    "latency_ns_per_conversion": 0.0,
                    "area_mm2_per_unit": 1.0,
                }
            },
            "dac": {
                "4": {
                    "energy_pj_per_conversion": 0.0,
                    "latency_ns_per_conversion": 0.0,
                    "area_mm2_per_unit": 1.0,
                }
            },
            "array": {
                "energy_pj_per_activation": 0.0,
                "latency_ns_per_activation": 0.0,
                "area_mm2_per_weight": 0.0,
                "area_mm2_per_array": 0.0,
                "arrays_per_weight": 4,
            },
            "analog_periphery": {
                "tia": {"area_mm2_per_unit": 1.0},
                "snh": {"area_mm2_per_unit": 0.0},
                "mux": {"area_mm2_per_unit": 0.0},
                "io_buffers": {"area_mm2_per_unit": 0.0},
                "subarray_switches": {"area_mm2_per_unit": 0.0},
                "write_drivers": {"area_mm2_per_unit": 0.0},
            },
            "digital": {
                "attention": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "softmax": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "elementwise": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "kv_cache": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "digital_overhead_area_mm2_per_layer": 0.0,
            },
        }
    }
    lib_path.write_text(json.dumps(lib_payload), encoding="utf-8")

    hardware = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "split_area_units_test",
            "library_file": str(lib_path),
            "analog": {
                "xbar_size": 32,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 4},
            },
        }
    )

    report = estimate_sweep(model=model, hardware=hardware, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    area_components = report["area_breakdown_mm2"]["on_chip_components"]
    assert area_components["dac_mm2"] == expected_dac_units
    assert area_components["adc_draft_mm2"] == expected_adc_units_per_path
    assert area_components["adc_residual_mm2"] == expected_adc_units_per_path
    assert area_components["tia_mm2"] == expected_tia_units

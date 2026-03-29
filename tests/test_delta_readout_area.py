import json

import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig
from selfspec_calculator.estimator import estimate_sweep
from selfspec_calculator.stats import SpeculationStats


def test_delta_readout_area_adds_one_extra_dac_per_logical_tile(tmp_path) -> None:
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

    # For d_model=64, xbar=32, mlp FFN:
    # qkv: ceil(192/32)*ceil(64/32) = 12
    # wo:  ceil(64/32)*ceil(64/32) = 4
    # ffn up/down: 16 + 16
    # total logical tiles per layer = 48
    baseline_dac_units = 48.0 * 32.0
    delta_extra_dac_units = 48.0

    lib_path = tmp_path / "lib.json"
    lib_payload = {
        "delta_area_units_test": {
            "adc": {
                "4": {
                    "energy_pj_per_conversion": 0.0,
                    "latency_ns_per_conversion": 0.0,
                    "area_mm2_per_unit": 0.0,
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

    hardware_base = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "delta_area_units_test",
            "library_file": str(lib_path),
            "analog": {
                "xbar_size": 32,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 4},
            },
        }
    )
    hardware_delta = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "delta_area_units_test",
            "library_file": str(lib_path),
            "analog": {
                "xbar_size": 32,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 4},
                "delta_readout": {
                    "draft": {"enabled": True},
                    "verify": {"enabled": True},
                },
            },
        }
    )

    baseline = estimate_sweep(model=model, hardware=hardware_base, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    delta = estimate_sweep(model=model, hardware=hardware_delta, stats=stats, prompt_lengths=[64]).model_dump(mode="json")

    assert baseline["area_breakdown_mm2"]["on_chip_components"]["dac_mm2"] == pytest.approx(baseline_dac_units)
    assert delta["area_breakdown_mm2"]["on_chip_components"]["dac_mm2"] == pytest.approx(
        baseline_dac_units + delta_extra_dac_units
    )

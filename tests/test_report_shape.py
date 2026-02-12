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

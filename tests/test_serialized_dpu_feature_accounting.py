import json
from copy import deepcopy
from pathlib import Path

import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig
from selfspec_calculator.estimator import estimate_point, estimate_sweep
from selfspec_calculator.stats import SpeculationStats


BASE_MODEL = {
    "n_layers": 1,
    "d_model": 16,
    "n_heads": 4,
    "activation_bits": 8,
    "ffn_type": "mlp",
    "ffn_expansion": 2.0,
}


def _base_knob_hardware(*, overrides: dict | None = None) -> HardwareConfig:
    payload = {
        "reuse_policy": "reuse",
        "library": "puma_like_v1",
        "analog": {
            "xbar_size": 16,
            "num_columns_per_adc": 4,
            "dac_bits": 4,
            "adc": {"draft_bits": 4, "residual_bits": 4},
        },
    }
    if overrides:
        payload.update(overrides)
    return HardwareConfig.model_validate(payload)


def _step_index_legacy_hardware() -> HardwareConfig:
    return HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "costs": {
                "analog_draft": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "analog_full": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "analog_verify_reuse": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "digital_attention": {"energy_pj_per_mac": 1.0, "latency_ns_per_mac": 1.0},
                "digital_softmax": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "digital_elementwise": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "kv_cache": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
                "analog_weight_area": {"area_mm2_per_weight": 0.0},
                "digital_overhead_area_mm2_per_layer": 0.0,
            },
        }
    )


def _write_custom_library(path: Path, *, qk_cost: float | None = None, elementwise_cost: float = 1.0) -> str:
    digital = {
        "attention": {"energy_pj_per_mac": 1.0, "latency_ns_per_mac": 1.0},
        "softmax": {"energy_pj_per_mac": 1.0, "latency_ns_per_mac": 1.0},
        "elementwise": {"energy_pj_per_mac": elementwise_cost, "latency_ns_per_mac": elementwise_cost},
        "kv_cache": {"energy_pj_per_mac": 1.0, "latency_ns_per_mac": 1.0},
        "digital_overhead_area_mm2_per_layer": 0.0,
    }
    if qk_cost is not None:
        digital["features"] = {
            "attention_qk": {"energy_pj_per_mac": qk_cost, "latency_ns_per_mac": qk_cost},
        }

    payload = {
        "feature_override_test": {
            "adc": {"4": {"energy_pj_per_conversion": 0.0, "latency_ns_per_conversion": 0.0, "area_mm2_per_unit": 0.0}},
            "dac": {"4": {"energy_pj_per_conversion": 0.0, "latency_ns_per_conversion": 0.0, "area_mm2_per_unit": 0.0}},
            "array": {
                "energy_pj_per_activation": 0.0,
                "latency_ns_per_activation": 0.0,
                "area_mm2_per_weight": 0.0,
            },
            "digital": digital,
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return "feature_override_test"


def _custom_library_hardware(*, tmp_path: Path, qk_cost: float | None = None, elementwise_cost: float = 1.0) -> HardwareConfig:
    lib_path = tmp_path / f"lib_{qk_cost}_{elementwise_cost}.json"
    library_name = _write_custom_library(lib_path, qk_cost=qk_cost, elementwise_cost=elementwise_cost)
    return _base_knob_hardware(
        overrides={
            "library": library_name,
            "library_file": str(lib_path),
        }
    )


def test_serialized_draft_and_verify_drafted_use_step_indexed_attention_costs() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _step_index_legacy_hardware()
    k = 4
    l_prompt = 10
    stats = SpeculationStats(k=k, histogram={0: 1.0})

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=l_prompt)

    d_head = model.d_head
    expected_qk_ops = sum(model.n_heads * (l_prompt + i) * d_head for i in range(k))
    assert breakdown.draft.stages.qk_energy_pj == pytest.approx(expected_qk_ops)
    assert breakdown.verify_drafted.stages.qk_energy_pj == pytest.approx(expected_qk_ops)


def test_serialized_verify_bonus_uses_prompt_plus_k_context() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _step_index_legacy_hardware()
    k = 3
    l_prompt = 12
    stats = SpeculationStats(k=k, histogram={0: 1.0})

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=l_prompt)
    expected_bonus_qk_ops = model.n_heads * (l_prompt + k) * model.d_head
    assert breakdown.verify_bonus.stages.qk_energy_pj == pytest.approx(expected_bonus_qk_ops)


def test_compute_and_movement_channels_are_separated_without_double_counting() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _base_knob_hardware(
        overrides={
            "memory": {
                "sram": {"read_energy_pj_per_byte": 0.1, "write_energy_pj_per_byte": 0.2},
                "hbm": {"read_energy_pj_per_byte": 1.0, "write_energy_pj_per_byte": 2.0},
                "fabric": {"read_energy_pj_per_byte": 0.01, "write_energy_pj_per_byte": 0.01},
            }
        }
    )
    stats = SpeculationStats(k=2, histogram={0: 1.0})

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=16)

    assert breakdown.total.channels is not None
    assert breakdown.total.channels.movement_energy_pj > 0.0
    assert breakdown.total.channels.compute_energy_pj > 0.0
    assert breakdown.total.energy_pj == pytest.approx(
        breakdown.total.channels.compute_energy_pj + breakdown.total.channels.movement_energy_pj
    )
    assert breakdown.total.dpu_features is not None
    assert breakdown.total.dpu_features.kv_cache_update_energy_pj == pytest.approx(0.0)


def test_report_includes_structured_movement_exclusions() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _base_knob_hardware(
        overrides={
            "memory": {
                "sram": {"read_energy_pj_per_byte": 0.1, "write_energy_pj_per_byte": 0.2},
                "hbm": {"read_energy_pj_per_byte": 1.0, "write_energy_pj_per_byte": 2.0},
                "fabric": {"read_energy_pj_per_byte": 0.01, "write_energy_pj_per_byte": 0.01},
            }
        }
    )
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    report = estimate_sweep(model=model, hardware=hardware, stats=stats, prompt_lengths=[16])
    assert report.movement_accounting is not None
    assert any("non_kv_intermediate" in item for item in report.movement_accounting.excluded)


def test_explicit_feature_override_changes_only_that_feature_contribution(tmp_path: Path) -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    hw_base = _custom_library_hardware(tmp_path=tmp_path, qk_cost=None)
    hw_override = _custom_library_hardware(tmp_path=tmp_path, qk_cost=3.0)

    _, base_breakdown = estimate_point(model=model, hardware=hw_base, stats=stats, l_prompt=8)
    _, override_breakdown = estimate_point(model=model, hardware=hw_override, stats=stats, l_prompt=8)
    override_report = estimate_sweep(model=model, hardware=hw_override, stats=stats, prompt_lengths=[8])

    assert override_report.dpu_feature_mapping is not None
    assert override_report.dpu_feature_mapping["attention_qk"].startswith("explicit:")
    assert override_report.dpu_feature_mapping["attention_pv"].startswith("mapped:")

    assert base_breakdown.draft.dpu_features is not None
    assert override_breakdown.draft.dpu_features is not None
    assert override_breakdown.draft.dpu_features.attention_qk_energy_pj > base_breakdown.draft.dpu_features.attention_qk_energy_pj

    assert override_breakdown.draft.dpu_features.attention_pv_energy_pj == pytest.approx(
        base_breakdown.draft.dpu_features.attention_pv_energy_pj
    )
    assert override_breakdown.draft.dpu_features.attention_softmax_energy_pj == pytest.approx(
        base_breakdown.draft.dpu_features.attention_softmax_energy_pj
    )
    assert override_breakdown.draft.dpu_features.ffn_activation_energy_pj == pytest.approx(
        base_breakdown.draft.dpu_features.ffn_activation_energy_pj
    )


def test_attention_parallel_units_scale_feature_latency() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    hw1 = _base_knob_hardware(overrides={"soc": {"attention_cim_units": 1}})
    hw8 = _base_knob_hardware(overrides={"soc": {"attention_cim_units": 8}})

    _, b1 = estimate_point(model=model, hardware=hw1, stats=stats, l_prompt=16)
    _, b8 = estimate_point(model=model, hardware=hw8, stats=stats, l_prompt=16)

    assert b1.draft.dpu_features is not None
    assert b8.draft.dpu_features is not None
    assert b8.draft.dpu_features.attention_qk_latency_ns == pytest.approx(b1.draft.dpu_features.attention_qk_latency_ns / 8.0)
    assert b8.draft.dpu_features.attention_pv_latency_ns == pytest.approx(b1.draft.dpu_features.attention_pv_latency_ns / 8.0)
    assert b8.draft.dpu_features.attention_qk_energy_pj == pytest.approx(b1.draft.dpu_features.attention_qk_energy_pj)
    assert b8.draft.dpu_features.attention_pv_energy_pj == pytest.approx(b1.draft.dpu_features.attention_pv_energy_pj)


def test_swiglu_gate_feature_is_accounted_separately(tmp_path: Path) -> None:
    stats = SpeculationStats(k=1, histogram={0: 1.0})
    hw = _custom_library_hardware(tmp_path=tmp_path, qk_cost=None, elementwise_cost=1.0)

    model_mlp = ModelConfig.model_validate(BASE_MODEL)
    model_swiglu = ModelConfig.model_validate({**BASE_MODEL, "ffn_type": "swiglu"})

    _, b_mlp = estimate_point(model=model_mlp, hardware=hw, stats=stats, l_prompt=8)
    _, b_swiglu = estimate_point(model=model_swiglu, hardware=hw, stats=stats, l_prompt=8)

    assert b_mlp.draft.dpu_features is not None
    assert b_swiglu.draft.dpu_features is not None

    assert b_mlp.draft.dpu_features.ffn_gate_multiply_ops == pytest.approx(0.0)
    assert b_mlp.draft.dpu_features.ffn_gate_multiply_energy_pj == pytest.approx(0.0)
    assert b_swiglu.draft.dpu_features.ffn_gate_multiply_ops > 0.0
    assert b_swiglu.draft.dpu_features.ffn_gate_multiply_energy_pj > 0.0
    assert (
        b_swiglu.draft.stages.elementwise_energy_pj - b_mlp.draft.stages.elementwise_energy_pj
    ) == pytest.approx(b_swiglu.draft.dpu_features.ffn_gate_multiply_energy_pj)


def test_compatibility_mapping_matches_explicit_equivalent_library(tmp_path: Path) -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=2, histogram={0: 1.0})

    default_hw = _base_knob_hardware()
    runtime_libs = default_hw.runtime_libraries()
    explicit = deepcopy(runtime_libs["puma_like_v1"])
    explicit["digital"]["features"] = {
        "attention_qk": deepcopy(explicit["digital"]["attention"]),
        "attention_softmax": deepcopy(explicit["digital"]["softmax"]),
        "attention_pv": deepcopy(explicit["digital"]["attention"]),
        "ffn_activation": deepcopy(explicit["digital"]["elementwise"]),
        "ffn_gate_multiply": deepcopy(explicit["digital"]["elementwise"]),
        "kv_cache_update": deepcopy(explicit["digital"]["kv_cache"]),
    }

    lib_path = tmp_path / "explicit_lib.json"
    lib_path.write_text(json.dumps({"explicit_feature_lib": explicit}), encoding="utf-8")
    explicit_hw = _base_knob_hardware(
        overrides={
            "library": "explicit_feature_lib",
            "library_file": str(lib_path),
        }
    )

    m_default, b_default = estimate_point(model=model, hardware=default_hw, stats=stats, l_prompt=16)
    m_explicit, b_explicit = estimate_point(model=model, hardware=explicit_hw, stats=stats, l_prompt=16)

    assert m_explicit.energy_pj_per_token == pytest.approx(m_default.energy_pj_per_token)
    assert m_explicit.latency_ns_per_token == pytest.approx(m_default.latency_ns_per_token)
    assert b_explicit.total.energy_pj == pytest.approx(b_default.total.energy_pj)
    assert b_explicit.total.latency_ns == pytest.approx(b_default.total.latency_ns)

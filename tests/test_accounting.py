from math import ceil

import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig
from selfspec_calculator.estimator import estimate_point
from selfspec_calculator.stats import SpeculationStats


BASE_MODEL = {
    "n_layers": 1,
    "d_model": 128,
    "n_heads": 8,
    "activation_bits": 12,
    "ffn_type": "mlp",
    "ffn_expansion": 4.0,
}


def _knob_hardware(*, reuse_policy: str = "reuse", dac_bits: int = 4) -> HardwareConfig:
    return HardwareConfig.model_validate(
        {
            "reuse_policy": reuse_policy,
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": dac_bits,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
        }
    )


def _legacy_hardware(*, reuse_policy: str = "reuse", buffers_latency_ns_per_op: float = 0.0) -> HardwareConfig:
    return HardwareConfig.model_validate(
        {
            "reuse_policy": reuse_policy,
            "costs": {
                "analog_draft": {"energy_pj_per_mac": 0.001, "latency_ns_per_mac": 0.001},
                "analog_full": {"energy_pj_per_mac": 0.002, "latency_ns_per_mac": 0.0015},
                "analog_verify_reuse": {"energy_pj_per_mac": 0.0006, "latency_ns_per_mac": 0.0008},
                "digital_attention": {"energy_pj_per_mac": 0.0004, "latency_ns_per_mac": 0.0007},
                "digital_softmax": {"energy_pj_per_mac": 0.00005, "latency_ns_per_mac": 0.00005},
                "digital_elementwise": {"energy_pj_per_mac": 0.00002, "latency_ns_per_mac": 0.00002},
                "kv_cache": {"energy_pj_per_mac": 0.0001, "latency_ns_per_mac": 0.0001},
                "analog_weight_area": {"area_mm2_per_weight": 1e-9},
                "digital_overhead_area_mm2_per_layer": 0.01,
            },
            "soc": {
                "buffers_add": {"energy_pj_per_op": 0.01, "latency_ns_per_op": buffers_latency_ns_per_op},
            },
        }
    )


def test_dac_slicing_scales_analog_counts_and_dac_energy() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=0, histogram={0: 1.0})

    _, wide_breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(dac_bits=12),
        stats=stats,
        l_prompt=64,
    )
    _, sliced_breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(dac_bits=4),
        stats=stats,
        l_prompt=64,
    )

    wide_counts = wide_breakdown.verify_bonus.activation_counts
    sliced_counts = sliced_breakdown.verify_bonus.activation_counts
    assert wide_counts is not None
    assert sliced_counts is not None

    assert sliced_counts.dac_conversions == pytest.approx(wide_counts.dac_conversions * 3.0)
    assert sliced_counts.adc_draft_conversions == pytest.approx(wide_counts.adc_draft_conversions * 3.0)
    assert sliced_counts.adc_residual_conversions == pytest.approx(wide_counts.adc_residual_conversions * 3.0)

    wide_components = wide_breakdown.verify_bonus.components
    sliced_components = sliced_breakdown.verify_bonus.components
    assert wide_components is not None
    assert sliced_components is not None
    assert sliced_components.arrays_energy_pj == pytest.approx(wide_components.arrays_energy_pj * 3.0)
    assert sliced_components.adc_draft_energy_pj == pytest.approx(wide_components.adc_draft_energy_pj * 3.0)
    assert sliced_components.adc_residual_energy_pj == pytest.approx(wide_components.adc_residual_energy_pj * 3.0)
    assert sliced_components.dac_energy_pj > wide_components.dac_energy_pj


def test_knob_verify_drafted_reread_uses_full_reads() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    _, reuse_breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(reuse_policy="reuse", dac_bits=4),
        stats=stats,
        l_prompt=64,
    )
    _, reread_breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(reuse_policy="reread", dac_bits=4),
        stats=stats,
        l_prompt=64,
    )

    reuse_counts = reuse_breakdown.verify_drafted.activation_counts
    reread_counts = reread_breakdown.verify_drafted.activation_counts
    assert reuse_counts is not None
    assert reread_counts is not None

    assert reuse_counts.adc_draft_conversions == pytest.approx(0.0)
    assert reuse_counts.adc_residual_conversions > 0.0
    assert reread_counts.adc_draft_conversions == pytest.approx(reuse_counts.adc_residual_conversions)
    assert reread_counts.adc_residual_conversions == pytest.approx(reuse_counts.adc_residual_conversions)
    assert reread_counts.array_activations > reuse_counts.array_activations
    assert reread_breakdown.verify_drafted.energy_pj > reuse_breakdown.verify_drafted.energy_pj


def test_reuse_with_full_precision_draft_has_zero_verify_analog_reads() -> None:
    model = ModelConfig.model_validate(
        {
            **BASE_MODEL,
            "draft_policy": {
                "default": {"qkv": "full", "wo": "full", "ffn": "full"},
            },
        }
    )
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    _, breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(reuse_policy="reuse", dac_bits=4),
        stats=stats,
        l_prompt=64,
    )

    counts = breakdown.verify_drafted.activation_counts
    components = breakdown.verify_drafted.components
    assert counts is not None
    assert components is not None

    assert counts.array_activations == pytest.approx(0.0)
    assert counts.dac_conversions == pytest.approx(0.0)
    assert counts.adc_draft_conversions == pytest.approx(0.0)
    assert counts.adc_residual_conversions == pytest.approx(0.0)

    assert components.arrays_energy_pj == pytest.approx(0.0)
    assert components.dac_energy_pj == pytest.approx(0.0)
    assert components.adc_draft_energy_pj == pytest.approx(0.0)
    assert components.adc_residual_energy_pj == pytest.approx(0.0)


def test_phase_specific_buffers_add_overrides_apply_to_draft_and_verify() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})
    hardware = HardwareConfig.model_validate(
        {
            "reuse_policy": "reread",
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
            "soc": {
                "buffers_add": {
                    "energy_pj_per_op": 0.01,
                    "latency_ns_per_op": 0.0,
                    "draft": {"latency_ns_per_op": 0.0},
                    "verify": {"latency_ns_per_op": 10.0},
                },
            },
        }
    )

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=64)

    assert breakdown.draft.components is not None
    assert breakdown.verify_drafted.components is not None
    assert breakdown.verify_bonus.components is not None
    assert breakdown.draft.components.buffers_add_latency_ns == pytest.approx(0.0)
    assert breakdown.verify_drafted.components.buffers_add_latency_ns > 0.0
    assert breakdown.verify_bonus.components.buffers_add_latency_ns > 0.0


def test_split_adc_modes_use_shared_dac_across_active_arrays() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    _, reuse_breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(reuse_policy="reuse", dac_bits=4),
        stats=stats,
        l_prompt=64,
    )
    reuse_verify_counts = reuse_breakdown.verify_drafted.activation_counts
    reuse_bonus_counts = reuse_breakdown.verify_bonus.activation_counts
    assert reuse_verify_counts is not None
    assert reuse_bonus_counts is not None

    # Reuse verify for drafted tokens: residual-only read (Arrays 2-4, ADC-Residual only).
    assert reuse_verify_counts.adc_draft_conversions == pytest.approx(0.0)
    assert reuse_verify_counts.adc_residual_conversions > 0.0
    assert reuse_verify_counts.dac_conversions == pytest.approx(reuse_verify_counts.adc_residual_conversions)

    # Bonus token: full read (Arrays 1-4, both ADC paths).
    assert reuse_bonus_counts.adc_draft_conversions > 0.0
    assert reuse_bonus_counts.adc_residual_conversions > 0.0
    assert reuse_bonus_counts.adc_draft_conversions == pytest.approx(reuse_bonus_counts.adc_residual_conversions)
    assert reuse_bonus_counts.dac_conversions == pytest.approx(reuse_bonus_counts.adc_draft_conversions)

    _, reread_breakdown = estimate_point(
        model=model,
        hardware=_knob_hardware(reuse_policy="reread", dac_bits=4),
        stats=stats,
        l_prompt=64,
    )
    reread_verify_counts = reread_breakdown.verify_drafted.activation_counts
    assert reread_verify_counts is not None

    # Reread verify for drafted tokens: full read, matching verify-bonus ADC usage.
    assert reread_verify_counts.adc_draft_conversions == pytest.approx(reuse_bonus_counts.adc_draft_conversions)
    assert reread_verify_counts.adc_residual_conversions == pytest.approx(reuse_bonus_counts.adc_residual_conversions)
    assert reread_verify_counts.dac_conversions == pytest.approx(reuse_bonus_counts.dac_conversions)


def test_split_adc_columns_reduce_draft_and_residual_scan_latency() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    baseline_hw = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 128,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
        }
    )
    split_hw = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 128,
                "dac_bits": 4,
                "adc": {
                    "num_columns_per_adc": {"draft": 16, "residual": 32},
                    "draft_bits": 4,
                    "residual_bits": 12,
                },
            },
        }
    )

    _, baseline_breakdown = estimate_point(model=model, hardware=baseline_hw, stats=stats, l_prompt=64)
    _, split_breakdown = estimate_point(model=model, hardware=split_hw, stats=stats, l_prompt=64)

    assert baseline_breakdown.draft.components is not None
    assert split_breakdown.draft.components is not None
    assert baseline_breakdown.verify_drafted.components is not None
    assert split_breakdown.verify_drafted.components is not None

    assert split_breakdown.draft.components.adc_draft_latency_ns == pytest.approx(
        baseline_breakdown.draft.components.adc_draft_latency_ns / 8.0
    )
    assert split_breakdown.verify_drafted.components.adc_residual_latency_ns == pytest.approx(
        baseline_breakdown.verify_drafted.components.adc_residual_latency_ns / 4.0
    )


def test_draft_activation_bit_override_reduces_draft_slices_only() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    base_hw = _knob_hardware(dac_bits=4)
    low_draft_hw = HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
            "soc": {
                "draft_activation_bits": 8,
                "verify_activation_bits": 16,
            },
        }
    )

    _, base_breakdown = estimate_point(model=model, hardware=base_hw, stats=stats, l_prompt=64)
    _, low_breakdown = estimate_point(model=model, hardware=low_draft_hw, stats=stats, l_prompt=64)

    assert base_breakdown.draft.activation_counts is not None
    assert low_breakdown.draft.activation_counts is not None
    assert base_breakdown.verify_bonus.activation_counts is not None
    assert low_breakdown.verify_bonus.activation_counts is not None

    assert low_breakdown.draft.activation_counts.dac_conversions == pytest.approx(
        base_breakdown.draft.activation_counts.dac_conversions * (2.0 / 3.0)
    )
    assert low_breakdown.verify_bonus.activation_counts.dac_conversions == pytest.approx(
        base_breakdown.verify_bonus.activation_counts.dac_conversions * (4.0 / 3.0)
    )


def test_legacy_buffers_add_latency_uses_incremental_overlap() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})
    hardware = _legacy_hardware(reuse_policy="reuse", buffers_latency_ns_per_op=1.0)

    _, breakdown = estimate_point(
        model=model,
        hardware=hardware,
        stats=stats,
        l_prompt=64,
    )

    outputs = {"qkv": 3 * model.d_model, "wo": model.d_model, "ffn": model.effective_d_ff + model.d_model}
    analog_macs = {
        "qkv": 3 * model.d_model * model.d_model,
        "wo": model.d_model * model.d_model,
        "ffn": 2 * model.d_model * model.effective_d_ff,
    }
    expected_draft = sum(
        max(0.0, float(outputs[name]) - analog_macs[name] * hardware.costs.analog_draft.latency_ns_per_mac)
        for name in outputs
    )
    expected_verify = sum(
        max(0.0, 2.0 * float(outputs[name]) - analog_macs[name] * hardware.costs.analog_verify_reuse.latency_ns_per_mac)
        for name in outputs
    )
    raw_serialized = sum(float(v) for v in outputs.values())

    assert breakdown.draft.stages.buffers_add_latency_ns == pytest.approx(expected_draft)
    assert breakdown.verify_drafted.stages.buffers_add_latency_ns == pytest.approx(expected_verify)
    assert breakdown.draft.stages.buffers_add_latency_ns < raw_serialized
    assert breakdown.verify_drafted.stages.buffers_add_latency_ns > breakdown.draft.stages.buffers_add_latency_ns

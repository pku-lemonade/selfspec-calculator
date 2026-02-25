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


def test_reuse_vs_reread_changes_verify_drafted_analog_work() -> None:
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
    assert reread_counts.adc_draft_conversions > 0.0
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

    # Re-read verify for drafted tokens: full read (Arrays 1-4, both ADC paths).
    assert reread_verify_counts.adc_draft_conversions > 0.0
    assert reread_verify_counts.adc_residual_conversions > 0.0
    assert reread_verify_counts.adc_draft_conversions == pytest.approx(reread_verify_counts.adc_residual_conversions)
    assert reread_verify_counts.dac_conversions == pytest.approx(reread_verify_counts.adc_draft_conversions)

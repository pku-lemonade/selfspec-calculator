import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig
from selfspec_calculator.estimator import estimate_point
from selfspec_calculator.stats import SpeculationStats


BASE_MODEL = {
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 8,
    "activation_bits": 12,
    "ffn_type": "mlp",
    "ffn_expansion": 4.0,
}


def _base_knob_hardware(**overrides) -> HardwareConfig:
    return HardwareConfig.model_validate(
        {
            "reuse_policy": "reuse",
            "library": "puma_like_v1",
            "analog": {
                "xbar_size": 128,
                "num_columns_per_adc": 16,
                "dac_bits": 4,
                "adc": {"draft_bits": 4, "residual_bits": 12},
            },
            **overrides,
        }
    )


def _memory_hardware(*, k: int, histogram: dict[int, float], l_prompt: int) -> tuple[ModelConfig, HardwareConfig, SpeculationStats, int]:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _base_knob_hardware(
        memory={
            "sram": {
                "read_energy_pj_per_byte": 0.1,
                "write_energy_pj_per_byte": 0.2,
                "read_bandwidth_GBps": 1000.0,
                "write_bandwidth_GBps": 1000.0,
            },
            "hbm": {
                "read_energy_pj_per_byte": 1.0,
                "write_energy_pj_per_byte": 2.0,
                "read_bandwidth_GBps": 1000.0,
                "write_bandwidth_GBps": 1000.0,
            },
            "fabric": {
                "read_energy_pj_per_byte": 0.01,
                "write_energy_pj_per_byte": 0.01,
                "read_bandwidth_GBps": 1000.0,
                "write_bandwidth_GBps": 1000.0,
            },
            "kv_cache": {
                "hbm": {"value_bytes_per_elem": 1, "scale_bytes": 2, "scales_per_token_per_head": 2},
            },
        }
    )
    stats = SpeculationStats(k=k, histogram=histogram)
    return model, hardware, stats, l_prompt


def test_backward_compatible_defaults_keep_new_fields_zero_or_null() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _base_knob_hardware()
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=64)

    assert breakdown.total.memory_traffic is None
    assert breakdown.total.stages.buffers_add_energy_pj == pytest.approx(0.0)
    assert breakdown.total.stages.control_energy_pj == pytest.approx(0.0)

    assert breakdown.total.components is not None
    assert breakdown.total.components.tia_energy_pj == pytest.approx(0.0)
    assert breakdown.total.components.sram_energy_pj == pytest.approx(0.0)
    assert breakdown.total.components.hbm_energy_pj == pytest.approx(0.0)
    assert breakdown.total.components.fabric_energy_pj == pytest.approx(0.0)


def test_kv_hbm_read_bytes_increase_monotonically_with_l_prompt() -> None:
    model, hardware, stats, _ = _memory_hardware(k=2, histogram={0: 1.0, 2: 0.0}, l_prompt=0)

    _, b32 = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=32)
    _, b64 = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=64)

    assert b32.total.memory_traffic is not None
    assert b64.total.memory_traffic is not None
    assert b64.total.memory_traffic.hbm_read_bytes > b32.total.memory_traffic.hbm_read_bytes


def test_kv_traffic_increases_with_k() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _base_knob_hardware(
        memory={
            "sram": {"read_energy_pj_per_byte": 0.1, "write_energy_pj_per_byte": 0.2},
            "hbm": {"read_energy_pj_per_byte": 1.0, "write_energy_pj_per_byte": 2.0},
            "fabric": {"read_energy_pj_per_byte": 0.01, "write_energy_pj_per_byte": 0.01},
        }
    )

    _, b1 = estimate_point(model=model, hardware=hardware, stats=SpeculationStats(k=1, histogram={0: 1.0}), l_prompt=64)
    _, b4 = estimate_point(model=model, hardware=hardware, stats=SpeculationStats(k=4, histogram={0: 1.0}), l_prompt=64)

    assert b1.total.memory_traffic is not None
    assert b4.total.memory_traffic is not None
    assert b4.total.memory_traffic.hbm_read_bytes > b1.total.memory_traffic.hbm_read_bytes
    assert b4.total.memory_traffic.sram_read_bytes > b1.total.memory_traffic.sram_read_bytes


def test_hbm_read_bytes_independent_of_acceptance_histogram() -> None:
    model, hardware, _, l_prompt = _memory_hardware(k=4, histogram={0: 1.0}, l_prompt=64)

    _, b_mismatch = estimate_point(
        model=model,
        hardware=hardware,
        stats=SpeculationStats(k=4, histogram={0: 1.0}),
        l_prompt=l_prompt,
    )
    _, b_accept = estimate_point(
        model=model,
        hardware=hardware,
        stats=SpeculationStats(k=4, histogram={4: 1.0}),
        l_prompt=l_prompt,
    )

    assert b_mismatch.total.memory_traffic is not None
    assert b_accept.total.memory_traffic is not None
    assert b_mismatch.total.memory_traffic.hbm_read_bytes == pytest.approx(b_accept.total.memory_traffic.hbm_read_bytes)


def test_hbm_write_bytes_are_commit_only() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    d_model = model.d_model
    n_heads = model.n_heads
    n_layers = model.n_layers

    fmt = {"value_bytes_per_elem": 1, "scale_bytes": 2, "scales_per_token_per_head": 2}
    bytes_per_token_per_layer = 2 * d_model * fmt["value_bytes_per_elem"] + n_heads * fmt["scales_per_token_per_head"] * fmt["scale_bytes"]

    hardware = _base_knob_hardware(
        memory={
            "sram": {"read_energy_pj_per_byte": 0.1, "write_energy_pj_per_byte": 0.2},
            "hbm": {"read_energy_pj_per_byte": 1.0, "write_energy_pj_per_byte": 2.0},
            "fabric": {"read_energy_pj_per_byte": 0.01, "write_energy_pj_per_byte": 0.01},
            "kv_cache": {"hbm": fmt},
        }
    )

    k = 4
    _, b0 = estimate_point(model=model, hardware=hardware, stats=SpeculationStats(k=k, histogram={0: 1.0}), l_prompt=64)
    _, bK = estimate_point(model=model, hardware=hardware, stats=SpeculationStats(k=k, histogram={k: 1.0}), l_prompt=64)

    assert b0.draft.memory_traffic is not None
    assert b0.verify_drafted.memory_traffic is not None
    assert b0.verify_bonus.memory_traffic is not None
    assert b0.draft.memory_traffic.hbm_write_bytes == pytest.approx(0.0)
    assert b0.verify_drafted.memory_traffic.hbm_write_bytes == pytest.approx(0.0)

    assert bK.draft.memory_traffic is not None
    assert bK.verify_drafted.memory_traffic is not None
    assert bK.verify_bonus.memory_traffic is not None
    assert bK.draft.memory_traffic.hbm_write_bytes == pytest.approx(0.0)
    assert bK.verify_drafted.memory_traffic.hbm_write_bytes == pytest.approx(0.0)

    expected_mismatch = 1.0 * n_layers * bytes_per_token_per_layer
    expected_accept = float(k + 1) * n_layers * bytes_per_token_per_layer
    assert b0.verify_bonus.memory_traffic.hbm_write_bytes == pytest.approx(expected_mismatch)
    assert bK.verify_bonus.memory_traffic.hbm_write_bytes == pytest.approx(expected_accept)


def test_nonzero_periphery_buffers_control_increase_totals() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    baseline_hw = _base_knob_hardware()
    enabled_hw = _base_knob_hardware(
        soc={
            "buffers_add": {"energy_pj_per_op": 0.01, "latency_ns_per_op": 0.02},
            "control": {"energy_pj_per_token": 1.0, "latency_ns_per_token": 2.0},
        },
        analog={
            "xbar_size": 128,
            "num_columns_per_adc": 16,
            "dac_bits": 4,
            "adc": {"draft_bits": 4, "residual_bits": 12},
            "periphery": {"tia": {"energy_pj_per_op": 0.001, "latency_ns_per_op": 0.002}},
        },
    )

    _, b0 = estimate_point(model=model, hardware=baseline_hw, stats=stats, l_prompt=64)
    _, b1 = estimate_point(model=model, hardware=enabled_hw, stats=stats, l_prompt=64)

    assert b0.total.components is not None
    assert b1.total.components is not None
    assert b0.total.components.tia_energy_pj == pytest.approx(0.0)
    assert b1.total.components.tia_energy_pj > 0.0
    assert b1.total.stages.buffers_add_energy_pj > 0.0
    assert b1.total.stages.control_energy_pj > 0.0
    assert b1.total.energy_pj > b0.total.energy_pj


def test_layer_pipelined_schedule_reduces_reported_latency() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware_serial = _base_knob_hardware(soc={"schedule": "serialized"})
    hardware_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})
    stats = SpeculationStats(k=2, histogram={0: 1.0})

    m0, _ = estimate_point(model=model, hardware=hardware_serial, stats=stats, l_prompt=64)
    m1, _ = estimate_point(model=model, hardware=hardware_pipe, stats=stats, l_prompt=64)

    assert m1.energy_pj_per_token == pytest.approx(m0.energy_pj_per_token)
    assert m1.latency_ns_per_token <= m0.latency_ns_per_token


def test_layer_pipelined_can_be_memory_bottlenecked_and_is_not_naive_divide_by_layers() -> None:
    model = ModelConfig.model_validate(
        {
            "n_layers": 4,
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
        "costs": {
            "analog_draft": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "analog_full": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "analog_verify_reuse": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "digital_attention": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "digital_softmax": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "digital_elementwise": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "kv_cache": {"energy_pj_per_mac": 0.0, "latency_ns_per_mac": 0.0},
            "analog_weight_area": {"area_mm2_per_weight": 0.0},
            "digital_overhead_area_mm2_per_layer": 0.0,
        },
        "memory": {
            "sram": {"read_bandwidth_GBps": 1e12, "write_bandwidth_GBps": 1e12},
            "hbm": {"read_bandwidth_GBps": 1e12, "write_bandwidth_GBps": 1e12, "read_latency_ns": 1000.0},
            "fabric": {"read_bandwidth_GBps": 1e12, "write_bandwidth_GBps": 1e12},
        },
    }

    hw_serial = HardwareConfig.model_validate({**base_hw, "soc": {"schedule": "serialized"}})
    hw_pipe = HardwareConfig.model_validate({**base_hw, "soc": {"schedule": "layer-pipelined"}})

    m0, _ = estimate_point(model=model, hardware=hw_serial, stats=stats, l_prompt=64)
    m1, _ = estimate_point(model=model, hardware=hw_pipe, stats=stats, l_prompt=64)

    assert m1.energy_pj_per_token == pytest.approx(m0.energy_pj_per_token)
    assert m1.latency_ns_per_token == pytest.approx(m0.latency_ns_per_token)
    assert m1.latency_ns_per_token > (m0.latency_ns_per_token / model.n_layers) * 2.0


def test_verify_setup_is_charged_once_per_burst() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware = _base_knob_hardware(
        soc={
            "verify_setup": {"energy_pj_per_burst": 10.0, "latency_ns_per_burst": 100.0},
        }
    )

    _, b1 = estimate_point(model=model, hardware=hardware, stats=SpeculationStats(k=1, histogram={0: 1.0}), l_prompt=64)
    _, b4 = estimate_point(model=model, hardware=hardware, stats=SpeculationStats(k=4, histogram={0: 1.0}), l_prompt=64)

    assert b1.verify_bonus.stages.control_latency_ns == pytest.approx(b4.verify_bonus.stages.control_latency_ns)
    assert b1.verify_bonus.stages.control_latency_ns == pytest.approx(model.n_layers * 100.0)


def test_max_context_tokens_enforced_when_set() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=4, histogram={0: 1.0})

    hardware = _base_knob_hardware(
        memory={
            "kv_cache": {"max_context_tokens": 64},
        }
    )

    with pytest.raises(ValueError, match=r"max_context_tokens"):
        estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=61)


def test_max_context_tokens_not_enforced_when_unset() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=4, histogram={0: 1.0})

    hardware = _base_knob_hardware(
        memory={
            "kv_cache": {},
        }
    )

    estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=61)

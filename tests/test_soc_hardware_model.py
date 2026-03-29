from math import ceil

import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig
from selfspec_calculator.estimator import (
    _max_layer_compute_latencies_ns_knob,
    _memory_cost_from_traffic,
    estimate_point,
    estimate_sweep,
)
from selfspec_calculator.report import MemoryTraffic
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


def test_draft_buffers_add_latency_uses_stream_overlap() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})
    hardware = _base_knob_hardware(
        soc={"buffers_add": {"energy_pj_per_op": 0.01, "latency_ns_per_op": 1.0}},
    )

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=64)

    specs = hardware.resolve_knob_specs()
    num_slices = ceil(model.activation_bits / hardware.analog.dac_bits)
    tile_counts = {"qkv": 2, "wo": 1, "ffn": 4}
    total_stream_steps = sum(tile_count * num_slices * hardware.analog.num_columns_per_adc for tile_count in tile_counts.values())
    expected_per_layer = total_stream_steps * max(
        0.0,
        hardware.soc.buffers_add.latency_ns_per_op - specs.adc_draft.latency_ns_per_conversion,
    )
    expected = model.n_layers * expected_per_layer

    assert breakdown.draft.components is not None
    assert breakdown.draft.components.buffers_add_latency_ns == pytest.approx(expected)


def test_verify_reuse_buffers_add_exceeds_draft_when_final_add_is_needed() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})
    hardware = _base_knob_hardware(
        soc={"buffers_add": {"energy_pj_per_op": 0.01, "latency_ns_per_op": 1.0}},
    )

    _, breakdown = estimate_point(model=model, hardware=hardware, stats=stats, l_prompt=64)

    specs = hardware.resolve_knob_specs()
    num_slices = ceil(model.activation_bits / hardware.analog.dac_bits)
    tile_counts = {"qkv": 2, "wo": 1, "ffn": 4}
    total_stream_steps = sum(tile_count * num_slices * hardware.analog.num_columns_per_adc for tile_count in tile_counts.values())
    expected_per_layer = total_stream_steps * max(
        0.0,
        2.0 * hardware.soc.buffers_add.latency_ns_per_op - specs.adc_residual.latency_ns_per_conversion,
    )
    expected = model.n_layers * expected_per_layer

    assert breakdown.draft.components is not None
    assert breakdown.verify_drafted.components is not None
    assert breakdown.verify_drafted.components.buffers_add_latency_ns == pytest.approx(expected)
    assert breakdown.verify_drafted.components.buffers_add_latency_ns > breakdown.draft.components.buffers_add_latency_ns


def test_output_stream_periphery_latency_does_not_stack_when_adc_is_slower() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=0, histogram={0: 1.0})

    base_hw = _base_knob_hardware()
    periph_hw = _base_knob_hardware(
        analog={
            "xbar_size": 128,
            "num_columns_per_adc": 16,
            "dac_bits": 4,
            "adc": {"draft_bits": 4, "residual_bits": 12},
            "periphery": {
                "tia": {"energy_pj_per_op": 0.001, "latency_ns_per_op": 0.02},
                "snh": {"energy_pj_per_op": 0.001, "latency_ns_per_op": 0.01},
                "mux": {"energy_pj_per_op": 0.001, "latency_ns_per_op": 0.01},
            },
        },
    )

    _, base_breakdown = estimate_point(model=model, hardware=base_hw, stats=stats, l_prompt=64)
    _, periph_breakdown = estimate_point(model=model, hardware=periph_hw, stats=stats, l_prompt=64)

    assert periph_breakdown.verify_bonus.components is not None
    assert periph_breakdown.verify_bonus.components.tia_latency_ns > 0.0
    assert periph_breakdown.verify_bonus.components.snh_latency_ns > 0.0
    assert periph_breakdown.verify_bonus.components.mux_latency_ns > 0.0
    assert periph_breakdown.verify_bonus.latency_ns == pytest.approx(base_breakdown.verify_bonus.latency_ns)



def test_delta_readout_adds_extra_dac_cost_and_area() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    hw_base = _base_knob_hardware(
        analog={
            "xbar_size": 128,
            "num_columns_per_adc": 16,
            "dac_bits": 4,
            "adc": {"draft_bits": 4, "residual_bits": 12},
        }
    )
    hw_delta = _base_knob_hardware(
        analog={
            "xbar_size": 128,
            "num_columns_per_adc": 16,
            "dac_bits": 4,
            "adc": {"draft_bits": 4, "residual_bits": 12},
            "delta_readout": {
                "draft": {"enabled": True, "dac_bits": 8},
                "verify": {"enabled": True, "dac_bits": 8},
            },
        }
    )

    _, b0 = estimate_point(model=model, hardware=hw_base, stats=stats, l_prompt=64)
    _, b1 = estimate_point(model=model, hardware=hw_delta, stats=stats, l_prompt=64)

    assert b0.draft.components is not None
    assert b1.draft.components is not None
    assert b1.draft.components.dac_energy_pj > b0.draft.components.dac_energy_pj
    assert b1.verify_drafted.components.dac_energy_pj > b0.verify_drafted.components.dac_energy_pj

    r0 = estimate_sweep(model=model, hardware=hw_base, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    r1 = estimate_sweep(model=model, hardware=hw_delta, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    assert r1["area_breakdown_mm2"]["on_chip_components"]["dac_mm2"] > r0["area_breakdown_mm2"]["on_chip_components"]["dac_mm2"]

def test_memory_cost_uses_transfer_path_bottleneck_not_serial_sum() -> None:
    hardware = _base_knob_hardware(
        memory={
            "sram": {
                "read_energy_pj_per_byte": 0.1,
                "write_energy_pj_per_byte": 0.2,
                "read_bandwidth_GBps": 1000.0,
                "write_bandwidth_GBps": 1000.0,
                "read_latency_ns": 1.0,
                "write_latency_ns": 2.0,
            },
            "hbm": {
                "read_energy_pj_per_byte": 1.0,
                "write_energy_pj_per_byte": 2.0,
                "read_bandwidth_GBps": 1000.0,
                "write_bandwidth_GBps": 1000.0,
                "read_latency_ns": 10.0,
                "write_latency_ns": 20.0,
            },
            "fabric": {
                "read_energy_pj_per_byte": 0.01,
                "write_energy_pj_per_byte": 0.02,
                "read_bandwidth_GBps": 1000.0,
                "write_bandwidth_GBps": 1000.0,
                "read_latency_ns": 3.0,
                "write_latency_ns": 4.0,
            },
        }
    )
    traffic = MemoryTraffic(
        hbm_read_bytes=1000.0,
        hbm_write_bytes=500.0,
        fabric_read_bytes=1000.0,
        fabric_write_bytes=500.0,
    )

    energy, latency = _memory_cost_from_traffic(hardware=hardware, traffic=traffic)

    expected_energy = (1000.0 * 1.0) + (500.0 * 2.0) + (1000.0 * 0.01) + (500.0 * 0.02)
    expected_latency = max(1000.0 / 1000.0 + 10.0, 1000.0 / 1000.0 + 3.0) + max(
        500.0 / 1000.0 + 20.0,
        500.0 / 1000.0 + 4.0,
    )

    assert energy == pytest.approx(expected_energy)
    assert latency == pytest.approx(expected_latency)


def test_layer_pipelined_schedule_reduces_reported_latency() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hardware_serial = _base_knob_hardware(soc={"schedule": "serialized"})
    hardware_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})
    stats = SpeculationStats(k=2, histogram={0: 1.0})

    m0, _ = estimate_point(model=model, hardware=hardware_serial, stats=stats, l_prompt=64)
    m1, _ = estimate_point(model=model, hardware=hardware_pipe, stats=stats, l_prompt=64)

    assert m1.energy_pj_per_token <= m0.energy_pj_per_token
    assert m1.latency_ns_per_token <= m0.latency_ns_per_token


def test_zero_leakage_matches_dynamic_only_energy() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=3, histogram={0: 0.6, 3: 0.4})

    hw_base = _base_knob_hardware(soc={"schedule": "layer-pipelined"})
    hw_zero = _base_knob_hardware(
        soc={"schedule": "layer-pipelined"},
        leakage_power={
            "arrays_nw": 0.0,
            "attention_engine_nw": 0.0,
            "sram_nw": 0.0,
            "hbm_nw": 0.0,
            "fabric_nw": 0.0,
        },
    )

    m_base, _ = estimate_point(model=model, hardware=hw_base, stats=stats, l_prompt=64)
    m_zero, _ = estimate_point(model=model, hardware=hw_zero, stats=stats, l_prompt=64)

    assert m_zero.energy_pj_per_token == pytest.approx(m_base.energy_pj_per_token)
    assert m_zero.latency_ns_per_token == pytest.approx(m_base.latency_ns_per_token)


def test_positive_leakage_adds_expected_energy_delta() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=2, histogram={0: 1.0})
    total_leakage_nw = 7_500_000.0

    hw_base = _base_knob_hardware(soc={"schedule": "serialized"})
    hw_leak = _base_knob_hardware(
        soc={"schedule": "serialized"},
        leakage_power={
            "arrays_nw": 5_000_000.0,
            "control_nw": 2_500_000.0,
        },
    )

    m_base, _ = estimate_point(model=model, hardware=hw_base, stats=stats, l_prompt=64)
    m_leak, _ = estimate_point(model=model, hardware=hw_leak, stats=stats, l_prompt=64)

    expected_delta = total_leakage_nw * m_base.latency_ns_per_token * 1e-6
    assert m_leak.latency_ns_per_token == pytest.approx(m_base.latency_ns_per_token)
    assert m_leak.energy_pj_per_token - m_base.energy_pj_per_token == pytest.approx(expected_delta)


def test_leakage_energy_uses_schedule_aware_burst_latency() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=3, histogram={0: 1.0})
    total_leakage_nw = 4_000_000.0

    hw_serial = _base_knob_hardware(soc={"schedule": "serialized"})
    hw_serial_leak = _base_knob_hardware(
        soc={"schedule": "serialized"},
        leakage_power={"arrays_nw": total_leakage_nw},
    )
    hw_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})
    hw_pipe_leak = _base_knob_hardware(
        soc={"schedule": "layer-pipelined"},
        leakage_power={"arrays_nw": total_leakage_nw},
    )

    m_serial, _ = estimate_point(model=model, hardware=hw_serial, stats=stats, l_prompt=64)
    m_serial_leak, _ = estimate_point(model=model, hardware=hw_serial_leak, stats=stats, l_prompt=64)
    m_pipe, _ = estimate_point(model=model, hardware=hw_pipe, stats=stats, l_prompt=64)
    m_pipe_leak, _ = estimate_point(model=model, hardware=hw_pipe_leak, stats=stats, l_prompt=64)

    delta_serial = m_serial_leak.energy_pj_per_token - m_serial.energy_pj_per_token
    delta_pipe = m_pipe_leak.energy_pj_per_token - m_pipe.energy_pj_per_token

    assert delta_serial == pytest.approx(total_leakage_nw * m_serial.latency_ns_per_token * 1e-6)
    assert delta_pipe == pytest.approx(total_leakage_nw * m_pipe.latency_ns_per_token * 1e-6)
    assert m_pipe.latency_ns_per_token <= m_serial.latency_ns_per_token
    assert delta_pipe <= delta_serial


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

    assert m1.energy_pj_per_token <= m0.energy_pj_per_token
    assert m1.latency_ns_per_token > (m0.latency_ns_per_token / model.n_layers) * 2.0
    assert m1.latency_ns_per_token > 0.0


def test_layer_pipelined_k0_matches_stop_and_go_full_precision_baseline() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=0, histogram={0: 1.0})

    hw_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})
    hw_serial = _base_knob_hardware(soc={"schedule": "serialized"})

    m_pipe, b_pipe = estimate_point(model=model, hardware=hw_pipe, stats=stats, l_prompt=64)
    m_serial, b_serial = estimate_point(model=model, hardware=hw_serial, stats=stats, l_prompt=64)

    assert m_pipe.latency_ns_per_token == pytest.approx(m_serial.latency_ns_per_token)
    assert m_pipe.energy_pj_per_token == pytest.approx(m_serial.energy_pj_per_token)
    assert b_pipe.verify_bonus.energy_pj == pytest.approx(b_serial.verify_bonus.energy_pj)
    assert b_pipe.verify_bonus.latency_ns == pytest.approx(b_serial.verify_bonus.latency_ns)


def test_layer_pipelined_verify_burst_latency_is_fixed_across_acceptance_outcomes() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    hw_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})

    k = 5
    m0, _ = estimate_point(model=model, hardware=hw_pipe, stats=SpeculationStats(k=k, histogram={0: 1.0}), l_prompt=64)
    m5, _ = estimate_point(model=model, hardware=hw_pipe, stats=SpeculationStats(k=k, histogram={k: 1.0}), l_prompt=64)

    burst0 = m0.latency_ns_per_token * 1.0
    burst5 = m5.latency_ns_per_token * float(k + 1)

    assert burst0 == pytest.approx(burst5)


def test_layer_pipelined_verify_work_is_fixed_but_commit_writes_follow_acceptance() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    d_model = model.d_model
    n_heads = model.n_heads
    n_layers = model.n_layers

    fmt = {"value_bytes_per_elem": 1, "scale_bytes": 2, "scales_per_token_per_head": 2}
    bytes_per_token_per_layer = 2 * d_model * fmt["value_bytes_per_elem"] + n_heads * fmt["scales_per_token_per_head"] * fmt["scale_bytes"]
    k = 4

    hw_pipe = _base_knob_hardware(
        soc={"schedule": "layer-pipelined"},
        memory={
            "sram": {"read_energy_pj_per_byte": 0.1, "write_energy_pj_per_byte": 0.2},
            "hbm": {"read_energy_pj_per_byte": 1.0, "write_energy_pj_per_byte": 2.0},
            "fabric": {"read_energy_pj_per_byte": 0.01, "write_energy_pj_per_byte": 0.01},
            "kv_cache": {"hbm": fmt},
        },
    )
    _, b_mismatch = estimate_point(model=model, hardware=hw_pipe, stats=SpeculationStats(k=k, histogram={0: 1.0}), l_prompt=64)
    _, b_accept = estimate_point(model=model, hardware=hw_pipe, stats=SpeculationStats(k=k, histogram={k: 1.0}), l_prompt=64)

    assert b_mismatch.verify_drafted.energy_pj == pytest.approx(b_accept.verify_drafted.energy_pj)
    assert b_mismatch.verify_drafted.latency_ns == pytest.approx(b_accept.verify_drafted.latency_ns)
    assert b_mismatch.verify_bonus.stages.qkv_energy_pj == pytest.approx(b_accept.verify_bonus.stages.qkv_energy_pj)
    assert b_mismatch.verify_bonus.stages.qk_energy_pj == pytest.approx(b_accept.verify_bonus.stages.qk_energy_pj)

    assert b_mismatch.verify_bonus.memory_traffic is not None
    assert b_accept.verify_bonus.memory_traffic is not None
    expected_mismatch_write = 1.0 * n_layers * bytes_per_token_per_layer
    expected_accept_write = float(k + 1) * n_layers * bytes_per_token_per_layer
    assert b_mismatch.verify_bonus.memory_traffic.hbm_write_bytes == pytest.approx(expected_mismatch_write)
    assert b_accept.verify_bonus.memory_traffic.hbm_write_bytes == pytest.approx(expected_accept_write)


def test_layer_pipelined_verify_burst_includes_first_token_fill_latency() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    k = 3
    l_prompt = 64
    hw_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})
    hw_serial = _base_knob_hardware(soc={"schedule": "serialized"})

    m_pipe, b_pipe = estimate_point(model=model, hardware=hw_pipe, stats=SpeculationStats(k=k, histogram={k: 1.0}), l_prompt=l_prompt)
    _, b_first = estimate_point(
        model=model,
        hardware=hw_serial,
        stats=SpeculationStats(k=1, histogram={0: 1.0}),
        l_prompt=l_prompt,
    )

    specs = hw_pipe.resolve_knob_specs()
    verify_periods = []
    for i in range(k):
        _draft, verify_drafted, _bonus = _max_layer_compute_latencies_ns_knob(
            model=model,
            hardware=hw_pipe,
            specs=specs,
            l_prompt=l_prompt + i,
        )
        verify_periods.append(verify_drafted)

    _draft, _verify_drafted, verify_bonus_period = _max_layer_compute_latencies_ns_knob(
        model=model,
        hardware=hw_pipe,
        specs=specs,
        l_prompt=l_prompt + k,
    )

    expected_burst = (
        b_pipe.draft.latency_ns
        + b_first.verify_drafted.latency_ns
        + sum(verify_periods[1:])
        + verify_bonus_period
    )
    assert m_pipe.latency_ns_per_token * float(k + 1) == pytest.approx(expected_burst)


def test_layer_pipelined_keeps_draft_phase_serialized() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=4, histogram={0: 0.4, 2: 0.3, 4: 0.3})
    hw_serial = _base_knob_hardware(soc={"schedule": "serialized"})
    hw_pipe = _base_knob_hardware(soc={"schedule": "layer-pipelined"})

    _, b_serial = estimate_point(model=model, hardware=hw_serial, stats=stats, l_prompt=64)
    _, b_pipe = estimate_point(model=model, hardware=hw_pipe, stats=stats, l_prompt=64)

    assert b_pipe.draft.energy_pj == pytest.approx(b_serial.draft.energy_pj)
    assert b_pipe.draft.latency_ns == pytest.approx(b_serial.draft.latency_ns)


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


def test_attention_cim_units_scale_qk_pv_latency_and_area() -> None:
    model = ModelConfig.model_validate(BASE_MODEL)
    stats = SpeculationStats(k=1, histogram={0: 1.0})

    # Attention SRAM-CIM unit split:
    # - SRAM storage area from CACTI-style memory area/capacity ratio
    # - MAC logic area from a separate per-unit knob (to be replaced by DC later)
    hw1 = _base_knob_hardware(
        soc={"attention_cim_units": 1, "attention_cim_mac_area_mm2_per_unit": 0.25},
        memory={"sram": {"area_mm2": 2.0, "capacity_bytes": 131072}},
    )
    hw8 = _base_knob_hardware(
        soc={"attention_cim_units": 8, "attention_cim_mac_area_mm2_per_unit": 0.25},
        memory={"sram": {"area_mm2": 2.0, "capacity_bytes": 131072}},
    )

    _, b1 = estimate_point(model=model, hardware=hw1, stats=stats, l_prompt=64)
    _, b8 = estimate_point(model=model, hardware=hw8, stats=stats, l_prompt=64)

    # QK/PV are mapped to SRAM-CIM attention units: latency scales with unit count,
    # while energy remains per-MAC.
    assert b8.draft.stages.qk_latency_ns == pytest.approx(b1.draft.stages.qk_latency_ns / 8.0)
    assert b8.draft.stages.pv_latency_ns == pytest.approx(b1.draft.stages.pv_latency_ns / 8.0)
    assert b8.draft.stages.qk_energy_pj == pytest.approx(b1.draft.stages.qk_energy_pj)
    assert b8.draft.stages.pv_energy_pj == pytest.approx(b1.draft.stages.pv_energy_pj)

    assert hw1.analog is not None
    # activation_bits=12 -> 2 bytes/element, logical array = 128x128 elements
    unit_sram_bytes = float(hw1.analog.xbar_size * hw1.analog.xbar_size * 2)
    unit_sram_area = unit_sram_bytes * (2.0 / 131072.0)
    unit_mac_area = 0.25
    expected_delta_sram = float(model.n_layers) * (8.0 - 1.0) * unit_sram_area
    expected_delta_mac = float(model.n_layers) * (8.0 - 1.0) * unit_mac_area

    r1 = estimate_sweep(model=model, hardware=hw1, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    r8 = estimate_sweep(model=model, hardware=hw8, stats=stats, prompt_lengths=[64]).model_dump(mode="json")
    c1 = r1["area_breakdown_mm2"]["on_chip_components"]
    c8 = r8["area_breakdown_mm2"]["on_chip_components"]
    assert c8["attention_cim_sram_mm2"] - c1["attention_cim_sram_mm2"] == pytest.approx(expected_delta_sram)
    assert c8["attention_cim_mac_mm2"] - c1["attention_cim_mac_mm2"] == pytest.approx(expected_delta_mac)
    # Base digital overhead remains separate from attention-CIM split components.
    assert c8["digital_overhead_mm2"] == pytest.approx(c1["digital_overhead_mm2"])


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

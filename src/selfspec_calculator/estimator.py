from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from typing import Any

from .config import (
    HardwareConfig,
    HardwareMode,
    InputPaths,
    ModelConfig,
    PrecisionMode,
    ReusePolicy,
    ResolvedKnobSpecs,
    ScheduleMode,
)
from .report import (
    AnalogActivationCounts,
    AreaBreakdownMm2,
    AreaComponentsMm2,
    BaselineDelta,
    Breakdown,
    ComponentBreakdown,
    MemoryTraffic,
    Metrics,
    PhaseBreakdown,
    Report,
    StageBreakdown,
    SweepPoint,
)
from .stats import SpeculationStats, expected_committed_tokens_per_burst


ANALOG_STAGES = ("qkv", "wo", "ffn")
DIGITAL_STAGES = ("qk", "pv", "softmax", "elementwise", "kv_cache")


def _kv_bytes_per_token_per_layer(*, d_model: int, n_heads: int, fmt) -> int:  # noqa: ANN001
    payload_bytes = 2 * d_model * int(fmt.value_bytes_per_elem)
    metadata_bytes = n_heads * int(fmt.scales_per_token_per_head) * int(fmt.scale_bytes)
    return payload_bytes + metadata_bytes


def _mem_energy_latency(*, tech, read_bytes: float, write_bytes: float) -> tuple[float, float]:  # noqa: ANN001
    energy = read_bytes * tech.read_energy_pj_per_byte + write_bytes * tech.write_energy_pj_per_byte

    latency = 0.0
    if read_bytes > 0:
        if tech.read_bandwidth_GBps > 0:
            latency += read_bytes / tech.read_bandwidth_GBps
        latency += tech.read_latency_ns
    if write_bytes > 0:
        if tech.write_bandwidth_GBps > 0:
            latency += write_bytes / tech.write_bandwidth_GBps
        latency += tech.write_latency_ns

    return energy, latency


def _kv_memory_traffic_by_phase(
    *,
    model: ModelConfig,
    hardware: HardwareConfig,
    stats: SpeculationStats,
    l_prompt: int,
) -> dict[str, MemoryTraffic]:
    if hardware.memory is None:
        z = MemoryTraffic()
        return {"draft": z, "verify_drafted": z, "verify_bonus": z}

    k = stats.k
    n_layers = model.n_layers
    d_model = model.d_model
    n_heads = model.n_heads

    fmt_hbm = hardware.memory.kv_cache.hbm
    fmt_sram = hardware.memory.kv_cache.resolved_sram()

    bytes_hbm_token = _kv_bytes_per_token_per_layer(d_model=d_model, n_heads=n_heads, fmt=fmt_hbm)
    bytes_sram_token = _kv_bytes_per_token_per_layer(d_model=d_model, n_heads=n_heads, fmt=fmt_sram)

    def tokens_to_bytes(tokens: float, bytes_per_token: int) -> float:
        return float(tokens) * float(n_layers) * float(bytes_per_token)

    # HBM reads: base context (prompt) is always served from HBM; no mismatch gating in v1.
    hbm_read_per_step_bytes = tokens_to_bytes(l_prompt, bytes_hbm_token)
    draft_hbm_read = float(k) * hbm_read_per_step_bytes
    verify_drafted_hbm_read = float(k) * hbm_read_per_step_bytes
    verify_bonus_hbm_read = 1.0 * hbm_read_per_step_bytes

    # SRAM traffic: speculative within-burst KV buffer.
    if k <= 0:
        draft_sram_read = 0.0
        verify_drafted_sram_read = 0.0
        verify_bonus_sram_read = 0.0
        draft_sram_write = 0.0
        verify_drafted_sram_write = 0.0
        verify_bonus_sram_write = 0.0
    else:
        draft_sram_read = tokens_to_bytes(k * (k - 1) / 2.0, bytes_sram_token)
        verify_drafted_sram_read = tokens_to_bytes(k * (k - 1) / 2.0, bytes_sram_token)
        verify_bonus_sram_read = tokens_to_bytes(float(k), bytes_sram_token)

        draft_sram_write = tokens_to_bytes(float(k), bytes_sram_token)
        verify_drafted_sram_write = tokens_to_bytes(float(k), bytes_sram_token)
        verify_bonus_sram_write = tokens_to_bytes(1.0, bytes_sram_token)

    # HBM writes: commit-only (policy B).
    committed_tokens = expected_committed_tokens_per_burst(stats)
    hbm_write = tokens_to_bytes(committed_tokens, bytes_hbm_token)

    draft = MemoryTraffic(
        sram_read_bytes=draft_sram_read,
        sram_write_bytes=draft_sram_write,
        hbm_read_bytes=draft_hbm_read,
        hbm_write_bytes=0.0,
    )
    verify_drafted = MemoryTraffic(
        sram_read_bytes=verify_drafted_sram_read,
        sram_write_bytes=verify_drafted_sram_write,
        hbm_read_bytes=verify_drafted_hbm_read,
        hbm_write_bytes=0.0,
    )
    verify_bonus = MemoryTraffic(
        sram_read_bytes=verify_bonus_sram_read,
        sram_write_bytes=verify_bonus_sram_write,
        hbm_read_bytes=verify_bonus_hbm_read,
        hbm_write_bytes=hbm_write,
    )

    # Fabric: model as bytes moved for both SRAM and HBM traffic (conservative, configurable by energy/byte and BW).
    for traffic in (draft, verify_drafted, verify_bonus):
        traffic.fabric_read_bytes = traffic.sram_read_bytes + traffic.hbm_read_bytes
        traffic.fabric_write_bytes = traffic.sram_write_bytes + traffic.hbm_write_bytes

    return {"draft": draft, "verify_drafted": verify_drafted, "verify_bonus": verify_bonus}


def _add_memory_traffic_costs(
    *,
    breakdown: Breakdown,
    traffic: MemoryTraffic,
    hardware: HardwareConfig,
) -> Breakdown:
    if hardware.memory is None:
        return breakdown

    sram_e, sram_t = _mem_energy_latency(
        tech=hardware.memory.sram, read_bytes=traffic.sram_read_bytes, write_bytes=traffic.sram_write_bytes
    )
    hbm_e, hbm_t = _mem_energy_latency(
        tech=hardware.memory.hbm, read_bytes=traffic.hbm_read_bytes, write_bytes=traffic.hbm_write_bytes
    )
    fabric_e, fabric_t = _mem_energy_latency(
        tech=hardware.memory.fabric,
        read_bytes=traffic.fabric_read_bytes,
        write_bytes=traffic.fabric_write_bytes,
    )

    mem_energy = sram_e + hbm_e + fabric_e
    mem_latency = sram_t + hbm_t + fabric_t

    stages = breakdown.stages.add_energy_latency("kv_cache", mem_energy, mem_latency)

    components = breakdown.components or ComponentBreakdown()
    components = components.add_energy_latency("sram", sram_e, sram_t)
    components = components.add_energy_latency("hbm", hbm_e, hbm_t)
    components = components.add_energy_latency("fabric", fabric_e, fabric_t)

    return Breakdown.from_stage_breakdown(
        stages,
        components=components,
        activation_counts=breakdown.activation_counts,
        memory_traffic=traffic,
    )


def _mac_counts_per_token(model: ModelConfig, l_prompt: int) -> dict[str, int]:
    d_model = model.d_model
    d_ff = model.effective_d_ff
    d_head = model.d_head
    n_heads = model.n_heads

    qkv_macs = 3 * d_model * d_model
    wo_macs = d_model * d_model
    if model.ffn_type.value == "mlp":
        ffn_macs = 2 * d_model * d_ff
    else:
        ffn_macs = 3 * d_model * d_ff

    qk_macs = n_heads * l_prompt * d_head
    pv_macs = n_heads * l_prompt * d_head
    softmax_ops = n_heads * l_prompt
    elementwise_ops = d_ff
    kv_cache_ops = d_model

    return {
        "qkv": qkv_macs,
        "wo": wo_macs,
        "ffn": ffn_macs,
        "qk": qk_macs,
        "pv": pv_macs,
        "softmax": softmax_ops,
        "elementwise": elementwise_ops,
        "kv_cache": kv_cache_ops,
    }


def _weights_per_layer(model: ModelConfig) -> dict[str, int]:
    d_model = model.d_model
    d_ff = model.effective_d_ff

    qkv_weights = 3 * d_model * d_model
    wo_weights = d_model * d_model
    if model.ffn_type.value == "mlp":
        ffn_weights = 2 * d_model * d_ff
    else:
        ffn_weights = 3 * d_model * d_ff
    return {"qkv": qkv_weights, "wo": wo_weights, "ffn": ffn_weights}


def _analog_cost_for_block(hardware: HardwareConfig, precision: PrecisionMode) -> tuple[float, float]:
    assert hardware.costs is not None
    if precision == PrecisionMode.full:
        c = hardware.costs.analog_full
    else:
        c = hardware.costs.analog_draft
    return (c.energy_pj_per_mac, c.latency_ns_per_mac)


def _verify_additional_cost_for_block(
    hardware: HardwareConfig,
    executed_precision_in_draft: PrecisionMode,
    token_kind: str,
) -> tuple[float, float]:
    assert hardware.costs is not None
    if token_kind == "bonus":
        c = hardware.costs.analog_full
        return (c.energy_pj_per_mac, c.latency_ns_per_mac)

    if hardware.reuse_policy == ReusePolicy.reread:
        c = hardware.costs.analog_full
        return (c.energy_pj_per_mac, c.latency_ns_per_mac)

    if executed_precision_in_draft == PrecisionMode.full:
        return (0.0, 0.0)

    c = hardware.costs.analog_verify_reuse
    return (c.energy_pj_per_mac, c.latency_ns_per_mac)


def _digital_costs_legacy(hardware: HardwareConfig) -> dict[str, tuple[float, float]]:
    assert hardware.costs is not None
    return {
        "qk": (hardware.costs.digital_attention.energy_pj_per_mac, hardware.costs.digital_attention.latency_ns_per_mac),
        "pv": (hardware.costs.digital_attention.energy_pj_per_mac, hardware.costs.digital_attention.latency_ns_per_mac),
        "softmax": (hardware.costs.digital_softmax.energy_pj_per_mac, hardware.costs.digital_softmax.latency_ns_per_mac),
        "elementwise": (
            hardware.costs.digital_elementwise.energy_pj_per_mac,
            hardware.costs.digital_elementwise.latency_ns_per_mac,
        ),
        "kv_cache": (hardware.costs.kv_cache.energy_pj_per_mac, hardware.costs.kv_cache.latency_ns_per_mac),
    }


def _digital_costs_knob(specs: ResolvedKnobSpecs) -> dict[str, tuple[float, float]]:
    return {
        "qk": (specs.digital.attention.energy_pj_per_mac, specs.digital.attention.latency_ns_per_mac),
        "pv": (specs.digital.attention.energy_pj_per_mac, specs.digital.attention.latency_ns_per_mac),
        "softmax": (specs.digital.softmax.energy_pj_per_mac, specs.digital.softmax.latency_ns_per_mac),
        "elementwise": (specs.digital.elementwise.energy_pj_per_mac, specs.digital.elementwise.latency_ns_per_mac),
        "kv_cache": (specs.digital.kv_cache.energy_pj_per_mac, specs.digital.kv_cache.latency_ns_per_mac),
    }


def _legacy_components_from_stages(stages: StageBreakdown) -> ComponentBreakdown:
    analog_energy = stages.qkv_energy_pj + stages.wo_energy_pj + stages.ffn_energy_pj
    analog_latency = stages.qkv_latency_ns + stages.wo_latency_ns + stages.ffn_latency_ns
    return ComponentBreakdown(
        arrays_energy_pj=analog_energy,
        arrays_latency_ns=analog_latency,
        attention_engine_energy_pj=stages.qk_energy_pj + stages.pv_energy_pj,
        attention_engine_latency_ns=stages.qk_latency_ns + stages.pv_latency_ns,
        kv_cache_energy_pj=stages.kv_cache_energy_pj,
        kv_cache_latency_ns=stages.kv_cache_latency_ns,
        softmax_unit_energy_pj=stages.softmax_energy_pj,
        softmax_unit_latency_ns=stages.softmax_latency_ns,
        elementwise_unit_energy_pj=stages.elementwise_energy_pj,
        elementwise_unit_latency_ns=stages.elementwise_latency_ns,
        buffers_add_energy_pj=stages.buffers_add_energy_pj,
        buffers_add_latency_ns=stages.buffers_add_latency_ns,
        control_energy_pj=stages.control_energy_pj,
        control_latency_ns=stages.control_latency_ns,
    )


def _attention_cim_unit_area_mm2(*, specs: ResolvedKnobSpecs, xbar_size: int) -> float:
    if specs.array.area_mm2_per_array is not None and specs.array.area_mm2_per_array > 0.0:
        return float(specs.array.area_mm2_per_array)
    # Backward-compatible fallback for libraries that only provide per-weight area.
    return float(specs.array.area_mm2_per_weight) * float(xbar_size * xbar_size)


def _area_mm2(model: ModelConfig, hardware: HardwareConfig) -> StageBreakdown:
    if hardware.mode == HardwareMode.legacy:
        weights = _weights_per_layer(model)
        assert hardware.costs is not None
        analog_area_per_weight = hardware.costs.analog_weight_area.area_mm2_per_weight
        qkv = weights["qkv"] * analog_area_per_weight
        wo = weights["wo"] * analog_area_per_weight
        ffn = weights["ffn"] * analog_area_per_weight
        digital = hardware.costs.digital_overhead_area_mm2_per_layer
    else:
        assert hardware.analog is not None
        specs = hardware.resolve_knob_specs()
        arrays_per_weight = float(specs.array.arrays_per_weight)
        # Preferred path: compute required array count from model shapes/xbar size
        # and multiply by physical array area when the library provides it.
        if specs.array.area_mm2_per_array is not None and specs.array.area_mm2_per_array > 0.0:
            tiles = _tile_counts(model, hardware.analog.xbar_size)
            qkv = float(tiles["qkv"]) * arrays_per_weight * specs.array.area_mm2_per_array
            wo = float(tiles["wo"]) * arrays_per_weight * specs.array.area_mm2_per_array
            ffn = float(tiles["ffn"]) * arrays_per_weight * specs.array.area_mm2_per_array
        else:
            weights = _weights_per_layer(model)
            qkv = weights["qkv"] * specs.array.area_mm2_per_weight
            wo = weights["wo"] * specs.array.area_mm2_per_weight
            ffn = weights["ffn"] * specs.array.area_mm2_per_weight
        attention_cim_area_per_layer = (
            float(hardware.soc.attention_cim_units)
            * _attention_cim_unit_area_mm2(specs=specs, xbar_size=hardware.analog.xbar_size)
        )
        digital = specs.digital.digital_overhead_area_mm2_per_layer + attention_cim_area_per_layer

    scale = model.n_layers
    return StageBreakdown(
        qkv_mm2=qkv * scale,
        wo_mm2=wo * scale,
        ffn_mm2=ffn * scale,
        digital_mm2=digital * scale,
    )


def _area_breakdown_mm2(model: ModelConfig, hardware: HardwareConfig) -> AreaBreakdownMm2:
    stage_area = _area_mm2(model, hardware)
    arrays_mm2 = stage_area.qkv_mm2 + stage_area.wo_mm2 + stage_area.ffn_mm2
    digital_overhead_mm2 = stage_area.digital_mm2

    dac_mm2 = 0.0
    adc_draft_mm2 = 0.0
    adc_residual_mm2 = 0.0

    tia_mm2 = 0.0
    snh_mm2 = 0.0
    mux_mm2 = 0.0
    io_buffers_mm2 = 0.0
    subarray_switches_mm2 = 0.0
    write_drivers_mm2 = 0.0

    if hardware.mode == HardwareMode.knob_based and hardware.analog is not None:
        specs = hardware.resolve_knob_specs()
        num_tiles_per_layer = _tile_counts(model, hardware.analog.xbar_size)
        tiles_total_logical = float(model.n_layers) * float(sum(num_tiles_per_layer.values()))
        # Physical arrays scale with replication (arrays_per_weight), while shared readout/input
        # interfaces (DAC + ADC paths) follow logical array-group count under the 1+3 split.
        tiles_total_physical = tiles_total_logical * float(specs.array.arrays_per_weight)

        dac_units = tiles_total_logical * float(hardware.analog.xbar_size)
        adc_units_per_path = tiles_total_logical * float(hardware.analog.xbar_size // hardware.analog.num_columns_per_adc)

        dac_mm2 = dac_units * specs.dac.area_mm2_per_unit
        adc_draft_mm2 = adc_units_per_path * specs.adc_draft.area_mm2_per_unit
        adc_residual_mm2 = adc_units_per_path * specs.adc_residual.area_mm2_per_unit

        periph = hardware.analog.periphery
        adc_total_units = 2.0 * adc_units_per_path
        tia_mm2 = adc_total_units * periph.tia.area_mm2_per_unit
        snh_mm2 = adc_total_units * periph.snh.area_mm2_per_unit
        mux_mm2 = adc_total_units * periph.mux.area_mm2_per_unit
        io_buffers_mm2 = adc_total_units * periph.io_buffers.area_mm2_per_unit
        subarray_switches_mm2 = tiles_total_physical * periph.subarray_switches.area_mm2_per_unit
        write_drivers_mm2 = dac_units * periph.write_drivers.area_mm2_per_unit

    sram_mm2 = 0.0
    fabric_mm2 = 0.0
    off_chip_hbm_mm2 = 0.0
    if hardware.memory is not None:
        sram_mm2 = hardware.memory.sram.area_mm2
        fabric_mm2 = hardware.memory.fabric.area_mm2
        off_chip_hbm_mm2 = hardware.memory.hbm.area_mm2

    on_chip_components = AreaComponentsMm2(
        arrays_mm2=arrays_mm2,
        dac_mm2=dac_mm2,
        adc_draft_mm2=adc_draft_mm2,
        adc_residual_mm2=adc_residual_mm2,
        tia_mm2=tia_mm2,
        snh_mm2=snh_mm2,
        mux_mm2=mux_mm2,
        io_buffers_mm2=io_buffers_mm2,
        subarray_switches_mm2=subarray_switches_mm2,
        write_drivers_mm2=write_drivers_mm2,
        sram_mm2=sram_mm2,
        fabric_mm2=fabric_mm2,
        digital_overhead_mm2=digital_overhead_mm2,
    )
    on_chip_mm2 = sum(getattr(on_chip_components, f) for f in type(on_chip_components).model_fields)

    return AreaBreakdownMm2(
        on_chip_mm2=on_chip_mm2,
        off_chip_hbm_mm2=off_chip_hbm_mm2,
        on_chip_components=on_chip_components,
    )


def _token_step_costs_legacy(model: ModelConfig, hardware: HardwareConfig, l_prompt: int) -> tuple[Breakdown, Breakdown]:
    macs = _mac_counts_per_token(model, l_prompt)
    digital_costs = _digital_costs_legacy(hardware)
    digital_stages = DIGITAL_STAGES if hardware.memory is None else tuple(s for s in DIGITAL_STAGES if s != "kv_cache")
    analog_outputs = {s: sum(m_out for m_out, _n_in in shapes) for s, shapes in _analog_stage_shapes(model).items()}
    buf_knobs = hardware.soc.buffers_add

    def stage_energy_latency(stage: str, energy_per_mac: float, latency_per_mac: float) -> tuple[float, float]:
        m = macs[stage]
        return (m * energy_per_mac, m * latency_per_mac)

    draft_stage = StageBreakdown()
    verify_full_stage = StageBreakdown()

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)
        for block, precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            e_per, t_per = _analog_cost_for_block(hardware, precision)
            e, t = stage_energy_latency(block, e_per, t_per)
            draft_stage = draft_stage.add_energy_latency(block, e, t)

            assert hardware.costs is not None
            e_full, t_full = stage_energy_latency(
                block,
                hardware.costs.analog_full.energy_pj_per_mac,
                hardware.costs.analog_full.latency_ns_per_mac,
            )
            verify_full_stage = verify_full_stage.add_energy_latency(block, e_full, t_full)

            outputs = float(analog_outputs[block])
            if hardware.reuse_policy == ReusePolicy.reuse:
                draft_stage = draft_stage.add_energy_latency(
                    "buffers_add",
                    outputs * buf_knobs.energy_pj_per_op,
                    outputs * buf_knobs.latency_ns_per_op,
                )
            if precision == PrecisionMode.full:
                draft_stage = draft_stage.add_energy_latency(
                    "buffers_add",
                    outputs * buf_knobs.energy_pj_per_op,
                    outputs * buf_knobs.latency_ns_per_op,
                )
            verify_full_stage = verify_full_stage.add_energy_latency(
                "buffers_add",
                outputs * buf_knobs.energy_pj_per_op,
                outputs * buf_knobs.latency_ns_per_op,
            )

        for stage in digital_stages:
            e_per, t_per = digital_costs[stage]
            e, t = stage_energy_latency(stage, e_per, t_per)
            draft_stage = draft_stage.add_energy_latency(stage, e, t)
            verify_full_stage = verify_full_stage.add_energy_latency(stage, e, t)

    ctrl_e_tok = model.n_layers * hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = model.n_layers * hardware.soc.control.latency_ns_per_token
    draft_stage = draft_stage.add_energy_latency("control", ctrl_e_tok, ctrl_t_tok)
    verify_full_stage = verify_full_stage.add_energy_latency("control", ctrl_e_tok, ctrl_t_tok)

    ctrl_e_burst = model.n_layers * hardware.soc.control.energy_pj_per_burst
    ctrl_t_burst = model.n_layers * hardware.soc.control.latency_ns_per_burst
    setup_e_burst = model.n_layers * hardware.soc.verify_setup.energy_pj_per_burst
    setup_t_burst = model.n_layers * hardware.soc.verify_setup.latency_ns_per_burst
    verify_full_stage = verify_full_stage.add_energy_latency(
        "control",
        ctrl_e_burst + setup_e_burst,
        ctrl_t_burst + setup_t_burst,
    )

    return (
        Breakdown.from_stage_breakdown(draft_stage, components=_legacy_components_from_stages(draft_stage)),
        Breakdown.from_stage_breakdown(
            verify_full_stage,
            components=_legacy_components_from_stages(verify_full_stage),
        ),
    )


def _verify_drafted_token_additional_stage_legacy(
    model: ModelConfig, hardware: HardwareConfig, l_prompt: int
) -> Breakdown:
    macs = _mac_counts_per_token(model, l_prompt)
    digital_costs = _digital_costs_legacy(hardware)
    digital_stages = DIGITAL_STAGES if hardware.memory is None else tuple(s for s in DIGITAL_STAGES if s != "kv_cache")
    analog_outputs = {s: sum(m_out for m_out, _n_in in shapes) for s, shapes in _analog_stage_shapes(model).items()}
    buf_knobs = hardware.soc.buffers_add

    additional = StageBreakdown()

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)
        for block, executed_precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            e_per, t_per = _verify_additional_cost_for_block(hardware, executed_precision, token_kind="drafted")
            additional = additional.add_energy_latency(block, macs[block] * e_per, macs[block] * t_per)

            outputs = float(analog_outputs[block])
            if hardware.reuse_policy == ReusePolicy.reread:
                additional = additional.add_energy_latency(
                    "buffers_add",
                    outputs * buf_knobs.energy_pj_per_op,
                    outputs * buf_knobs.latency_ns_per_op,
                )
            else:
                if executed_precision == PrecisionMode.full:
                    additional = additional.add_energy_latency(
                        "buffers_add",
                        outputs * buf_knobs.energy_pj_per_op,
                        outputs * buf_knobs.latency_ns_per_op,
                    )
                else:
                    additional = additional.add_energy_latency(
                        "buffers_add",
                        2.0 * outputs * buf_knobs.energy_pj_per_op,
                        2.0 * outputs * buf_knobs.latency_ns_per_op,
                    )

        for stage in digital_stages:
            e_per, t_per = digital_costs[stage]
            additional = additional.add_energy_latency(stage, macs[stage] * e_per, macs[stage] * t_per)

    ctrl_e_tok = model.n_layers * hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = model.n_layers * hardware.soc.control.latency_ns_per_token
    additional = additional.add_energy_latency("control", ctrl_e_tok, ctrl_t_tok)

    return Breakdown.from_stage_breakdown(additional, components=_legacy_components_from_stages(additional))


def _analog_stage_shapes(model: ModelConfig) -> dict[str, list[tuple[int, int]]]:
    d_model = model.d_model
    d_ff = model.effective_d_ff
    if model.ffn_type.value == "mlp":
        ffn_shapes = [(d_ff, d_model), (d_model, d_ff)]
    else:
        ffn_shapes = [(d_ff, d_model), (d_ff, d_model), (d_model, d_ff)]
    return {
        "qkv": [(3 * d_model, d_model)],
        "wo": [(d_model, d_model)],
        "ffn": ffn_shapes,
    }


def _tile_counts(model: ModelConfig, xbar_size: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for stage, shapes in _analog_stage_shapes(model).items():
        total = 0
        for m_out, n_in in shapes:
            tiles_out = ceil(m_out / xbar_size)
            tiles_in = ceil(n_in / xbar_size)
            total += tiles_out * tiles_in
        out[stage] = total
    return out


def _parallel_latency_split(lat_a: float, lat_b: float) -> tuple[float, float, float]:
    total = max(lat_a, lat_b)
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    if lat_a <= 0.0:
        return (0.0, total, total)
    if lat_b <= 0.0:
        return (total, 0.0, total)
    denom = lat_a + lat_b
    return (total * lat_a / denom, total * lat_b / denom, total)


def _analog_mode(
    mode_name: str,
) -> tuple[int, bool, bool]:
    if mode_name == "draft_default":
        return (1, True, False)
    if mode_name == "draft_full":
        return (4, True, True)
    if mode_name == "verify_residual_only":
        return (3, False, True)
    if mode_name in {"verify_full", "verify_bonus"}:
        return (4, True, True)
    if mode_name == "none":
        return (0, False, False)
    raise ValueError(f"Unsupported analog mode: {mode_name}")


class _TokenAccumulator:
    def __init__(self) -> None:
        self.stages = StageBreakdown()
        self.components = ComponentBreakdown()
        self.activation_counts = AnalogActivationCounts()

    def add_stage(self, stage: str, energy_pj: float, latency_ns: float) -> None:
        self.stages = self.stages.add_energy_latency(stage, energy_pj, latency_ns)

    def add_component(self, component: str, energy_pj: float, latency_ns: float) -> None:
        self.components = self.components.add_energy_latency(component, energy_pj, latency_ns)

    def add_analog_counts(
        self,
        *,
        array_activations: float,
        dac_conversions: float,
        adc_draft_conversions: float,
        adc_residual_conversions: float,
    ) -> None:
        self.activation_counts = self.activation_counts.plus(
            AnalogActivationCounts(
                array_activations=array_activations,
                dac_conversions=dac_conversions,
                adc_draft_conversions=adc_draft_conversions,
                adc_residual_conversions=adc_residual_conversions,
            )
        )

    def to_breakdown(self) -> Breakdown:
        return Breakdown.from_stage_breakdown(
            self.stages,
            components=self.components,
            activation_counts=self.activation_counts,
        )


def _add_knob_analog_stage(
    *,
    acc: _TokenAccumulator,
    stage: str,
    num_tiles: int,
    num_slices: int,
    xbar_size: int,
    adc_steps: int,
    specs: ResolvedKnobSpecs,
    periphery: Any,  # AnalogPeripheryKnobs
    mode_name: str,
) -> None:
    active_arrays, use_adc_draft, use_adc_residual = _analog_mode(mode_name)
    if active_arrays == 0:
        return

    base_reads = float(num_tiles * num_slices)
    array_activations = base_reads * active_arrays
    # Shared-DAC model: one conversion stream per logical input-column group,
    # broadcast to all active residual subarrays.
    dac_conversions = base_reads * xbar_size
    adc_draft_conversions = base_reads * xbar_size if use_adc_draft else 0.0
    adc_residual_conversions = base_reads * xbar_size if use_adc_residual else 0.0

    array_energy = array_activations * specs.array.energy_pj_per_activation
    dac_energy = dac_conversions * specs.dac.energy_pj_per_conversion
    adc_draft_energy = adc_draft_conversions * specs.adc_draft.energy_pj_per_conversion
    adc_residual_energy = adc_residual_conversions * specs.adc_residual.energy_pj_per_conversion

    array_latency = base_reads * specs.array.latency_ns_per_activation
    dac_latency = base_reads * specs.dac.latency_ns_per_conversion
    adc_draft_scan = (
        base_reads * adc_steps * specs.adc_draft.latency_ns_per_conversion if use_adc_draft else 0.0
    )
    adc_residual_scan = (
        base_reads * adc_steps * specs.adc_residual.latency_ns_per_conversion if use_adc_residual else 0.0
    )
    adc_draft_latency, adc_residual_latency, adc_latency = _parallel_latency_split(adc_draft_scan, adc_residual_scan)

    # Optional analog periphery (TIA, SNH, muxing, buffering, switches, drivers).
    adc_path_outputs = adc_draft_conversions + adc_residual_conversions
    _, _, adc_scan_latency_steps = _parallel_latency_split(adc_draft_scan, adc_residual_scan)

    def periph_energy_latency(spec, *, energy_ops: float, latency_ops: float) -> tuple[float, float]:  # noqa: ANN001
        return (energy_ops * spec.energy_pj_per_op, latency_ops * spec.latency_ns_per_op)

    tia_e, tia_t = periph_energy_latency(periphery.tia, energy_ops=adc_path_outputs, latency_ops=adc_scan_latency_steps)
    snh_e, snh_t = periph_energy_latency(periphery.snh, energy_ops=adc_path_outputs, latency_ops=adc_scan_latency_steps)
    mux_e, mux_t = periph_energy_latency(periphery.mux, energy_ops=adc_path_outputs, latency_ops=adc_scan_latency_steps)
    io_e, io_t = periph_energy_latency(
        periphery.io_buffers,
        energy_ops=adc_path_outputs,
        latency_ops=adc_scan_latency_steps,
    )
    sw_e, sw_t = periph_energy_latency(periphery.subarray_switches, energy_ops=array_activations, latency_ops=base_reads)
    wd_e, wd_t = periph_energy_latency(periphery.write_drivers, energy_ops=dac_conversions, latency_ops=base_reads)

    stage_energy = array_energy + dac_energy + adc_draft_energy + adc_residual_energy
    stage_latency = array_latency + dac_latency + adc_latency

    stage_energy += tia_e + snh_e + mux_e + io_e + sw_e + wd_e
    stage_latency += tia_t + snh_t + mux_t + io_t + sw_t + wd_t

    acc.add_stage(stage, stage_energy, stage_latency)
    acc.add_component("arrays", array_energy, array_latency)
    acc.add_component("dac", dac_energy, dac_latency)
    acc.add_component("adc_draft", adc_draft_energy, adc_draft_latency)
    acc.add_component("adc_residual", adc_residual_energy, adc_residual_latency)
    acc.add_component("tia", tia_e, tia_t)
    acc.add_component("snh", snh_e, snh_t)
    acc.add_component("mux", mux_e, mux_t)
    acc.add_component("io_buffers", io_e, io_t)
    acc.add_component("subarray_switches", sw_e, sw_t)
    acc.add_component("write_drivers", wd_e, wd_t)
    acc.add_analog_counts(
        array_activations=array_activations,
        dac_conversions=dac_conversions,
        adc_draft_conversions=adc_draft_conversions,
        adc_residual_conversions=adc_residual_conversions,
    )


def _add_knob_digital_stage(
    *,
    acc: _TokenAccumulator,
    stage: str,
    macs: int,
    energy_per_mac: float,
    latency_per_mac: float,
    parallel_units: int = 1,
) -> None:
    energy = macs * energy_per_mac
    latency = (macs * latency_per_mac) / float(max(parallel_units, 1))
    acc.add_stage(stage, energy, latency)
    if stage in {"qk", "pv"}:
        acc.add_component("attention_engine", energy, latency)
    elif stage == "softmax":
        acc.add_component("softmax_unit", energy, latency)
    elif stage == "elementwise":
        acc.add_component("elementwise_unit", energy, latency)
    elif stage == "kv_cache":
        acc.add_component("kv_cache", energy, latency)


def _token_step_costs_knob(
    model: ModelConfig,
    hardware: HardwareConfig,
    specs: ResolvedKnobSpecs,
    l_prompt: int,
) -> tuple[Breakdown, Breakdown]:
    assert hardware.analog is not None
    macs = _mac_counts_per_token(model, l_prompt)
    digital_costs = _digital_costs_knob(specs)
    digital_stages = DIGITAL_STAGES if hardware.memory is None else tuple(s for s in DIGITAL_STAGES if s != "kv_cache")
    num_tiles = _tile_counts(model, hardware.analog.xbar_size)
    num_slices = ceil(model.activation_bits / hardware.analog.dac_bits)
    buf_knobs = hardware.soc.buffers_add

    def add_buffers_add(acc: _TokenAccumulator, ops: float) -> None:
        if ops <= 0.0:
            return
        energy = ops * buf_knobs.energy_pj_per_op
        latency = ops * buf_knobs.latency_ns_per_op
        acc.add_stage("buffers_add", energy, latency)
        acc.add_component("buffers_add", energy, latency)

    draft = _TokenAccumulator()
    verify_full = _TokenAccumulator()

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)
        for stage, precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            _add_knob_analog_stage(
                acc=draft,
                stage=stage,
                num_tiles=num_tiles[stage],
                num_slices=num_slices,
                xbar_size=hardware.analog.xbar_size,
                adc_steps=hardware.analog.num_columns_per_adc,
                specs=specs,
                periphery=hardware.analog.periphery,
                mode_name="draft_full" if precision == PrecisionMode.full else "draft_default",
            )
            _add_knob_analog_stage(
                acc=verify_full,
                stage=stage,
                num_tiles=num_tiles[stage],
                num_slices=num_slices,
                xbar_size=hardware.analog.xbar_size,
                adc_steps=hardware.analog.num_columns_per_adc,
                specs=specs,
                periphery=hardware.analog.periphery,
                mode_name="verify_bonus",
            )

            outputs = float(num_tiles[stage] * num_slices * hardware.analog.xbar_size)
            if hardware.reuse_policy == ReusePolicy.reuse:
                add_buffers_add(draft, outputs)  # buffer D_reg / full outputs for reuse
            if precision == PrecisionMode.full:
                add_buffers_add(draft, outputs)  # ADC-output combine
            add_buffers_add(verify_full, outputs)  # ADC-output combine (bonus token)

        for stage in digital_stages:
            e_per, t_per = digital_costs[stage]
            parallel_units = hardware.soc.attention_cim_units if stage in {"qk", "pv"} else 1
            _add_knob_digital_stage(
                acc=draft,
                stage=stage,
                macs=macs[stage],
                energy_per_mac=e_per,
                latency_per_mac=t_per,
                parallel_units=parallel_units,
            )
            _add_knob_digital_stage(
                acc=verify_full,
                stage=stage,
                macs=macs[stage],
                energy_per_mac=e_per,
                latency_per_mac=t_per,
                parallel_units=parallel_units,
            )

    ctrl_e_tok = model.n_layers * hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = model.n_layers * hardware.soc.control.latency_ns_per_token
    draft.add_stage("control", ctrl_e_tok, ctrl_t_tok)
    draft.add_component("control", ctrl_e_tok, ctrl_t_tok)
    verify_full.add_stage("control", ctrl_e_tok, ctrl_t_tok)
    verify_full.add_component("control", ctrl_e_tok, ctrl_t_tok)

    ctrl_e_burst = model.n_layers * hardware.soc.control.energy_pj_per_burst
    ctrl_t_burst = model.n_layers * hardware.soc.control.latency_ns_per_burst
    verify_full.add_stage("control", ctrl_e_burst, ctrl_t_burst)
    verify_full.add_component("control", ctrl_e_burst, ctrl_t_burst)

    setup_e_burst = model.n_layers * hardware.soc.verify_setup.energy_pj_per_burst
    setup_t_burst = model.n_layers * hardware.soc.verify_setup.latency_ns_per_burst
    verify_full.add_stage("control", setup_e_burst, setup_t_burst)
    verify_full.add_component("control", setup_e_burst, setup_t_burst)

    return draft.to_breakdown(), verify_full.to_breakdown()


def _verify_drafted_token_additional_stage_knob(
    model: ModelConfig,
    hardware: HardwareConfig,
    specs: ResolvedKnobSpecs,
    l_prompt: int,
) -> Breakdown:
    assert hardware.analog is not None
    macs = _mac_counts_per_token(model, l_prompt)
    digital_costs = _digital_costs_knob(specs)
    digital_stages = DIGITAL_STAGES if hardware.memory is None else tuple(s for s in DIGITAL_STAGES if s != "kv_cache")
    num_tiles = _tile_counts(model, hardware.analog.xbar_size)
    num_slices = ceil(model.activation_bits / hardware.analog.dac_bits)
    buf_knobs = hardware.soc.buffers_add

    def add_buffers_add(acc: _TokenAccumulator, ops: float) -> None:
        if ops <= 0.0:
            return
        energy = ops * buf_knobs.energy_pj_per_op
        latency = ops * buf_knobs.latency_ns_per_op
        acc.add_stage("buffers_add", energy, latency)
        acc.add_component("buffers_add", energy, latency)

    additional = _TokenAccumulator()

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)
        for stage, executed_precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            if hardware.reuse_policy == ReusePolicy.reread:
                mode_name = "verify_full"
            else:
                if executed_precision == PrecisionMode.full:
                    mode_name = "none"
                else:
                    mode_name = "verify_residual_only"

            _add_knob_analog_stage(
                acc=additional,
                stage=stage,
                num_tiles=num_tiles[stage],
                num_slices=num_slices,
                xbar_size=hardware.analog.xbar_size,
                adc_steps=hardware.analog.num_columns_per_adc,
                specs=specs,
                periphery=hardware.analog.periphery,
                mode_name=mode_name,
            )

            outputs = float(num_tiles[stage] * num_slices * hardware.analog.xbar_size)
            if hardware.reuse_policy == ReusePolicy.reread:
                add_buffers_add(additional, outputs)  # ADC-output combine (re-read full)
            else:
                if executed_precision == PrecisionMode.full:
                    add_buffers_add(additional, outputs)  # buffer read of stored full outputs
                else:
                    add_buffers_add(additional, 2.0 * outputs)  # buffer read + Final = D_reg + C

        for stage in digital_stages:
            e_per, t_per = digital_costs[stage]
            parallel_units = hardware.soc.attention_cim_units if stage in {"qk", "pv"} else 1
            _add_knob_digital_stage(
                acc=additional,
                stage=stage,
                macs=macs[stage],
                energy_per_mac=e_per,
                latency_per_mac=t_per,
                parallel_units=parallel_units,
            )

    ctrl_e_tok = model.n_layers * hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = model.n_layers * hardware.soc.control.latency_ns_per_token
    additional.add_stage("control", ctrl_e_tok, ctrl_t_tok)
    additional.add_component("control", ctrl_e_tok, ctrl_t_tok)

    return additional.to_breakdown()


def _max_layer_compute_latencies_ns_knob(
    *,
    model: ModelConfig,
    hardware: HardwareConfig,
    specs: ResolvedKnobSpecs,
    l_prompt: int,
) -> tuple[float, float, float]:
    assert hardware.analog is not None
    macs = _mac_counts_per_token(model, l_prompt)
    digital_costs = _digital_costs_knob(specs)
    digital_stages = DIGITAL_STAGES if hardware.memory is None else tuple(s for s in DIGITAL_STAGES if s != "kv_cache")
    num_tiles = _tile_counts(model, hardware.analog.xbar_size)
    num_slices = ceil(model.activation_bits / hardware.analog.dac_bits)
    buf_knobs = hardware.soc.buffers_add

    def add_buffers_add(acc: _TokenAccumulator, ops: float) -> None:
        if ops <= 0.0:
            return
        energy = ops * buf_knobs.energy_pj_per_op
        latency = ops * buf_knobs.latency_ns_per_op
        acc.add_stage("buffers_add", energy, latency)
        acc.add_component("buffers_add", energy, latency)

    ctrl_e_tok = hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = hardware.soc.control.latency_ns_per_token
    ctrl_e_burst = hardware.soc.control.energy_pj_per_burst
    ctrl_t_burst = hardware.soc.control.latency_ns_per_burst
    setup_e_burst = hardware.soc.verify_setup.energy_pj_per_burst
    setup_t_burst = hardware.soc.verify_setup.latency_ns_per_burst

    max_draft = 0.0
    max_verify_drafted = 0.0
    max_verify_bonus = 0.0

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)

        draft = _TokenAccumulator()
        verify_drafted = _TokenAccumulator()
        verify_bonus = _TokenAccumulator()

        for stage, executed_precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            _add_knob_analog_stage(
                acc=draft,
                stage=stage,
                num_tiles=num_tiles[stage],
                num_slices=num_slices,
                xbar_size=hardware.analog.xbar_size,
                adc_steps=hardware.analog.num_columns_per_adc,
                specs=specs,
                periphery=hardware.analog.periphery,
                mode_name="draft_full" if executed_precision == PrecisionMode.full else "draft_default",
            )

            if hardware.reuse_policy == ReusePolicy.reread:
                verify_mode_name = "verify_full"
            else:
                if executed_precision == PrecisionMode.full:
                    verify_mode_name = "none"
                else:
                    verify_mode_name = "verify_residual_only"
            _add_knob_analog_stage(
                acc=verify_drafted,
                stage=stage,
                num_tiles=num_tiles[stage],
                num_slices=num_slices,
                xbar_size=hardware.analog.xbar_size,
                adc_steps=hardware.analog.num_columns_per_adc,
                specs=specs,
                periphery=hardware.analog.periphery,
                mode_name=verify_mode_name,
            )

            _add_knob_analog_stage(
                acc=verify_bonus,
                stage=stage,
                num_tiles=num_tiles[stage],
                num_slices=num_slices,
                xbar_size=hardware.analog.xbar_size,
                adc_steps=hardware.analog.num_columns_per_adc,
                specs=specs,
                periphery=hardware.analog.periphery,
                mode_name="verify_bonus",
            )

            outputs = float(num_tiles[stage] * num_slices * hardware.analog.xbar_size)

            if hardware.reuse_policy == ReusePolicy.reuse:
                add_buffers_add(draft, outputs)  # buffer D_reg / full outputs for reuse
            if executed_precision == PrecisionMode.full:
                add_buffers_add(draft, outputs)  # ADC-output combine
            add_buffers_add(verify_bonus, outputs)  # ADC-output combine (bonus token)

            if hardware.reuse_policy == ReusePolicy.reread:
                add_buffers_add(verify_drafted, outputs)  # ADC-output combine (re-read full)
            else:
                if executed_precision == PrecisionMode.full:
                    add_buffers_add(verify_drafted, outputs)  # buffer read of stored full outputs
                else:
                    add_buffers_add(verify_drafted, 2.0 * outputs)  # buffer read + Final = D_reg + C

        for stage in digital_stages:
            e_per, t_per = digital_costs[stage]
            parallel_units = hardware.soc.attention_cim_units if stage in {"qk", "pv"} else 1
            _add_knob_digital_stage(
                acc=draft,
                stage=stage,
                macs=macs[stage],
                energy_per_mac=e_per,
                latency_per_mac=t_per,
                parallel_units=parallel_units,
            )
            _add_knob_digital_stage(
                acc=verify_drafted,
                stage=stage,
                macs=macs[stage],
                energy_per_mac=e_per,
                latency_per_mac=t_per,
                parallel_units=parallel_units,
            )
            _add_knob_digital_stage(
                acc=verify_bonus,
                stage=stage,
                macs=macs[stage],
                energy_per_mac=e_per,
                latency_per_mac=t_per,
                parallel_units=parallel_units,
            )

        for acc in (draft, verify_drafted, verify_bonus):
            acc.add_stage("control", ctrl_e_tok, ctrl_t_tok)
            acc.add_component("control", ctrl_e_tok, ctrl_t_tok)

        verify_bonus.add_stage("control", ctrl_e_burst + setup_e_burst, ctrl_t_burst + setup_t_burst)
        verify_bonus.add_component("control", ctrl_e_burst + setup_e_burst, ctrl_t_burst + setup_t_burst)

        max_draft = max(max_draft, draft.to_breakdown().latency_ns)
        max_verify_drafted = max(max_verify_drafted, verify_drafted.to_breakdown().latency_ns)
        max_verify_bonus = max(max_verify_bonus, verify_bonus.to_breakdown().latency_ns)

    return max_draft, max_verify_drafted, max_verify_bonus


def _max_layer_compute_latencies_ns_legacy(
    *,
    model: ModelConfig,
    hardware: HardwareConfig,
    l_prompt: int,
) -> tuple[float, float, float]:
    macs = _mac_counts_per_token(model, l_prompt)
    digital_costs = _digital_costs_legacy(hardware)
    digital_stages = DIGITAL_STAGES if hardware.memory is None else tuple(s for s in DIGITAL_STAGES if s != "kv_cache")
    analog_outputs = {s: sum(m_out for m_out, _n_in in shapes) for s, shapes in _analog_stage_shapes(model).items()}
    buf_knobs = hardware.soc.buffers_add

    def stage_energy_latency(stage: str, energy_per_mac: float, latency_per_mac: float) -> tuple[float, float]:
        m = macs[stage]
        return (m * energy_per_mac, m * latency_per_mac)

    ctrl_e_tok = hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = hardware.soc.control.latency_ns_per_token
    ctrl_e_burst = hardware.soc.control.energy_pj_per_burst
    ctrl_t_burst = hardware.soc.control.latency_ns_per_burst
    setup_e_burst = hardware.soc.verify_setup.energy_pj_per_burst
    setup_t_burst = hardware.soc.verify_setup.latency_ns_per_burst

    max_draft = 0.0
    max_verify_drafted = 0.0
    max_verify_bonus = 0.0

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)

        draft = StageBreakdown()
        verify_drafted = StageBreakdown()
        verify_bonus = StageBreakdown()

        for block, executed_precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            e_per, t_per = _analog_cost_for_block(hardware, executed_precision)
            e, t = stage_energy_latency(block, e_per, t_per)
            draft = draft.add_energy_latency(block, e, t)

            assert hardware.costs is not None
            e_full, t_full = stage_energy_latency(
                block,
                hardware.costs.analog_full.energy_pj_per_mac,
                hardware.costs.analog_full.latency_ns_per_mac,
            )
            verify_bonus = verify_bonus.add_energy_latency(block, e_full, t_full)

            e_add, t_add = _verify_additional_cost_for_block(hardware, executed_precision, token_kind="drafted")
            verify_drafted = verify_drafted.add_energy_latency(block, macs[block] * e_add, macs[block] * t_add)

            outputs = float(analog_outputs[block])
            if hardware.reuse_policy == ReusePolicy.reuse:
                draft = draft.add_energy_latency(
                    "buffers_add",
                    outputs * buf_knobs.energy_pj_per_op,
                    outputs * buf_knobs.latency_ns_per_op,
                )
            if executed_precision == PrecisionMode.full:
                draft = draft.add_energy_latency(
                    "buffers_add",
                    outputs * buf_knobs.energy_pj_per_op,
                    outputs * buf_knobs.latency_ns_per_op,
                )

            verify_bonus = verify_bonus.add_energy_latency(
                "buffers_add",
                outputs * buf_knobs.energy_pj_per_op,
                outputs * buf_knobs.latency_ns_per_op,
            )

            if hardware.reuse_policy == ReusePolicy.reread:
                verify_drafted = verify_drafted.add_energy_latency(
                    "buffers_add",
                    outputs * buf_knobs.energy_pj_per_op,
                    outputs * buf_knobs.latency_ns_per_op,
                )
            else:
                if executed_precision == PrecisionMode.full:
                    verify_drafted = verify_drafted.add_energy_latency(
                        "buffers_add",
                        outputs * buf_knobs.energy_pj_per_op,
                        outputs * buf_knobs.latency_ns_per_op,
                    )
                else:
                    verify_drafted = verify_drafted.add_energy_latency(
                        "buffers_add",
                        2.0 * outputs * buf_knobs.energy_pj_per_op,
                        2.0 * outputs * buf_knobs.latency_ns_per_op,
                    )

        for stage in digital_stages:
            e_per, t_per = digital_costs[stage]
            e, t = stage_energy_latency(stage, e_per, t_per)
            draft = draft.add_energy_latency(stage, e, t)
            verify_drafted = verify_drafted.add_energy_latency(stage, e, t)
            verify_bonus = verify_bonus.add_energy_latency(stage, e, t)

        draft = draft.add_energy_latency("control", ctrl_e_tok, ctrl_t_tok)
        verify_drafted = verify_drafted.add_energy_latency("control", ctrl_e_tok, ctrl_t_tok)
        verify_bonus = verify_bonus.add_energy_latency("control", ctrl_e_tok, ctrl_t_tok)
        verify_bonus = verify_bonus.add_energy_latency(
            "control",
            ctrl_e_burst + setup_e_burst,
            ctrl_t_burst + setup_t_burst,
        )

        max_draft = max(max_draft, Breakdown.from_stage_breakdown(draft).latency_ns)
        max_verify_drafted = max(max_verify_drafted, Breakdown.from_stage_breakdown(verify_drafted).latency_ns)
        max_verify_bonus = max(max_verify_bonus, Breakdown.from_stage_breakdown(verify_bonus).latency_ns)

    return max_draft, max_verify_drafted, max_verify_bonus


def _baseline_stats() -> SpeculationStats:
    return SpeculationStats(k=0, histogram={0: 1.0})


def estimate_point(
    model: ModelConfig,
    hardware: HardwareConfig,
    stats: SpeculationStats,
    l_prompt: int,
) -> tuple[Metrics, PhaseBreakdown]:
    if hardware.memory is not None:
        max_context_tokens = hardware.memory.kv_cache.max_context_tokens
        if max_context_tokens is not None and l_prompt + stats.k > max_context_tokens:
            raise ValueError(
                "Max context capacity exceeded: "
                f"L_prompt ({l_prompt}) + K ({stats.k}) = {l_prompt + stats.k} > "
                f"memory.kv_cache.max_context_tokens ({max_context_tokens})"
            )

    if hardware.mode == HardwareMode.legacy:
        draft_step, verify_full_step = _token_step_costs_legacy(model, hardware, l_prompt)
        verify_drafted_additional = _verify_drafted_token_additional_stage_legacy(model, hardware, l_prompt)
    else:
        specs = hardware.resolve_knob_specs()
        draft_step, verify_full_step = _token_step_costs_knob(model, hardware, specs, l_prompt)
        verify_drafted_additional = _verify_drafted_token_additional_stage_knob(model, hardware, specs, l_prompt)

    draft_phase = draft_step.scale(stats.k)
    verify_drafted_phase = verify_drafted_additional.scale(stats.k)
    verify_bonus_phase = verify_full_step

    if hardware.memory is not None:
        traffic = _kv_memory_traffic_by_phase(model=model, hardware=hardware, stats=stats, l_prompt=l_prompt)
        draft_phase = _add_memory_traffic_costs(breakdown=draft_phase, traffic=traffic["draft"], hardware=hardware)
        verify_drafted_phase = _add_memory_traffic_costs(
            breakdown=verify_drafted_phase,
            traffic=traffic["verify_drafted"],
            hardware=hardware,
        )
        verify_bonus_phase = _add_memory_traffic_costs(
            breakdown=verify_bonus_phase,
            traffic=traffic["verify_bonus"],
            hardware=hardware,
        )

    total_stages = draft_phase.stages.plus(verify_drafted_phase.stages).plus(verify_bonus_phase.stages)

    total_components = None
    if (
        draft_phase.components is not None
        and verify_drafted_phase.components is not None
        and verify_bonus_phase.components is not None
    ):
        total_components = (
            draft_phase.components.plus(verify_drafted_phase.components).plus(verify_bonus_phase.components)
        )

    total_activation_counts = None
    if (
        draft_phase.activation_counts is not None
        and verify_drafted_phase.activation_counts is not None
        and verify_bonus_phase.activation_counts is not None
    ):
        total_activation_counts = (
            draft_phase.activation_counts.plus(verify_drafted_phase.activation_counts).plus(verify_bonus_phase.activation_counts)
        )

    total_memory_traffic = None
    if (
        draft_phase.memory_traffic is not None
        and verify_drafted_phase.memory_traffic is not None
        and verify_bonus_phase.memory_traffic is not None
    ):
        total_memory_traffic = (
            draft_phase.memory_traffic.plus(verify_drafted_phase.memory_traffic).plus(verify_bonus_phase.memory_traffic)
        )

    total_phase = Breakdown.from_stage_breakdown(
        total_stages,
        components=total_components,
        activation_counts=total_activation_counts,
        memory_traffic=total_memory_traffic,
    )

    e_burst = total_phase.energy_pj
    t_burst = total_phase.latency_ns

    committed = expected_committed_tokens_per_burst(stats)
    if committed <= 0:
        raise ValueError("Expected committed tokens per burst must be > 0")

    energy_per_token_pj = e_burst / committed
    latency_per_token_ns = t_burst / committed

    if hardware.soc.schedule == ScheduleMode.layer_pipelined:
        if hardware.mode == HardwareMode.legacy:
            max_layer_draft, max_layer_verify_drafted, max_layer_verify_bonus = _max_layer_compute_latencies_ns_legacy(
                model=model,
                hardware=hardware,
                l_prompt=l_prompt,
            )
        else:
            specs = hardware.resolve_knob_specs()
            max_layer_draft, max_layer_verify_drafted, max_layer_verify_bonus = _max_layer_compute_latencies_ns_knob(
                model=model,
                hardware=hardware,
                specs=specs,
                l_prompt=l_prompt,
            )

        mem_draft_per_step = 0.0
        mem_verify_drafted_per_step = 0.0
        mem_verify_bonus = 0.0
        if hardware.memory is not None:
            if stats.k > 0:
                mem_draft_per_step = draft_phase.stages.kv_cache_latency_ns / float(stats.k)
                mem_verify_drafted_per_step = verify_drafted_phase.stages.kv_cache_latency_ns / float(stats.k)
            mem_verify_bonus = verify_bonus_phase.stages.kv_cache_latency_ns

        t_draft = max(max_layer_draft, mem_draft_per_step)
        t_verify_drafted = max(max_layer_verify_drafted, mem_verify_drafted_per_step)
        t_verify_bonus = max(max_layer_verify_bonus, mem_verify_bonus)

        t_burst_pipe = float(stats.k) * t_draft + float(stats.k) * t_verify_drafted + t_verify_bonus
        latency_per_token_ns = t_burst_pipe / committed

    throughput_tokens_per_s = 0.0 if latency_per_token_ns == 0 else 1e9 / latency_per_token_ns
    tokens_per_joule = 0.0 if energy_per_token_pj == 0 else 1e12 / energy_per_token_pj

    breakdown = PhaseBreakdown(
        draft=draft_phase,
        verify_drafted=verify_drafted_phase,
        verify_bonus=verify_bonus_phase,
        total=total_phase,
    )
    metrics = Metrics(
        energy_pj_per_token=energy_per_token_pj,
        latency_ns_per_token=latency_per_token_ns,
        throughput_tokens_per_s=throughput_tokens_per_s,
        tokens_per_joule=tokens_per_joule,
    )
    return metrics, breakdown


def estimate_sweep(
    model: ModelConfig,
    hardware: HardwareConfig,
    stats: SpeculationStats,
    prompt_lengths: list[int],
    paths: dict[str, str] | None = None,
) -> Report:
    paths_obj = None
    if paths is not None:
        paths_obj = InputPaths(**paths)

    points: list[SweepPoint] = []
    for l_prompt in prompt_lengths:
        speculative_metrics, speculative_breakdown = estimate_point(model, hardware, stats, l_prompt)
        baseline_metrics, baseline_breakdown = estimate_point(model, hardware, _baseline_stats(), l_prompt)
        delta = BaselineDelta.from_metrics(speculative_metrics, baseline_metrics)
        points.append(
            SweepPoint(
                l_prompt=l_prompt,
                speculative=speculative_metrics,
                baseline=baseline_metrics,
                delta=delta,
                breakdown=speculative_breakdown,
                baseline_breakdown=baseline_breakdown,
            )
        )

    break_even = None
    for p in sorted(points, key=lambda sp: sp.l_prompt):
        if p.delta.tokens_per_joule_ratio is not None and p.delta.tokens_per_joule_ratio > 1.0:
            break_even = p.l_prompt
            break

    resolved_library = hardware.resolved_library_payload()
    model_knobs: dict[str, Any] = {
        "activation_bits": model.activation_bits,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "ffn_type": model.ffn_type.value,
        "d_ff": model.effective_d_ff,
    }
    hardware_knobs: dict[str, Any] = {"reuse_policy": hardware.reuse_policy.value}
    if hardware.mode == HardwareMode.knob_based and hardware.analog is not None:
        hardware_knobs.update(
            {
                "library": hardware.selected_library,
                "xbar_size": hardware.analog.xbar_size,
                "num_columns_per_adc": hardware.analog.num_columns_per_adc,
                "dac_bits": hardware.analog.dac_bits,
                "adc": {
                    "draft_bits": hardware.analog.adc.draft_bits,
                    "residual_bits": hardware.analog.adc.residual_bits,
                },
            }
        )
        periph = hardware.analog.periphery
        if any(
            getattr(getattr(periph, name), field) != 0.0
            for name in ["tia", "snh", "mux", "io_buffers", "subarray_switches", "write_drivers"]
            for field in ["energy_pj_per_op", "latency_ns_per_op", "area_mm2_per_unit"]
        ):
            hardware_knobs["analog_periphery"] = periph.model_dump(mode="json")

    if hardware.memory is not None:
        hardware_knobs["memory"] = hardware.memory.model_dump(mode="json")

    if (
        hardware.soc.schedule != ScheduleMode.serialized
        or hardware.soc.verify_setup.energy_pj_per_burst != 0.0
        or hardware.soc.verify_setup.latency_ns_per_burst != 0.0
        or any(
            getattr(hardware.soc.buffers_add, field) != 0.0
            for field in ["energy_pj_per_op", "latency_ns_per_op", "area_mm2_per_unit"]
        )
        or any(
            getattr(hardware.soc.control, field) != 0.0
            for field in [
                "energy_pj_per_token",
                "latency_ns_per_token",
                "energy_pj_per_burst",
                "latency_ns_per_burst",
            ]
        )
    ):
        hardware_knobs["soc"] = hardware.soc.model_dump(mode="json")

    return Report(
        generated_at=datetime.now(timezone.utc).isoformat(),
        k=stats.k,
        reuse_policy=hardware.reuse_policy.value,
        hardware_mode=hardware.mode.value,
        resolved_library=resolved_library,
        model_knobs=model_knobs,
        hardware_knobs=hardware_knobs,
        paths=paths_obj,
        points=points,
        break_even_tokens_per_joule_l_prompt=break_even,
        area=_area_mm2(model, hardware),
        area_breakdown_mm2=_area_breakdown_mm2(model, hardware),
        notes=[
            "Analytical calculator (closed-form activation counts, no event simulation).",
            "No early-stop on mismatch; verifier suffix work is still charged.",
            "Breakdown latencies are serialized sums; `soc.schedule` affects how latency/token is reported.",
        ],
    )

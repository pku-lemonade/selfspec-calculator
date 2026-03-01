from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from typing import Any

from .config import (
    DpuFeatureCostOverrides,
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
    CostChannelBreakdown,
    ComponentBreakdown,
    DpuFeatureBreakdown,
    LeakageSummary,
    MemoryTraffic,
    Metrics,
    MovementAccountingCoverage,
    PhaseBreakdown,
    Report,
    StageBreakdown,
    SweepPoint,
)
from .stats import SpeculationStats, expected_committed_tokens_per_burst, normalize_histogram


ANALOG_STAGES = ("qkv", "wo", "ffn")
DPU_FEATURES = (
    "attention_qk",
    "attention_softmax",
    "attention_pv",
    "ffn_activation",
    "ffn_gate_multiply",
    "kv_cache_update",
)
DPU_STAGE_BY_FEATURE = {
    "attention_qk": "qk",
    "attention_softmax": "softmax",
    "attention_pv": "pv",
    "ffn_activation": "elementwise",
    "ffn_gate_multiply": "elementwise",
    "kv_cache_update": "kv_cache",
}
DPU_COMPONENT_BY_FEATURE = {
    "attention_qk": "attention_engine",
    "attention_softmax": "softmax_unit",
    "attention_pv": "attention_engine",
    "ffn_activation": "elementwise_unit",
    "ffn_gate_multiply": "elementwise_unit",
    "kv_cache_update": "kv_cache",
}
MOVEMENT_ACCOUNTING_COVERAGE = MovementAccountingCoverage(
    modeled=[
        "kv_cache_sram_read_write",
        "kv_cache_hbm_read_write",
        "kv_cache_fabric_transfer",
    ],
    proxy_modeled=[],
    excluded=[
        "non_kv_intermediate_activation_movement (attention scores, FFN intermediates, residual streams)",
        "weight_movement_and_prefetch_traffic",
    ],
    ownership_rules={
        "kv_cache_update": "Counted as DPU compute only when memory modeling is disabled; otherwise ownership moves to memory movement accounting.",
        "qk/pv/softmax/ffn digital features": "Owned by compute channel only.",
    },
)


def _pipeline_policy_metadata(schedule: ScheduleMode) -> dict[str, Any]:
    if schedule == ScheduleMode.layer_pipelined:
        return {
            "schedule": schedule.value,
            "draft_stage": "serialized",
            "verify_stage": "wavefront",
            "mismatch_policy": "stop-at-first-mismatch",
            "full_accept_policy": "execute-bonus-and-commit",
            "bonus_in_verify_wavefront": True,
        }
    return {
        "schedule": schedule.value,
        "draft_stage": "serialized",
        "verify_stage": "serialized",
        "mismatch_policy": "not-modeled",
        "full_accept_policy": "always-charge-verify-bonus-phase",
        "bonus_in_verify_wavefront": False,
    }


def _executed_verify_drafted_steps_for_outcome(*, k: int, accepted_prefix: int) -> int:
    if accepted_prefix < k:
        return accepted_prefix + 1
    return k


def _executes_verify_bonus_for_outcome(*, k: int, accepted_prefix: int) -> bool:
    return accepted_prefix == k


def _verify_setup_breakdown(model: ModelConfig, hardware: HardwareConfig) -> Breakdown:
    energy = float(model.n_layers) * (
        hardware.soc.control.energy_pj_per_burst + hardware.soc.verify_setup.energy_pj_per_burst
    )
    latency = float(model.n_layers) * (
        hardware.soc.control.latency_ns_per_burst + hardware.soc.verify_setup.latency_ns_per_burst
    )

    stages = StageBreakdown().add_energy_latency("control", energy, latency)
    components = ComponentBreakdown().add_energy_latency("control", energy, latency)
    channels = CostChannelBreakdown().add_energy_latency("compute", energy, latency)
    return Breakdown.from_stage_breakdown(stages, components=components, channels=channels)


def _total_leakage_power_nw(hardware: HardwareConfig) -> float:
    leakage = hardware.leakage_power
    return float(sum(getattr(leakage, field) for field in type(leakage).model_fields))


def _leakage_energy_pj(*, leakage_power_nw: float, burst_latency_ns: float) -> float:
    # nW * ns = 1e-6 pJ
    return float(leakage_power_nw) * float(burst_latency_ns) * 1e-6


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
    draft = MemoryTraffic()
    verify_drafted = MemoryTraffic()
    for i in range(k):
        step = _kv_memory_step_traffic(
            model=model,
            hardware=hardware,
            prompt_tokens_from_hbm=float(l_prompt),
            speculative_tokens_from_sram=float(i),
            speculative_tokens_to_sram=1.0,
            committed_tokens_to_hbm=0.0,
        )
        draft = draft.plus(step)
        verify_drafted = verify_drafted.plus(step)

    committed_tokens = expected_committed_tokens_per_burst(stats)
    verify_bonus = _kv_memory_step_traffic(
        model=model,
        hardware=hardware,
        prompt_tokens_from_hbm=float(l_prompt),
        speculative_tokens_from_sram=float(k),
        speculative_tokens_to_sram=1.0,
        committed_tokens_to_hbm=committed_tokens,
    )
    return {"draft": draft, "verify_drafted": verify_drafted, "verify_bonus": verify_bonus}


def _kv_memory_step_traffic(
    *,
    model: ModelConfig,
    hardware: HardwareConfig,
    prompt_tokens_from_hbm: float,
    speculative_tokens_from_sram: float,
    speculative_tokens_to_sram: float,
    committed_tokens_to_hbm: float,
) -> MemoryTraffic:
    if hardware.memory is None:
        return MemoryTraffic()

    fmt_hbm = hardware.memory.kv_cache.hbm
    fmt_sram = hardware.memory.kv_cache.resolved_sram()
    bytes_hbm_token = _kv_bytes_per_token_per_layer(d_model=model.d_model, n_heads=model.n_heads, fmt=fmt_hbm)
    bytes_sram_token = _kv_bytes_per_token_per_layer(d_model=model.d_model, n_heads=model.n_heads, fmt=fmt_sram)

    def tokens_to_bytes(tokens: float, bytes_per_token: int) -> float:
        return float(tokens) * float(model.n_layers) * float(bytes_per_token)

    traffic = MemoryTraffic(
        sram_read_bytes=tokens_to_bytes(speculative_tokens_from_sram, bytes_sram_token),
        sram_write_bytes=tokens_to_bytes(speculative_tokens_to_sram, bytes_sram_token),
        hbm_read_bytes=tokens_to_bytes(prompt_tokens_from_hbm, bytes_hbm_token),
        hbm_write_bytes=tokens_to_bytes(committed_tokens_to_hbm, bytes_hbm_token),
    )
    traffic.fabric_read_bytes = traffic.sram_read_bytes + traffic.hbm_read_bytes
    traffic.fabric_write_bytes = traffic.sram_write_bytes + traffic.hbm_write_bytes
    return traffic


def _memory_cost_from_traffic(*, hardware: HardwareConfig, traffic: MemoryTraffic) -> tuple[float, float]:
    if hardware.memory is None:
        return (0.0, 0.0)

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
    return (sram_e + hbm_e + fabric_e, sram_t + hbm_t + fabric_t)


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
    mem_energy, mem_latency = _memory_cost_from_traffic(hardware=hardware, traffic=traffic)

    stages = breakdown.stages.add_energy_latency("kv_cache", mem_energy, mem_latency)

    components = breakdown.components or ComponentBreakdown()
    components = components.add_energy_latency("sram", sram_e, sram_t)
    components = components.add_energy_latency("hbm", hbm_e, hbm_t)
    components = components.add_energy_latency("fabric", fabric_e, fabric_t)
    channels = breakdown.channels or CostChannelBreakdown()
    channels = channels.add_energy_latency("movement", mem_energy, mem_latency)

    return Breakdown.from_stage_breakdown(
        stages,
        components=components,
        activation_counts=breakdown.activation_counts,
        memory_traffic=traffic,
        dpu_features=breakdown.dpu_features,
        channels=channels,
    )


def _analog_macs_per_token(model: ModelConfig) -> dict[str, int]:
    d_model = model.d_model
    d_ff = model.effective_d_ff

    qkv_macs = 3 * d_model * d_model
    wo_macs = d_model * d_model
    if model.ffn_type.value == "mlp":
        ffn_macs = 2 * d_model * d_ff
    else:
        ffn_macs = 3 * d_model * d_ff

    return {
        "qkv": qkv_macs,
        "wo": wo_macs,
        "ffn": ffn_macs,
    }


def _dpu_feature_ops_per_token(model: ModelConfig, l_prompt: int) -> dict[str, int]:
    d_ff = model.effective_d_ff
    d_model = model.d_model
    qk_and_pv_ops = model.n_heads * l_prompt * model.d_head
    ffn_gate_multiply_ops = d_ff if model.ffn_type.value == "swiglu" else 0

    return {
        "attention_qk": qk_and_pv_ops,
        "attention_softmax": model.n_heads * l_prompt,
        "attention_pv": qk_and_pv_ops,
        "ffn_activation": d_ff,
        "ffn_gate_multiply": ffn_gate_multiply_ops,
        "kv_cache_update": d_model,
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


def _dpu_feature_costs_legacy(hardware: HardwareConfig) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    assert hardware.costs is not None
    overrides = hardware.costs.digital_features or DpuFeatureCostOverrides()
    mapping: dict[str, str] = {}

    def pick(
        *,
        feature: str,
        explicit,  # noqa: ANN001
        fallback,  # noqa: ANN001
        fallback_source: str,
    ) -> tuple[float, float]:
        if explicit is not None:
            mapping[feature] = f"explicit:costs.digital_features.{feature}"
            return (explicit.energy_pj_per_mac, explicit.latency_ns_per_mac)
        mapping[feature] = f"mapped:{fallback_source}"
        return (fallback.energy_pj_per_mac, fallback.latency_ns_per_mac)

    return (
        {
            "attention_qk": pick(
                feature="attention_qk",
                explicit=overrides.attention_qk,
                fallback=hardware.costs.digital_attention,
                fallback_source="costs.digital_attention",
            ),
            "attention_softmax": pick(
                feature="attention_softmax",
                explicit=overrides.attention_softmax,
                fallback=hardware.costs.digital_softmax,
                fallback_source="costs.digital_softmax",
            ),
            "attention_pv": pick(
                feature="attention_pv",
                explicit=overrides.attention_pv,
                fallback=hardware.costs.digital_attention,
                fallback_source="costs.digital_attention",
            ),
            "ffn_activation": pick(
                feature="ffn_activation",
                explicit=overrides.ffn_activation,
                fallback=hardware.costs.digital_elementwise,
                fallback_source="costs.digital_elementwise",
            ),
            "ffn_gate_multiply": pick(
                feature="ffn_gate_multiply",
                explicit=overrides.ffn_gate_multiply,
                fallback=hardware.costs.digital_elementwise,
                fallback_source="costs.digital_elementwise",
            ),
            "kv_cache_update": pick(
                feature="kv_cache_update",
                explicit=overrides.kv_cache_update,
                fallback=hardware.costs.kv_cache,
                fallback_source="costs.kv_cache",
            ),
        },
        mapping,
    )


def _dpu_feature_costs_knob(specs: ResolvedKnobSpecs) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    resolved, mapping = specs.digital.resolve_feature_costs()
    return (
        {
            "attention_qk": (resolved.attention_qk.energy_pj_per_mac, resolved.attention_qk.latency_ns_per_mac),
            "attention_softmax": (
                resolved.attention_softmax.energy_pj_per_mac,
                resolved.attention_softmax.latency_ns_per_mac,
            ),
            "attention_pv": (resolved.attention_pv.energy_pj_per_mac, resolved.attention_pv.latency_ns_per_mac),
            "ffn_activation": (resolved.ffn_activation.energy_pj_per_mac, resolved.ffn_activation.latency_ns_per_mac),
            "ffn_gate_multiply": (
                resolved.ffn_gate_multiply.energy_pj_per_mac,
                resolved.ffn_gate_multiply.latency_ns_per_mac,
            ),
            "kv_cache_update": (
                resolved.kv_cache_update.energy_pj_per_mac,
                resolved.kv_cache_update.latency_ns_per_mac,
            ),
        },
        mapping,
    )


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


def _add_dpu_feature_stage(
    *,
    acc: _TokenAccumulator,
    feature: str,
    ops: float,
    energy_per_op: float,
    latency_per_op: float,
    parallel_units: int = 1,
) -> None:
    if ops <= 0.0:
        return

    energy = ops * energy_per_op
    latency = (ops * latency_per_op) / float(max(parallel_units, 1))
    stage = DPU_STAGE_BY_FEATURE[feature]
    component = DPU_COMPONENT_BY_FEATURE[feature]
    acc.add_stage(stage, energy, latency)
    acc.add_component(component, energy, latency)
    acc.add_dpu_feature(feature, ops=ops, energy_pj=energy, latency_ns=latency)


def _attention_cim_total_units(model: ModelConfig, hardware: HardwareConfig) -> float:
    return float(model.n_layers) * float(hardware.soc.attention_cim_units)


def _attention_cim_unit_mac_area_mm2(*, hardware: HardwareConfig) -> float:
    return float(hardware.soc.attention_cim_mac_area_mm2_per_unit)


def _attention_cim_unit_sram_area_mm2(*, model: ModelConfig, hardware: HardwareConfig) -> float:
    if hardware.memory is None:
        return 0.0
    sram = hardware.memory.sram
    if sram.area_mm2 <= 0.0 or sram.capacity_bytes is None or sram.capacity_bytes <= 0:
        return 0.0
    assert hardware.analog is not None

    bits_per_element = hardware.soc.attention_cim_storage_bits_per_element or model.activation_bits
    bytes_per_element = ceil(bits_per_element / 8)
    logical_array_elements = hardware.analog.xbar_size * hardware.analog.xbar_size
    unit_sram_bytes = logical_array_elements * bytes_per_element
    area_per_byte = float(sram.area_mm2) / float(sram.capacity_bytes)
    return float(unit_sram_bytes) * area_per_byte


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
        digital = specs.digital.digital_overhead_area_mm2_per_layer

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
    attention_cim_sram_mm2 = 0.0
    attention_cim_mac_mm2 = 0.0

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
        attention_cim_units_total = _attention_cim_total_units(model, hardware)
        attention_cim_sram_mm2 = attention_cim_units_total * _attention_cim_unit_sram_area_mm2(
            model=model, hardware=hardware
        )
        attention_cim_mac_mm2 = attention_cim_units_total * _attention_cim_unit_mac_area_mm2(hardware=hardware)

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
        attention_cim_sram_mm2=attention_cim_sram_mm2,
        attention_cim_mac_mm2=attention_cim_mac_mm2,
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
    analog_macs = _analog_macs_per_token(model)
    dpu_feature_ops = _dpu_feature_ops_per_token(model, l_prompt)
    dpu_feature_costs, _mapping = _dpu_feature_costs_legacy(hardware)
    enabled_dpu_features = (
        DPU_FEATURES if hardware.memory is None else tuple(feature for feature in DPU_FEATURES if feature != "kv_cache_update")
    )
    analog_outputs = {s: sum(m_out for m_out, _n_in in shapes) for s, shapes in _analog_stage_shapes(model).items()}
    buf_knobs = hardware.soc.buffers_add

    draft = _TokenAccumulator()
    verify_full = _TokenAccumulator()

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)
        for block, precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            e_per, t_per = _analog_cost_for_block(hardware, precision)
            draft_energy = analog_macs[block] * e_per
            draft_latency = analog_macs[block] * t_per
            draft.add_stage(block, draft_energy, draft_latency)
            draft.add_component("arrays", draft_energy, draft_latency)

            assert hardware.costs is not None
            full_energy = analog_macs[block] * hardware.costs.analog_full.energy_pj_per_mac
            full_latency = analog_macs[block] * hardware.costs.analog_full.latency_ns_per_mac
            verify_full.add_stage(
                block,
                full_energy,
                full_latency,
            )
            verify_full.add_component("arrays", full_energy, full_latency)

            outputs = float(analog_outputs[block])
            if hardware.reuse_policy == ReusePolicy.reuse:
                energy = outputs * buf_knobs.energy_pj_per_op
                latency = outputs * buf_knobs.latency_ns_per_op
                draft.add_stage("buffers_add", energy, latency)
                draft.add_component("buffers_add", energy, latency)
            if precision == PrecisionMode.full:
                energy = outputs * buf_knobs.energy_pj_per_op
                latency = outputs * buf_knobs.latency_ns_per_op
                draft.add_stage("buffers_add", energy, latency)
                draft.add_component("buffers_add", energy, latency)

            bonus_energy = outputs * buf_knobs.energy_pj_per_op
            bonus_latency = outputs * buf_knobs.latency_ns_per_op
            verify_full.add_stage("buffers_add", bonus_energy, bonus_latency)
            verify_full.add_component("buffers_add", bonus_energy, bonus_latency)

        for feature in enabled_dpu_features:
            e_per, t_per = dpu_feature_costs[feature]
            parallel_units = hardware.soc.attention_cim_units if feature in {"attention_qk", "attention_pv"} else 1
            ops = float(dpu_feature_ops[feature])
            _add_dpu_feature_stage(
                acc=draft,
                feature=feature,
                ops=ops,
                energy_per_op=e_per,
                latency_per_op=t_per,
                parallel_units=parallel_units,
            )
            _add_dpu_feature_stage(
                acc=verify_full,
                feature=feature,
                ops=ops,
                energy_per_op=e_per,
                latency_per_op=t_per,
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
    setup_e_burst = model.n_layers * hardware.soc.verify_setup.energy_pj_per_burst
    setup_t_burst = model.n_layers * hardware.soc.verify_setup.latency_ns_per_burst
    verify_full.add_stage("control", ctrl_e_burst + setup_e_burst, ctrl_t_burst + setup_t_burst)
    verify_full.add_component("control", ctrl_e_burst + setup_e_burst, ctrl_t_burst + setup_t_burst)

    draft_breakdown = draft.to_breakdown().model_copy(update={"activation_counts": None})
    verify_full_breakdown = verify_full.to_breakdown().model_copy(update={"activation_counts": None})
    return draft_breakdown, verify_full_breakdown


def _verify_drafted_token_additional_stage_legacy(
    model: ModelConfig, hardware: HardwareConfig, l_prompt: int
) -> Breakdown:
    analog_macs = _analog_macs_per_token(model)
    dpu_feature_ops = _dpu_feature_ops_per_token(model, l_prompt)
    dpu_feature_costs, _mapping = _dpu_feature_costs_legacy(hardware)
    enabled_dpu_features = (
        DPU_FEATURES if hardware.memory is None else tuple(feature for feature in DPU_FEATURES if feature != "kv_cache_update")
    )
    analog_outputs = {s: sum(m_out for m_out, _n_in in shapes) for s, shapes in _analog_stage_shapes(model).items()}
    buf_knobs = hardware.soc.buffers_add

    additional = _TokenAccumulator()

    for layer in range(model.n_layers):
        policy = model.draft_policy.for_layer(layer)
        for block, executed_precision in {"qkv": policy.qkv, "wo": policy.wo, "ffn": policy.ffn}.items():
            e_per, t_per = _verify_additional_cost_for_block(hardware, executed_precision, token_kind="drafted")
            analog_energy = analog_macs[block] * e_per
            analog_latency = analog_macs[block] * t_per
            additional.add_stage(block, analog_energy, analog_latency)
            additional.add_component("arrays", analog_energy, analog_latency)

            outputs = float(analog_outputs[block])
            if hardware.reuse_policy == ReusePolicy.reread:
                energy = outputs * buf_knobs.energy_pj_per_op
                latency = outputs * buf_knobs.latency_ns_per_op
                additional.add_stage("buffers_add", energy, latency)
                additional.add_component("buffers_add", energy, latency)
            else:
                if executed_precision == PrecisionMode.full:
                    energy = outputs * buf_knobs.energy_pj_per_op
                    latency = outputs * buf_knobs.latency_ns_per_op
                    additional.add_stage("buffers_add", energy, latency)
                    additional.add_component("buffers_add", energy, latency)
                else:
                    energy = 2.0 * outputs * buf_knobs.energy_pj_per_op
                    latency = 2.0 * outputs * buf_knobs.latency_ns_per_op
                    additional.add_stage("buffers_add", energy, latency)
                    additional.add_component("buffers_add", energy, latency)

        for feature in enabled_dpu_features:
            e_per, t_per = dpu_feature_costs[feature]
            parallel_units = hardware.soc.attention_cim_units if feature in {"attention_qk", "attention_pv"} else 1
            _add_dpu_feature_stage(
                acc=additional,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
                parallel_units=parallel_units,
            )

    ctrl_e_tok = model.n_layers * hardware.soc.control.energy_pj_per_token
    ctrl_t_tok = model.n_layers * hardware.soc.control.latency_ns_per_token
    additional.add_stage("control", ctrl_e_tok, ctrl_t_tok)
    additional.add_component("control", ctrl_e_tok, ctrl_t_tok)

    return additional.to_breakdown().model_copy(update={"activation_counts": None})


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
        self.dpu_features = DpuFeatureBreakdown()
        self.channels = CostChannelBreakdown()

    def add_stage(self, stage: str, energy_pj: float, latency_ns: float) -> None:
        self.stages = self.stages.add_energy_latency(stage, energy_pj, latency_ns)
        self.channels = self.channels.add_energy_latency("compute", energy_pj, latency_ns)

    def add_component(self, component: str, energy_pj: float, latency_ns: float) -> None:
        self.components = self.components.add_energy_latency(component, energy_pj, latency_ns)

    def add_dpu_feature(self, feature: str, *, ops: float, energy_pj: float, latency_ns: float) -> None:
        self.dpu_features = self.dpu_features.add(feature, ops=ops, energy_pj=energy_pj, latency_ns=latency_ns)

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
            dpu_features=self.dpu_features,
            channels=self.channels,
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
    feature: str,
    ops: float,
    energy_per_op: float,
    latency_per_op: float,
    parallel_units: int = 1,
) -> None:
    _add_dpu_feature_stage(
        acc=acc,
        feature=feature,
        ops=ops,
        energy_per_op=energy_per_op,
        latency_per_op=latency_per_op,
        parallel_units=parallel_units,
    )


def _token_step_costs_knob(
    model: ModelConfig,
    hardware: HardwareConfig,
    specs: ResolvedKnobSpecs,
    l_prompt: int,
) -> tuple[Breakdown, Breakdown]:
    assert hardware.analog is not None
    dpu_feature_ops = _dpu_feature_ops_per_token(model, l_prompt)
    dpu_feature_costs, _mapping = _dpu_feature_costs_knob(specs)
    enabled_dpu_features = (
        DPU_FEATURES if hardware.memory is None else tuple(feature for feature in DPU_FEATURES if feature != "kv_cache_update")
    )
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

        for feature in enabled_dpu_features:
            e_per, t_per = dpu_feature_costs[feature]
            parallel_units = hardware.soc.attention_cim_units if feature in {"attention_qk", "attention_pv"} else 1
            _add_knob_digital_stage(
                acc=draft,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
                parallel_units=parallel_units,
            )
            _add_knob_digital_stage(
                acc=verify_full,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
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
    dpu_feature_ops = _dpu_feature_ops_per_token(model, l_prompt)
    dpu_feature_costs, _mapping = _dpu_feature_costs_knob(specs)
    enabled_dpu_features = (
        DPU_FEATURES if hardware.memory is None else tuple(feature for feature in DPU_FEATURES if feature != "kv_cache_update")
    )
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

        for feature in enabled_dpu_features:
            e_per, t_per = dpu_feature_costs[feature]
            parallel_units = hardware.soc.attention_cim_units if feature in {"attention_qk", "attention_pv"} else 1
            _add_knob_digital_stage(
                acc=additional,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
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
    dpu_feature_ops = _dpu_feature_ops_per_token(model, l_prompt)
    dpu_feature_costs, _mapping = _dpu_feature_costs_knob(specs)
    enabled_dpu_features = (
        DPU_FEATURES if hardware.memory is None else tuple(feature for feature in DPU_FEATURES if feature != "kv_cache_update")
    )
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

        for feature in enabled_dpu_features:
            e_per, t_per = dpu_feature_costs[feature]
            parallel_units = hardware.soc.attention_cim_units if feature in {"attention_qk", "attention_pv"} else 1
            _add_knob_digital_stage(
                acc=draft,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
                parallel_units=parallel_units,
            )
            _add_knob_digital_stage(
                acc=verify_drafted,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
                parallel_units=parallel_units,
            )
            _add_knob_digital_stage(
                acc=verify_bonus,
                feature=feature,
                ops=float(dpu_feature_ops[feature]),
                energy_per_op=e_per,
                latency_per_op=t_per,
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
    analog_macs = _analog_macs_per_token(model)
    dpu_feature_ops = _dpu_feature_ops_per_token(model, l_prompt)
    dpu_feature_costs, _mapping = _dpu_feature_costs_legacy(hardware)
    enabled_dpu_features = (
        DPU_FEATURES if hardware.memory is None else tuple(feature for feature in DPU_FEATURES if feature != "kv_cache_update")
    )
    analog_outputs = {s: sum(m_out for m_out, _n_in in shapes) for s, shapes in _analog_stage_shapes(model).items()}
    buf_knobs = hardware.soc.buffers_add

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
            draft = draft.add_energy_latency(block, analog_macs[block] * e_per, analog_macs[block] * t_per)

            assert hardware.costs is not None
            verify_bonus = verify_bonus.add_energy_latency(
                block,
                analog_macs[block] * hardware.costs.analog_full.energy_pj_per_mac,
                analog_macs[block] * hardware.costs.analog_full.latency_ns_per_mac,
            )

            e_add, t_add = _verify_additional_cost_for_block(hardware, executed_precision, token_kind="drafted")
            verify_drafted = verify_drafted.add_energy_latency(
                block,
                analog_macs[block] * e_add,
                analog_macs[block] * t_add,
            )

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

        for feature in enabled_dpu_features:
            e_per, t_per = dpu_feature_costs[feature]
            ops = float(dpu_feature_ops[feature])
            parallel_units = hardware.soc.attention_cim_units if feature in {"attention_qk", "attention_pv"} else 1
            energy = ops * e_per
            latency = (ops * t_per) / float(max(parallel_units, 1))
            stage = DPU_STAGE_BY_FEATURE[feature]
            draft = draft.add_energy_latency(stage, energy, latency)
            verify_drafted = verify_drafted.add_energy_latency(stage, energy, latency)
            verify_bonus = verify_bonus.add_energy_latency(stage, energy, latency)

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


def _sum_breakdowns(lhs: Breakdown, rhs: Breakdown) -> Breakdown:
    stages = lhs.stages.plus(rhs.stages)

    components = None
    if lhs.components is not None and rhs.components is not None:
        components = lhs.components.plus(rhs.components)
    elif lhs.components is not None:
        components = lhs.components
    elif rhs.components is not None:
        components = rhs.components

    activation_counts = None
    if lhs.activation_counts is not None and rhs.activation_counts is not None:
        activation_counts = lhs.activation_counts.plus(rhs.activation_counts)
    elif lhs.activation_counts is not None:
        activation_counts = lhs.activation_counts
    elif rhs.activation_counts is not None:
        activation_counts = rhs.activation_counts

    memory_traffic = None
    if lhs.memory_traffic is not None and rhs.memory_traffic is not None:
        memory_traffic = lhs.memory_traffic.plus(rhs.memory_traffic)
    elif lhs.memory_traffic is not None:
        memory_traffic = lhs.memory_traffic
    elif rhs.memory_traffic is not None:
        memory_traffic = rhs.memory_traffic

    dpu_features = None
    if lhs.dpu_features is not None and rhs.dpu_features is not None:
        dpu_features = lhs.dpu_features.plus(rhs.dpu_features)
    elif lhs.dpu_features is not None:
        dpu_features = lhs.dpu_features
    elif rhs.dpu_features is not None:
        dpu_features = rhs.dpu_features

    channels = None
    if lhs.channels is not None and rhs.channels is not None:
        channels = lhs.channels.plus(rhs.channels)
    elif lhs.channels is not None:
        channels = lhs.channels
    elif rhs.channels is not None:
        channels = rhs.channels

    return Breakdown.from_stage_breakdown(
        stages,
        components=components,
        activation_counts=activation_counts,
        memory_traffic=memory_traffic,
        dpu_features=dpu_features,
        channels=channels,
    )


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
        draft_template, _ = _token_step_costs_legacy(model, hardware, l_prompt)
        verify_drafted_template = _verify_drafted_token_additional_stage_legacy(model, hardware, l_prompt)

        def step_costs(context_len: int) -> tuple[Breakdown, Breakdown, Breakdown]:
            draft_step, verify_full_step = _token_step_costs_legacy(model, hardware, context_len)
            verify_drafted_step = _verify_drafted_token_additional_stage_legacy(model, hardware, context_len)
            return draft_step, verify_drafted_step, verify_full_step

    else:
        specs = hardware.resolve_knob_specs()
        draft_template, _ = _token_step_costs_knob(model, hardware, specs, l_prompt)
        verify_drafted_template = _verify_drafted_token_additional_stage_knob(model, hardware, specs, l_prompt)

        def step_costs(context_len: int) -> tuple[Breakdown, Breakdown, Breakdown]:
            draft_step, verify_full_step = _token_step_costs_knob(model, hardware, specs, context_len)
            verify_drafted_step = _verify_drafted_token_additional_stage_knob(model, hardware, specs, context_len)
            return draft_step, verify_drafted_step, verify_full_step

    hist = normalize_histogram(stats.histogram)

    draft_phase = draft_template.scale(0.0)
    verify_drafted_steps: list[Breakdown] = []
    for i in range(stats.k):
        context_len = l_prompt + i
        draft_step_i, verify_drafted_step_i, _verify_full_step_i = step_costs(context_len)
        draft_phase = _sum_breakdowns(draft_phase, draft_step_i)
        verify_drafted_steps.append(verify_drafted_step_i)

    _, _, verify_bonus_full_step = step_costs(l_prompt + stats.k)

    if hardware.soc.schedule == ScheduleMode.layer_pipelined:
        verify_drafted_phase = verify_drafted_template.scale(0.0)
        for i, verify_step_i in enumerate(verify_drafted_steps):
            p_execute_i = sum(prob for accepted_prefix, prob in hist.items() if accepted_prefix >= i)
            verify_drafted_phase = _sum_breakdowns(verify_drafted_phase, verify_step_i.scale(p_execute_i))

        p_full_accept = hist.get(stats.k, 0.0)
        verify_setup_phase = _verify_setup_breakdown(model, hardware)
        verify_bonus_phase = _sum_breakdowns(
            verify_setup_phase.scale(1.0 - p_full_accept),
            verify_bonus_full_step.scale(p_full_accept),
        )
    else:
        verify_drafted_phase = verify_drafted_template.scale(0.0)
        for verify_step_i in verify_drafted_steps:
            verify_drafted_phase = _sum_breakdowns(verify_drafted_phase, verify_step_i)
        verify_bonus_phase = verify_bonus_full_step

    committed = expected_committed_tokens_per_burst(stats)
    if committed <= 0:
        raise ValueError("Expected committed tokens per burst must be > 0")

    if hardware.memory is not None:
        if hardware.soc.schedule == ScheduleMode.layer_pipelined:
            draft_traffic = MemoryTraffic()
            verify_drafted_traffic = MemoryTraffic()
            for i in range(stats.k):
                step_traffic_i = _kv_memory_step_traffic(
                    model=model,
                    hardware=hardware,
                    prompt_tokens_from_hbm=float(l_prompt),
                    speculative_tokens_from_sram=float(i),
                    speculative_tokens_to_sram=1.0,
                    committed_tokens_to_hbm=0.0,
                )
                draft_traffic = draft_traffic.plus(step_traffic_i)

                p_execute_i = sum(prob for accepted_prefix, prob in hist.items() if accepted_prefix >= i)
                verify_drafted_traffic = verify_drafted_traffic.plus(step_traffic_i.scale(p_execute_i))

            p_full_accept = hist.get(stats.k, 0.0)
            bonus_step_traffic = _kv_memory_step_traffic(
                model=model,
                hardware=hardware,
                prompt_tokens_from_hbm=float(l_prompt),
                speculative_tokens_from_sram=float(stats.k),
                speculative_tokens_to_sram=1.0,
                committed_tokens_to_hbm=0.0,
            )
            commit_traffic = _kv_memory_step_traffic(
                model=model,
                hardware=hardware,
                prompt_tokens_from_hbm=0.0,
                speculative_tokens_from_sram=0.0,
                speculative_tokens_to_sram=0.0,
                committed_tokens_to_hbm=committed,
            )
            verify_bonus_traffic = bonus_step_traffic.scale(p_full_accept).plus(commit_traffic)
        else:
            traffic = _kv_memory_traffic_by_phase(model=model, hardware=hardware, stats=stats, l_prompt=l_prompt)
            draft_traffic = traffic["draft"]
            verify_drafted_traffic = traffic["verify_drafted"]
            verify_bonus_traffic = traffic["verify_bonus"]

        draft_phase = _add_memory_traffic_costs(breakdown=draft_phase, traffic=draft_traffic, hardware=hardware)
        verify_drafted_phase = _add_memory_traffic_costs(
            breakdown=verify_drafted_phase,
            traffic=verify_drafted_traffic,
            hardware=hardware,
        )
        verify_bonus_phase = _add_memory_traffic_costs(
            breakdown=verify_bonus_phase,
            traffic=verify_bonus_traffic,
            hardware=hardware,
        )

    total_phase = _sum_breakdowns(_sum_breakdowns(draft_phase, verify_drafted_phase), verify_bonus_phase)
    dynamic_burst_energy_pj = total_phase.energy_pj
    effective_burst_latency_ns = total_phase.latency_ns

    if hardware.soc.schedule == ScheduleMode.layer_pipelined:
        if hardware.mode == HardwareMode.legacy:

            def max_layer_latencies(context_len: int) -> tuple[float, float, float]:
                return _max_layer_compute_latencies_ns_legacy(
                    model=model,
                    hardware=hardware,
                    l_prompt=context_len,
                )

        else:
            specs = hardware.resolve_knob_specs()

            def max_layer_latencies(context_len: int) -> tuple[float, float, float]:
                return _max_layer_compute_latencies_ns_knob(
                    model=model,
                    hardware=hardware,
                    specs=specs,
                    l_prompt=context_len,
                )

        verify_step_periods: list[float] = []
        for i in range(stats.k):
            context_len = l_prompt + i
            _max_draft_step, max_verify_drafted_step, _max_verify_bonus_step = max_layer_latencies(context_len)
            mem_verify_step = 0.0
            if hardware.memory is not None:
                verify_step_traffic = _kv_memory_step_traffic(
                    model=model,
                    hardware=hardware,
                    prompt_tokens_from_hbm=float(l_prompt),
                    speculative_tokens_from_sram=float(i),
                    speculative_tokens_to_sram=1.0,
                    committed_tokens_to_hbm=0.0,
                )
                _mem_energy, mem_verify_step = _memory_cost_from_traffic(hardware=hardware, traffic=verify_step_traffic)
            verify_step_periods.append(max(max_verify_drafted_step, mem_verify_step))

        _unused_draft, _unused_verify_drafted, max_verify_bonus_step = max_layer_latencies(l_prompt + stats.k)
        verify_setup_period = hardware.soc.control.latency_ns_per_burst + hardware.soc.verify_setup.latency_ns_per_burst
        draft_serial_latency = draft_phase.latency_ns

        t_burst_pipe = 0.0
        for accepted_prefix, prob in hist.items():
            executed_steps = _executed_verify_drafted_steps_for_outcome(k=stats.k, accepted_prefix=accepted_prefix)
            verify_wavefront_latency = sum(verify_step_periods[:executed_steps])
            committed_tokens_outcome = accepted_prefix + 1

            if _executes_verify_bonus_for_outcome(k=stats.k, accepted_prefix=accepted_prefix):
                mem_bonus = 0.0
                if hardware.memory is not None:
                    bonus_traffic = _kv_memory_step_traffic(
                        model=model,
                        hardware=hardware,
                        prompt_tokens_from_hbm=float(l_prompt),
                        speculative_tokens_from_sram=float(stats.k),
                        speculative_tokens_to_sram=1.0,
                        committed_tokens_to_hbm=float(committed_tokens_outcome),
                    )
                    _mem_energy, mem_bonus = _memory_cost_from_traffic(hardware=hardware, traffic=bonus_traffic)
                tail_latency = max(max_verify_bonus_step, mem_bonus)
            else:
                mem_commit = 0.0
                if hardware.memory is not None:
                    commit_traffic_outcome = _kv_memory_step_traffic(
                        model=model,
                        hardware=hardware,
                        prompt_tokens_from_hbm=0.0,
                        speculative_tokens_from_sram=0.0,
                        speculative_tokens_to_sram=0.0,
                        committed_tokens_to_hbm=float(committed_tokens_outcome),
                    )
                    _mem_energy, mem_commit = _memory_cost_from_traffic(hardware=hardware, traffic=commit_traffic_outcome)
                tail_latency = max(verify_setup_period, mem_commit)

            t_burst_pipe += prob * (draft_serial_latency + verify_wavefront_latency + tail_latency)

        effective_burst_latency_ns = t_burst_pipe

    leakage_power_nw = _total_leakage_power_nw(hardware)
    leakage_energy_pj = _leakage_energy_pj(
        leakage_power_nw=leakage_power_nw,
        burst_latency_ns=effective_burst_latency_ns,
    )

    total_burst_energy_pj = dynamic_burst_energy_pj + leakage_energy_pj
    energy_per_token_pj = total_burst_energy_pj / committed
    latency_per_token_ns = effective_burst_latency_ns / committed

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

    committed_tokens = expected_committed_tokens_per_burst(stats)
    leakage_power_nw = _total_leakage_power_nw(hardware)

    points: list[SweepPoint] = []
    for l_prompt in prompt_lengths:
        speculative_metrics, speculative_breakdown = estimate_point(model, hardware, stats, l_prompt)
        baseline_metrics, baseline_breakdown = estimate_point(model, hardware, _baseline_stats(), l_prompt)
        delta = BaselineDelta.from_metrics(speculative_metrics, baseline_metrics)
        burst_latency_ns = speculative_metrics.latency_ns_per_token * committed_tokens
        leakage_energy_pj = _leakage_energy_pj(
            leakage_power_nw=leakage_power_nw,
            burst_latency_ns=burst_latency_ns,
        )
        points.append(
            SweepPoint(
                l_prompt=l_prompt,
                speculative=speculative_metrics,
                baseline=baseline_metrics,
                delta=delta,
                breakdown=speculative_breakdown,
                baseline_breakdown=baseline_breakdown,
                leakage=LeakageSummary(
                    total_power_nw=leakage_power_nw,
                    energy_pj=leakage_energy_pj,
                    burst_latency_ns=burst_latency_ns,
                ),
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
        any(getattr(hardware.leakage_power, field) != 0.0 for field in type(hardware.leakage_power).model_fields)
        or bool(hardware.leakage_power.model_fields_set)
    ):
        hardware_knobs["leakage_power"] = hardware.leakage_power.model_dump(mode="json")

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

    if hardware.mode == HardwareMode.legacy:
        _, dpu_feature_mapping = _dpu_feature_costs_legacy(hardware)
    else:
        specs = hardware.resolve_knob_specs()
        _, dpu_feature_mapping = _dpu_feature_costs_knob(specs)

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
        dpu_feature_mapping=dpu_feature_mapping,
        movement_accounting=MOVEMENT_ACCOUNTING_COVERAGE,
        pipeline_policy=_pipeline_policy_metadata(hardware.soc.schedule),
        notes=[
            "Analytical calculator (closed-form activation counts, no event simulation).",
            "Serialized phase accounting is step-indexed with context growth L_i = L_prompt + i.",
            "Layer-pipelined mode keeps draft serialized and uses outcome-conditioned verify wavefront execution.",
            "Layer-pipelined verify stops at first mismatch; full acceptance executes and commits one bonus token.",
            "Leakage energy uses E_leak_burst[pJ] = P_leak_total[nW] * T_burst_effective[ns] * 1e-6.",
            "Leakage timing is schedule-aware: serialized uses serialized burst time, layer-pipelined uses pipelined burst time.",
            "Breakdown latencies are serialized sums; `soc.schedule` affects how latency/token is reported.",
            "Movement exclusions are explicit in `movement_accounting.excluded`.",
        ],
    )

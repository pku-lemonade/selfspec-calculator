from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .config import InputPaths


class Metrics(BaseModel):
    energy_pj_per_token: float = Field(..., ge=0.0)
    latency_ns_per_token: float = Field(..., ge=0.0)
    throughput_tokens_per_s: float = Field(..., ge=0.0)
    tokens_per_joule: float = Field(..., ge=0.0)


class BaselineDelta(BaseModel):
    energy_pj_per_token_ratio: float | None = None
    latency_ns_per_token_ratio: float | None = None
    throughput_tokens_per_s_ratio: float | None = None
    tokens_per_joule_ratio: float | None = None

    energy_pj_per_token_diff: float | None = None
    latency_ns_per_token_diff: float | None = None
    throughput_tokens_per_s_diff: float | None = None
    tokens_per_joule_diff: float | None = None

    @staticmethod
    def _ratio(a: float, b: float) -> float | None:
        if b == 0:
            return None
        return a / b

    @classmethod
    def from_metrics(cls, speculative: Metrics, baseline: Metrics) -> "BaselineDelta":
        return cls(
            energy_pj_per_token_ratio=cls._ratio(speculative.energy_pj_per_token, baseline.energy_pj_per_token),
            latency_ns_per_token_ratio=cls._ratio(speculative.latency_ns_per_token, baseline.latency_ns_per_token),
            throughput_tokens_per_s_ratio=cls._ratio(speculative.throughput_tokens_per_s, baseline.throughput_tokens_per_s),
            tokens_per_joule_ratio=cls._ratio(speculative.tokens_per_joule, baseline.tokens_per_joule),
            energy_pj_per_token_diff=speculative.energy_pj_per_token - baseline.energy_pj_per_token,
            latency_ns_per_token_diff=speculative.latency_ns_per_token - baseline.latency_ns_per_token,
            throughput_tokens_per_s_diff=speculative.throughput_tokens_per_s - baseline.throughput_tokens_per_s,
            tokens_per_joule_diff=speculative.tokens_per_joule - baseline.tokens_per_joule,
        )


class StageBreakdown(BaseModel):
    qkv_energy_pj: float = 0.0
    qkv_latency_ns: float = 0.0
    wo_energy_pj: float = 0.0
    wo_latency_ns: float = 0.0
    ffn_energy_pj: float = 0.0
    ffn_latency_ns: float = 0.0

    qk_energy_pj: float = 0.0
    qk_latency_ns: float = 0.0
    pv_energy_pj: float = 0.0
    pv_latency_ns: float = 0.0
    softmax_energy_pj: float = 0.0
    softmax_latency_ns: float = 0.0
    elementwise_energy_pj: float = 0.0
    elementwise_latency_ns: float = 0.0
    kv_cache_energy_pj: float = 0.0
    kv_cache_latency_ns: float = 0.0
    buffers_add_energy_pj: float = 0.0
    buffers_add_latency_ns: float = 0.0
    control_energy_pj: float = 0.0
    control_latency_ns: float = 0.0

    qkv_mm2: float = 0.0
    wo_mm2: float = 0.0
    ffn_mm2: float = 0.0
    digital_mm2: float = 0.0

    def plus(self, other: "StageBreakdown") -> "StageBreakdown":
        return self.model_copy(
            update={field: getattr(self, field) + getattr(other, field) for field in type(self).model_fields}
        )

    def add_energy_latency(self, stage: str, energy_pj: float, latency_ns: float) -> "StageBreakdown":
        update: dict[str, Any] = {}
        if stage in {"qkv", "wo", "ffn", "qk", "pv", "softmax", "elementwise", "kv_cache", "buffers_add", "control"}:
            update[f"{stage}_energy_pj"] = getattr(self, f"{stage}_energy_pj") + energy_pj
            update[f"{stage}_latency_ns"] = getattr(self, f"{stage}_latency_ns") + latency_ns
            return self.model_copy(update=update)
        raise KeyError(stage)


class AreaComponentsMm2(BaseModel):
    arrays_mm2: float = Field(0.0, ge=0.0)
    dac_mm2: float = Field(0.0, ge=0.0)
    adc_draft_mm2: float = Field(0.0, ge=0.0)
    adc_residual_mm2: float = Field(0.0, ge=0.0)
    attention_cim_sram_mm2: float = Field(0.0, ge=0.0)
    attention_cim_mac_mm2: float = Field(0.0, ge=0.0)

    tia_mm2: float = Field(0.0, ge=0.0)
    snh_mm2: float = Field(0.0, ge=0.0)
    mux_mm2: float = Field(0.0, ge=0.0)
    io_buffers_mm2: float = Field(0.0, ge=0.0)
    subarray_switches_mm2: float = Field(0.0, ge=0.0)
    write_drivers_mm2: float = Field(0.0, ge=0.0)

    sram_mm2: float = Field(0.0, ge=0.0)
    fabric_mm2: float = Field(0.0, ge=0.0)
    digital_overhead_mm2: float = Field(0.0, ge=0.0)


class AreaBreakdownMm2(BaseModel):
    on_chip_mm2: float = Field(0.0, ge=0.0)
    off_chip_hbm_mm2: float = Field(0.0, ge=0.0)
    on_chip_components: AreaComponentsMm2 = Field(default_factory=AreaComponentsMm2)


class ComponentBreakdown(BaseModel):
    arrays_energy_pj: float = 0.0
    arrays_latency_ns: float = 0.0
    dac_energy_pj: float = 0.0
    dac_latency_ns: float = 0.0
    adc_draft_energy_pj: float = 0.0
    adc_draft_latency_ns: float = 0.0
    adc_residual_energy_pj: float = 0.0
    adc_residual_latency_ns: float = 0.0
    tia_energy_pj: float = 0.0
    tia_latency_ns: float = 0.0
    snh_energy_pj: float = 0.0
    snh_latency_ns: float = 0.0
    mux_energy_pj: float = 0.0
    mux_latency_ns: float = 0.0
    io_buffers_energy_pj: float = 0.0
    io_buffers_latency_ns: float = 0.0
    subarray_switches_energy_pj: float = 0.0
    subarray_switches_latency_ns: float = 0.0
    write_drivers_energy_pj: float = 0.0
    write_drivers_latency_ns: float = 0.0

    attention_engine_energy_pj: float = 0.0
    attention_engine_latency_ns: float = 0.0
    kv_cache_energy_pj: float = 0.0
    kv_cache_latency_ns: float = 0.0
    sram_energy_pj: float = 0.0
    sram_latency_ns: float = 0.0
    hbm_energy_pj: float = 0.0
    hbm_latency_ns: float = 0.0
    fabric_energy_pj: float = 0.0
    fabric_latency_ns: float = 0.0
    softmax_unit_energy_pj: float = 0.0
    softmax_unit_latency_ns: float = 0.0
    elementwise_unit_energy_pj: float = 0.0
    elementwise_unit_latency_ns: float = 0.0
    buffers_add_energy_pj: float = 0.0
    buffers_add_latency_ns: float = 0.0
    control_energy_pj: float = 0.0
    control_latency_ns: float = 0.0

    def plus(self, other: "ComponentBreakdown") -> "ComponentBreakdown":
        return self.model_copy(
            update={field: getattr(self, field) + getattr(other, field) for field in type(self).model_fields}
        )

    def add_energy_latency(self, component: str, energy_pj: float, latency_ns: float) -> "ComponentBreakdown":
        prefix = {
            "arrays": "arrays",
            "dac": "dac",
            "adc_draft": "adc_draft",
            "adc_residual": "adc_residual",
            "tia": "tia",
            "snh": "snh",
            "mux": "mux",
            "io_buffers": "io_buffers",
            "subarray_switches": "subarray_switches",
            "write_drivers": "write_drivers",
            "attention_engine": "attention_engine",
            "kv_cache": "kv_cache",
            "sram": "sram",
            "hbm": "hbm",
            "fabric": "fabric",
            "softmax_unit": "softmax_unit",
            "elementwise_unit": "elementwise_unit",
            "buffers_add": "buffers_add",
            "control": "control",
        }.get(component)
        if prefix is None:
            raise KeyError(component)
        return self.model_copy(
            update={
                f"{prefix}_energy_pj": getattr(self, f"{prefix}_energy_pj") + energy_pj,
                f"{prefix}_latency_ns": getattr(self, f"{prefix}_latency_ns") + latency_ns,
            }
        )


class AnalogActivationCounts(BaseModel):
    array_activations: float = Field(0.0, ge=0.0)
    dac_conversions: float = Field(0.0, ge=0.0)
    adc_draft_conversions: float = Field(0.0, ge=0.0)
    adc_residual_conversions: float = Field(0.0, ge=0.0)

    def plus(self, other: "AnalogActivationCounts") -> "AnalogActivationCounts":
        return self.model_copy(
            update={field: getattr(self, field) + getattr(other, field) for field in type(self).model_fields}
        )

    def scale(self, factor: float) -> "AnalogActivationCounts":
        return self.model_copy(
            update={field: getattr(self, field) * factor for field in type(self).model_fields}
        )


class MemoryTraffic(BaseModel):
    sram_read_bytes: float = Field(0.0, ge=0.0)
    sram_write_bytes: float = Field(0.0, ge=0.0)
    hbm_read_bytes: float = Field(0.0, ge=0.0)
    hbm_write_bytes: float = Field(0.0, ge=0.0)
    fabric_read_bytes: float = Field(0.0, ge=0.0)
    fabric_write_bytes: float = Field(0.0, ge=0.0)

    def plus(self, other: "MemoryTraffic") -> "MemoryTraffic":
        return self.model_copy(
            update={field: getattr(self, field) + getattr(other, field) for field in type(self).model_fields}
        )

    def scale(self, factor: float) -> "MemoryTraffic":
        return self.model_copy(update={field: getattr(self, field) * factor for field in type(self).model_fields})


class CostChannelBreakdown(BaseModel):
    compute_energy_pj: float = Field(0.0, ge=0.0)
    compute_latency_ns: float = Field(0.0, ge=0.0)
    movement_energy_pj: float = Field(0.0, ge=0.0)
    movement_latency_ns: float = Field(0.0, ge=0.0)

    def plus(self, other: "CostChannelBreakdown") -> "CostChannelBreakdown":
        return self.model_copy(
            update={field: getattr(self, field) + getattr(other, field) for field in type(self).model_fields}
        )

    def scale(self, factor: float) -> "CostChannelBreakdown":
        return self.model_copy(update={field: getattr(self, field) * factor for field in type(self).model_fields})

    def add_energy_latency(self, channel: str, energy_pj: float, latency_ns: float) -> "CostChannelBreakdown":
        if channel == "compute":
            return self.model_copy(
                update={
                    "compute_energy_pj": self.compute_energy_pj + energy_pj,
                    "compute_latency_ns": self.compute_latency_ns + latency_ns,
                }
            )
        if channel == "movement":
            return self.model_copy(
                update={
                    "movement_energy_pj": self.movement_energy_pj + energy_pj,
                    "movement_latency_ns": self.movement_latency_ns + latency_ns,
                }
            )
        raise KeyError(channel)


class DpuFeatureBreakdown(BaseModel):
    attention_qk_ops: float = Field(0.0, ge=0.0)
    attention_qk_energy_pj: float = Field(0.0, ge=0.0)
    attention_qk_latency_ns: float = Field(0.0, ge=0.0)

    attention_softmax_ops: float = Field(0.0, ge=0.0)
    attention_softmax_energy_pj: float = Field(0.0, ge=0.0)
    attention_softmax_latency_ns: float = Field(0.0, ge=0.0)

    attention_pv_ops: float = Field(0.0, ge=0.0)
    attention_pv_energy_pj: float = Field(0.0, ge=0.0)
    attention_pv_latency_ns: float = Field(0.0, ge=0.0)

    ffn_activation_ops: float = Field(0.0, ge=0.0)
    ffn_activation_energy_pj: float = Field(0.0, ge=0.0)
    ffn_activation_latency_ns: float = Field(0.0, ge=0.0)

    ffn_gate_multiply_ops: float = Field(0.0, ge=0.0)
    ffn_gate_multiply_energy_pj: float = Field(0.0, ge=0.0)
    ffn_gate_multiply_latency_ns: float = Field(0.0, ge=0.0)

    kv_cache_update_ops: float = Field(0.0, ge=0.0)
    kv_cache_update_energy_pj: float = Field(0.0, ge=0.0)
    kv_cache_update_latency_ns: float = Field(0.0, ge=0.0)

    def plus(self, other: "DpuFeatureBreakdown") -> "DpuFeatureBreakdown":
        return self.model_copy(
            update={field: getattr(self, field) + getattr(other, field) for field in type(self).model_fields}
        )

    def scale(self, factor: float) -> "DpuFeatureBreakdown":
        return self.model_copy(update={field: getattr(self, field) * factor for field in type(self).model_fields})

    def add(self, feature: str, *, ops: float, energy_pj: float, latency_ns: float) -> "DpuFeatureBreakdown":
        if feature not in {
            "attention_qk",
            "attention_softmax",
            "attention_pv",
            "ffn_activation",
            "ffn_gate_multiply",
            "kv_cache_update",
        }:
            raise KeyError(feature)
        return self.model_copy(
            update={
                f"{feature}_ops": getattr(self, f"{feature}_ops") + ops,
                f"{feature}_energy_pj": getattr(self, f"{feature}_energy_pj") + energy_pj,
                f"{feature}_latency_ns": getattr(self, f"{feature}_latency_ns") + latency_ns,
            }
        )


class MovementAccountingCoverage(BaseModel):
    modeled: list[str] = Field(default_factory=list)
    proxy_modeled: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    ownership_rules: dict[str, str] = Field(default_factory=dict)


class Breakdown(BaseModel):
    energy_pj: float = Field(..., ge=0.0)
    latency_ns: float = Field(..., ge=0.0)
    stages: StageBreakdown
    components: ComponentBreakdown | None = None
    activation_counts: AnalogActivationCounts | None = None
    memory_traffic: MemoryTraffic | None = None
    dpu_features: DpuFeatureBreakdown | None = None
    channels: CostChannelBreakdown | None = None

    @classmethod
    def from_stage_breakdown(
        cls,
        stages: StageBreakdown,
        components: ComponentBreakdown | None = None,
        activation_counts: AnalogActivationCounts | None = None,
        memory_traffic: MemoryTraffic | None = None,
        dpu_features: DpuFeatureBreakdown | None = None,
        channels: CostChannelBreakdown | None = None,
    ) -> "Breakdown":
        energy = sum(
            getattr(stages, f"{s}_energy_pj")
            for s in ["qkv", "wo", "ffn", "qk", "pv", "softmax", "elementwise", "kv_cache", "buffers_add", "control"]
        )
        latency = sum(
            getattr(stages, f"{s}_latency_ns")
            for s in ["qkv", "wo", "ffn", "qk", "pv", "softmax", "elementwise", "kv_cache", "buffers_add", "control"]
        )
        return cls(
            energy_pj=energy,
            latency_ns=latency,
            stages=stages,
            components=components,
            activation_counts=activation_counts,
            memory_traffic=memory_traffic,
            dpu_features=dpu_features,
            channels=channels,
        )

    def scale(self, factor: float) -> "Breakdown":
        components = None
        if self.components is not None:
            components = self.components.model_copy(
                update={
                    field: getattr(self.components, field) * factor for field in type(self.components).model_fields
                }
            )
        activation_counts = None
        if self.activation_counts is not None:
            activation_counts = self.activation_counts.scale(factor)
        memory_traffic = None
        if self.memory_traffic is not None:
            memory_traffic = self.memory_traffic.scale(factor)
        dpu_features = None
        if self.dpu_features is not None:
            dpu_features = self.dpu_features.scale(factor)
        channels = None
        if self.channels is not None:
            channels = self.channels.scale(factor)
        return Breakdown(
            energy_pj=self.energy_pj * factor,
            latency_ns=self.latency_ns * factor,
            stages=self.stages.model_copy(
                update={
                    f"{s}_{m}": getattr(self.stages, f"{s}_{m}") * factor
                    for s in [
                        "qkv",
                        "wo",
                        "ffn",
                        "qk",
                        "pv",
                        "softmax",
                        "elementwise",
                        "kv_cache",
                        "buffers_add",
                        "control",
                    ]
                    for m in ["energy_pj", "latency_ns"]
                }
            ),
            components=components,
            activation_counts=activation_counts,
            memory_traffic=memory_traffic,
            dpu_features=dpu_features,
            channels=channels,
        )


class PhaseBreakdown(BaseModel):
    draft: Breakdown
    verify_drafted: Breakdown
    verify_bonus: Breakdown
    total: Breakdown


class SweepPoint(BaseModel):
    l_prompt: int = Field(..., ge=0)
    speculative: Metrics
    baseline: Metrics
    delta: BaselineDelta
    breakdown: PhaseBreakdown
    baseline_breakdown: PhaseBreakdown


class Report(BaseModel):
    generated_at: str
    k: int = Field(..., ge=0)
    reuse_policy: str
    hardware_mode: str
    resolved_library: dict[str, Any] | None = None
    model_knobs: dict[str, Any] | None = None
    hardware_knobs: dict[str, Any] | None = None
    paths: InputPaths | None = None
    points: list[SweepPoint]
    break_even_tokens_per_joule_l_prompt: int | None = None
    area: StageBreakdown
    area_breakdown_mm2: AreaBreakdownMm2
    dpu_feature_mapping: dict[str, str] | None = None
    movement_accounting: MovementAccountingCoverage | None = None
    pipeline_policy: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)

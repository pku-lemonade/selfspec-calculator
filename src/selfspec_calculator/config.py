from __future__ import annotations

from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class FfnType(str, Enum):
    mlp = "mlp"
    swiglu = "swiglu"


class PrecisionMode(str, Enum):
    draft = "draft"
    full = "full"


class BlockDraftPolicy(BaseModel):
    qkv: PrecisionMode = PrecisionMode.draft
    wo: PrecisionMode = PrecisionMode.draft
    ffn: PrecisionMode = PrecisionMode.draft


class DraftPrecisionPolicy(BaseModel):
    default: BlockDraftPolicy = Field(default_factory=BlockDraftPolicy)
    per_layer: dict[int, BlockDraftPolicy] = Field(default_factory=dict)

    def for_layer(self, layer: int) -> BlockDraftPolicy:
        return self.per_layer.get(layer, self.default)


class ModelConfig(BaseModel):
    name: str | None = None
    n_layers: int = Field(..., ge=1)
    d_model: int = Field(..., ge=1)
    n_heads: int = Field(..., ge=1)
    activation_bits: int = Field(..., ge=1)
    ffn_type: FfnType = FfnType.mlp
    d_ff: int | None = Field(default=None, ge=1)
    ffn_expansion: float | None = Field(default=4.0, ge=1.0)
    draft_policy: DraftPrecisionPolicy = Field(default_factory=DraftPrecisionPolicy)

    @field_validator("draft_policy")
    @classmethod
    def _validate_draft_policy(cls, v: DraftPrecisionPolicy, info):  # noqa: ANN001
        n_layers = info.data.get("n_layers")
        if n_layers is None:
            return v
        for layer in v.per_layer.keys():
            if layer < 0 or layer >= n_layers:
                raise ValueError(f"draft_policy.per_layer has invalid layer index: {layer} (n_layers={n_layers})")
        return v

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        return self.d_model // self.n_heads

    @property
    def effective_d_ff(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        if self.ffn_expansion is None:
            raise ValueError("Either d_ff or ffn_expansion must be provided")
        return int(round(self.d_model * self.ffn_expansion))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        data = _load_yaml(path)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid model config: {path}\n{exc}") from exc


class ReusePolicy(str, Enum):
    reuse = "reuse"
    reread = "reread"


class HardwareMode(str, Enum):
    legacy = "legacy"
    knob_based = "knob-based"


class ScheduleMode(str, Enum):
    serialized = "serialized"
    layer_pipelined = "layer-pipelined"


class PerMacCost(BaseModel):
    energy_pj_per_mac: float = Field(..., ge=0.0)
    latency_ns_per_mac: float = Field(..., ge=0.0)


class PerWeightArea(BaseModel):
    area_mm2_per_weight: float = Field(..., ge=0.0)


class DpuFeatureCostOverrides(BaseModel):
    attention_qk: PerMacCost | None = None
    attention_softmax: PerMacCost | None = None
    attention_pv: PerMacCost | None = None
    ffn_activation: PerMacCost | None = None
    ffn_gate_multiply: PerMacCost | None = None
    kv_cache_update: PerMacCost | None = None


class DpuFeatureCosts(BaseModel):
    attention_qk: PerMacCost
    attention_softmax: PerMacCost
    attention_pv: PerMacCost
    ffn_activation: PerMacCost
    ffn_gate_multiply: PerMacCost
    kv_cache_update: PerMacCost


class HardwareCosts(BaseModel):
    analog_draft: PerMacCost
    analog_full: PerMacCost
    analog_verify_reuse: PerMacCost
    digital_attention: PerMacCost
    digital_softmax: PerMacCost
    digital_elementwise: PerMacCost
    kv_cache: PerMacCost
    digital_features: DpuFeatureCostOverrides | None = None
    analog_weight_area: PerWeightArea
    digital_overhead_area_mm2_per_layer: float = Field(0.0, ge=0.0)


class AdcResolutionConfig(BaseModel):
    draft_bits: int = Field(..., ge=1)
    residual_bits: int = Field(..., ge=1)


class PerOpOverheadSpec(BaseModel):
    energy_pj_per_op: float = Field(0.0, ge=0.0)
    latency_ns_per_op: float = Field(0.0, ge=0.0)
    area_mm2_per_unit: float = Field(0.0, ge=0.0)


class AnalogPeripheryKnobs(BaseModel):
    tia: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    snh: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    mux: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    io_buffers: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    subarray_switches: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    write_drivers: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)


class AnalogKnobs(BaseModel):
    xbar_size: int = Field(..., ge=1)
    num_columns_per_adc: int = Field(..., ge=1)
    dac_bits: int = Field(..., ge=1)
    adc: AdcResolutionConfig
    periphery: AnalogPeripheryKnobs = Field(default_factory=AnalogPeripheryKnobs)

    @model_validator(mode="after")
    def _validate_divisibility(self) -> "AnalogKnobs":
        if self.xbar_size % self.num_columns_per_adc != 0:
            raise ValueError(
                f"analog.xbar_size ({self.xbar_size}) must be divisible by "
                f"analog.num_columns_per_adc ({self.num_columns_per_adc})"
            )
        return self


class PeripheralSpec(BaseModel):
    energy_pj_per_conversion: float = Field(..., ge=0.0)
    latency_ns_per_conversion: float = Field(..., ge=0.0)
    area_mm2_per_unit: float = Field(..., ge=0.0)


class AnalogArraySpec(BaseModel):
    energy_pj_per_activation: float = Field(..., ge=0.0)
    latency_ns_per_activation: float = Field(..., ge=0.0)
    area_mm2_per_weight: float = Field(..., ge=0.0)
    area_mm2_per_array: float | None = Field(default=None, ge=0.0)
    arrays_per_weight: int = Field(1, ge=1)


class VerifySetupKnobs(BaseModel):
    energy_pj_per_burst: float = Field(0.0, ge=0.0)
    latency_ns_per_burst: float = Field(0.0, ge=0.0)


class ControlOverheadKnobs(BaseModel):
    energy_pj_per_token: float = Field(0.0, ge=0.0)
    latency_ns_per_token: float = Field(0.0, ge=0.0)
    energy_pj_per_burst: float = Field(0.0, ge=0.0)
    latency_ns_per_burst: float = Field(0.0, ge=0.0)


class SocKnobs(BaseModel):
    schedule: ScheduleMode = ScheduleMode.serialized
    attention_cim_units: int = Field(1, ge=1)
    attention_cim_mac_area_mm2_per_unit: float = Field(0.0, ge=0.0)
    attention_cim_storage_bits_per_element: int | None = Field(default=None, ge=1)
    verify_setup: VerifySetupKnobs = Field(default_factory=VerifySetupKnobs)
    buffers_add: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    control: ControlOverheadKnobs = Field(default_factory=ControlOverheadKnobs)


class MemoryTechKnobs(BaseModel):
    read_energy_pj_per_byte: float = Field(0.0, ge=0.0)
    write_energy_pj_per_byte: float = Field(0.0, ge=0.0)
    read_bandwidth_GBps: float = Field(0.0, ge=0.0)
    write_bandwidth_GBps: float = Field(0.0, ge=0.0)
    read_latency_ns: float = Field(0.0, ge=0.0)
    write_latency_ns: float = Field(0.0, ge=0.0)
    area_mm2: float = Field(0.0, ge=0.0)
    capacity_bytes: int | None = Field(default=None, ge=1)


class KvCacheFormatKnobs(BaseModel):
    value_bytes_per_elem: int = Field(1, ge=0)
    scale_bytes: int = Field(2, ge=0)
    scales_per_token_per_head: int = Field(2, ge=0)


class KvCacheMemoryKnobs(BaseModel):
    hbm: KvCacheFormatKnobs = Field(default_factory=KvCacheFormatKnobs)
    sram: KvCacheFormatKnobs | None = None
    max_context_tokens: int | None = Field(default=None, ge=0)

    def resolved_sram(self) -> KvCacheFormatKnobs:
        return self.sram or self.hbm


class MemoryKnobs(BaseModel):
    sram: MemoryTechKnobs = Field(default_factory=MemoryTechKnobs)
    hbm: MemoryTechKnobs = Field(default_factory=MemoryTechKnobs)
    fabric: MemoryTechKnobs = Field(default_factory=MemoryTechKnobs)
    kv_cache: KvCacheMemoryKnobs = Field(default_factory=KvCacheMemoryKnobs)


class MemoryLibraryDefaults(BaseModel):
    sram: MemoryTechKnobs = Field(default_factory=MemoryTechKnobs)
    hbm: MemoryTechKnobs = Field(default_factory=MemoryTechKnobs)
    fabric: MemoryTechKnobs = Field(default_factory=MemoryTechKnobs)


class SocLibraryDefaults(BaseModel):
    attention_cim_mac_area_mm2_per_unit: float = Field(0.0, ge=0.0)
    attention_cim_storage_bits_per_element: int | None = Field(default=None, ge=1)
    verify_setup: VerifySetupKnobs = Field(default_factory=VerifySetupKnobs)
    buffers_add: PerOpOverheadSpec = Field(default_factory=PerOpOverheadSpec)
    control: ControlOverheadKnobs = Field(default_factory=ControlOverheadKnobs)


class DigitalCostDefaults(BaseModel):
    attention: PerMacCost
    softmax: PerMacCost
    elementwise: PerMacCost
    kv_cache: PerMacCost
    features: DpuFeatureCostOverrides | None = None
    digital_overhead_area_mm2_per_layer: float = Field(0.0, ge=0.0)

    def resolve_feature_costs(self) -> tuple[DpuFeatureCosts, dict[str, str]]:
        overrides = self.features or DpuFeatureCostOverrides()
        mapping: dict[str, str] = {}

        def pick(
            *,
            feature: str,
            explicit: PerMacCost | None,
            fallback: PerMacCost,
            fallback_source: str,
        ) -> PerMacCost:
            if explicit is not None:
                mapping[feature] = f"explicit:digital.features.{feature}"
                return explicit
            mapping[feature] = f"mapped:{fallback_source}"
            return fallback

        return (
            DpuFeatureCosts(
                attention_qk=pick(
                    feature="attention_qk",
                    explicit=overrides.attention_qk,
                    fallback=self.attention,
                    fallback_source="digital.attention",
                ),
                attention_softmax=pick(
                    feature="attention_softmax",
                    explicit=overrides.attention_softmax,
                    fallback=self.softmax,
                    fallback_source="digital.softmax",
                ),
                attention_pv=pick(
                    feature="attention_pv",
                    explicit=overrides.attention_pv,
                    fallback=self.attention,
                    fallback_source="digital.attention",
                ),
                ffn_activation=pick(
                    feature="ffn_activation",
                    explicit=overrides.ffn_activation,
                    fallback=self.elementwise,
                    fallback_source="digital.elementwise",
                ),
                ffn_gate_multiply=pick(
                    feature="ffn_gate_multiply",
                    explicit=overrides.ffn_gate_multiply,
                    fallback=self.elementwise,
                    fallback_source="digital.elementwise",
                ),
                kv_cache_update=pick(
                    feature="kv_cache_update",
                    explicit=overrides.kv_cache_update,
                    fallback=self.kv_cache,
                    fallback_source="digital.kv_cache",
                ),
            ),
            mapping,
        )


class ResolvedKnobSpecs(BaseModel):
    library: str
    dac_bits: int
    adc_draft_bits: int
    adc_residual_bits: int
    dac: PeripheralSpec
    adc_draft: PeripheralSpec
    adc_residual: PeripheralSpec
    array: AnalogArraySpec
    digital: DigitalCostDefaults


class HardwareConfig(BaseModel):
    reuse_policy: ReusePolicy = ReusePolicy.reuse
    library: str | None = None
    library_file: str | None = None
    soc: SocKnobs = Field(default_factory=SocKnobs)
    memory: MemoryKnobs | None = None
    analog: AnalogKnobs | None = None
    costs: HardwareCosts | None = None

    DEFAULT_LIBRARY: ClassVar[str] = "puma_like_v1"
    RUNTIME_LIBRARY_DEFAULT_FILE: ClassVar[Path] = (
        Path(__file__).resolve().parent / "libraries" / "runtime_libraries.json"
    )
    REQUIRED_LIBRARY_SECTIONS: ClassVar[tuple[str, ...]] = ("adc", "dac", "array", "digital")
    PAPER_LIBRARY_EXTRACTS: ClassVar[dict[str, dict[str, Any]]] = {
        "science_adi9405_2024": {
            "sources": [
                "reference/Programming memristor arrays with arbitrarily high precision for analog computing  Science.pdf",
                "reference/science.adi9405_sm.pdf",
            ],
            "notes": [
                "This extraction includes only values explicitly stated in the paper/supplement, plus closed-form derivations from those values.",
                "It is not directly runnable as a knob-based estimator library because the paper does not provide many required component specs.",
            ],
            "extracted_specs": {
                "array_geometry": [
                    {
                        "platform": "soc_fully_integrated",
                        "rows": 256,
                        "cols": 256,
                        "citation": "Science main text Fig. 2 caption; supplement Fig. S5",
                    },
                    {
                        "platform": "non_fully_integrated",
                        "rows": 128,
                        "cols": 64,
                        "citation": "Science main text (non-fully integrated platform); supplement Fig. S2",
                    },
                ],
                "soc": {
                    "process_node_nm": 65,
                    "cores_per_chip": 10,
                    "citation": "Supplementary Materials and Methods; supplement Fig. S5",
                },
                "vmm_operating_point": {
                    "latency_ns_per_vmm": 10.0,
                    "average_voltage_v": 0.05,
                    "average_cell_resistance_ohm": 10_000.0,
                    "subarrays_used_in_efficiency_example": 5,
                    "array_efficiency_tops_per_w": 160.0,
                    "citation": "Supplementary Text energy/time calculation section",
                },
                "comparison_assumptions": {
                    "adc_energy_pj_per_sample_assumed": 1.75,
                    "hbm_bandwidth_gb_per_s_per_w_assumed": 35.0,
                    "citation": "Supplementary Text energy/time calculation section",
                },
                "derived_for_library_alignment": {
                    "array_energy_pj_per_cell_activation": 0.0025,
                    "array_energy_derivation": "V^2/R * t, using V=0.05V, R=10kOhm, t=10ns",
                    "vmm_energy_pj_for_256x256_with_5_subarrays": 819.2,
                    "vmm_energy_derivation": "(256*256*5) * (0.05^2/10000) * 10ns",
                    "derived_from": "Supplementary Text values in vmm_operating_point",
                },
            },
            "missing_specs": [
                "array_geometry.128x128",
                "array.area_mm2_per_weight",
                "array.energy_pj_per_activation_standardized_for_estimator",
                "array.latency_ns_per_activation_standardized_for_estimator",
                "adc.bits_available",
                "adc.energy_pj_per_conversion_by_bits",
                "adc.latency_ns_per_conversion_by_bits",
                "adc.area_mm2_per_unit_by_bits",
                "dac.bits_available",
                "dac.energy_pj_per_conversion_by_bits",
                "dac.latency_ns_per_conversion_by_bits",
                "dac.area_mm2_per_unit_by_bits",
                "digital.attention.energy_pj_per_mac",
                "digital.attention.latency_ns_per_mac",
                "digital.softmax.energy_pj_per_mac",
                "digital.softmax.latency_ns_per_mac",
                "digital.elementwise.energy_pj_per_mac",
                "digital.elementwise.latency_ns_per_mac",
                "digital.kv_cache.energy_pj_per_mac",
                "digital.kv_cache.latency_ns_per_mac",
                "digital.digital_overhead_area_mm2_per_layer",
                "soc.verify_setup.energy_pj_per_burst",
                "soc.verify_setup.latency_ns_per_burst",
                "soc.buffers_add.energy_pj_per_op",
                "soc.buffers_add.latency_ns_per_op",
                "soc.buffers_add.area_mm2_per_unit",
                "soc.control.energy_pj_per_token",
                "soc.control.latency_ns_per_token",
                "soc.control.energy_pj_per_burst",
                "soc.control.latency_ns_per_burst",
                "memory.sram.read_energy_pj_per_byte",
                "memory.sram.write_energy_pj_per_byte",
                "memory.sram.read_bandwidth_GBps",
                "memory.sram.write_bandwidth_GBps",
                "memory.sram.read_latency_ns",
                "memory.sram.write_latency_ns",
                "memory.sram.area_mm2",
                "memory.hbm.read_energy_pj_per_byte",
                "memory.hbm.write_energy_pj_per_byte",
                "memory.hbm.read_bandwidth_GBps",
                "memory.hbm.write_bandwidth_GBps",
                "memory.hbm.read_latency_ns",
                "memory.hbm.write_latency_ns",
                "memory.hbm.area_mm2",
                "memory.fabric.read_energy_pj_per_byte",
                "memory.fabric.write_energy_pj_per_byte",
                "memory.fabric.read_bandwidth_GBps",
                "memory.fabric.write_bandwidth_GBps",
                "memory.fabric.read_latency_ns",
                "memory.fabric.write_latency_ns",
                "memory.fabric.area_mm2",
                "analog_periphery.tia.energy_pj_per_op",
                "analog_periphery.tia.latency_ns_per_op",
                "analog_periphery.tia.area_mm2_per_unit",
                "analog_periphery.snh.energy_pj_per_op",
                "analog_periphery.snh.latency_ns_per_op",
                "analog_periphery.snh.area_mm2_per_unit",
                "analog_periphery.mux.energy_pj_per_op",
                "analog_periphery.mux.latency_ns_per_op",
                "analog_periphery.mux.area_mm2_per_unit",
                "analog_periphery.io_buffers.energy_pj_per_op",
                "analog_periphery.io_buffers.latency_ns_per_op",
                "analog_periphery.io_buffers.area_mm2_per_unit",
                "analog_periphery.subarray_switches.energy_pj_per_op",
                "analog_periphery.subarray_switches.latency_ns_per_op",
                "analog_periphery.subarray_switches.area_mm2_per_unit",
                "analog_periphery.write_drivers.energy_pj_per_op",
                "analog_periphery.write_drivers.latency_ns_per_op",
                "analog_periphery.write_drivers.area_mm2_per_unit",
            ],
        }
    }

    @classmethod
    def _normalize_bit_table(
        cls,
        table: Any,
        *,
        library_name: str,
        section: str,
        source: str,
    ) -> dict[int, dict[str, Any]]:
        if not isinstance(table, dict):
            raise ValueError(
                f"Invalid library '{library_name}' in '{source}': section '{section}' must be a JSON object"
            )
        normalized: dict[int, dict[str, Any]] = {}
        for raw_bits, raw_spec in table.items():
            try:
                bits = int(raw_bits)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid library '{library_name}' in '{source}': section '{section}' has non-integer bit key "
                    f"{raw_bits!r}"
                ) from exc
            if bits <= 0:
                raise ValueError(
                    f"Invalid library '{library_name}' in '{source}': section '{section}' has non-positive bit key {bits}"
                )
            if not isinstance(raw_spec, dict):
                raise ValueError(
                    f"Invalid library '{library_name}' in '{source}': section '{section}' bit {bits} must map to an object"
                )
            normalized[bits] = deepcopy(raw_spec)
        return normalized

    @classmethod
    def _normalize_runtime_libraries(cls, payload: Any, *, source: str) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid library source '{source}': top-level JSON must be an object")

        normalized_payload: dict[str, dict[str, Any]] = {}
        for library_name, library_spec in payload.items():
            if not isinstance(library_name, str) or not library_name:
                raise ValueError(
                    f"Invalid library source '{source}': library names must be non-empty strings (got {library_name!r})"
                )
            if not isinstance(library_spec, dict):
                raise ValueError(
                    f"Invalid library '{library_name}' in '{source}': library definition must be an object"
                )

            missing = [section for section in cls.REQUIRED_LIBRARY_SECTIONS if section not in library_spec]
            if missing:
                raise ValueError(
                    f"Invalid library '{library_name}' in '{source}': missing required sections: {', '.join(missing)}"
                )

            normalized_spec = deepcopy(library_spec)
            normalized_spec["adc"] = cls._normalize_bit_table(
                library_spec.get("adc"),
                library_name=library_name,
                section="adc",
                source=source,
            )
            normalized_spec["dac"] = cls._normalize_bit_table(
                library_spec.get("dac"),
                library_name=library_name,
                section="dac",
                source=source,
            )
            normalized_payload[library_name] = normalized_spec

        return normalized_payload

    @classmethod
    def _load_runtime_libraries(cls, library_file: str | None) -> dict[str, dict[str, Any]]:
        path = cls.RUNTIME_LIBRARY_DEFAULT_FILE if library_file is None else Path(library_file)
        if not path.exists():
            raise ValueError(f"Library source file not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid library JSON in '{path}': {exc.msg} at line {exc.lineno}, column {exc.colno}"
            ) from exc
        return cls._normalize_runtime_libraries(raw, source=str(path))

    def runtime_libraries(self) -> dict[str, dict[str, Any]]:
        return self._load_runtime_libraries(self.library_file)

    @model_validator(mode="after")
    def _validate_mode(self) -> "HardwareConfig":
        has_analog = self.analog is not None
        has_costs = self.costs is not None
        if has_analog and has_costs:
            raise ValueError("hardware config is ambiguous: do not mix analog.* knob fields with legacy costs.*")
        if not has_analog and not has_costs:
            raise ValueError("hardware config must provide either analog.* knobs or legacy costs.*")
        if has_analog:
            self.resolve_knob_specs()
            self._apply_library_defaults()
        return self

    def _apply_library_defaults(self) -> None:
        lib = self.runtime_libraries().get(self.selected_library)
        if lib is None:
            return

        soc_defaults = SocLibraryDefaults.model_validate(lib.get("soc", {}))
        if "attention_cim_mac_area_mm2_per_unit" not in self.soc.model_fields_set:
            self.soc.attention_cim_mac_area_mm2_per_unit = soc_defaults.attention_cim_mac_area_mm2_per_unit
        if "attention_cim_storage_bits_per_element" not in self.soc.model_fields_set:
            self.soc.attention_cim_storage_bits_per_element = soc_defaults.attention_cim_storage_bits_per_element
        for field in ["energy_pj_per_burst", "latency_ns_per_burst"]:
            if field not in self.soc.verify_setup.model_fields_set:
                setattr(self.soc.verify_setup, field, getattr(soc_defaults.verify_setup, field))
        for field in ["energy_pj_per_op", "latency_ns_per_op", "area_mm2_per_unit"]:
            if field not in self.soc.buffers_add.model_fields_set:
                setattr(self.soc.buffers_add, field, getattr(soc_defaults.buffers_add, field))
        for field in ["energy_pj_per_token", "latency_ns_per_token", "energy_pj_per_burst", "latency_ns_per_burst"]:
            if field not in self.soc.control.model_fields_set:
                setattr(self.soc.control, field, getattr(soc_defaults.control, field))

        assert self.analog is not None
        periphery_defaults = AnalogPeripheryKnobs.model_validate(lib.get("analog_periphery", {}))
        for name in ["tia", "snh", "mux", "io_buffers", "subarray_switches", "write_drivers"]:
            cur_spec = getattr(self.analog.periphery, name)
            def_spec = getattr(periphery_defaults, name)
            for field in ["energy_pj_per_op", "latency_ns_per_op", "area_mm2_per_unit"]:
                if field not in cur_spec.model_fields_set:
                    setattr(cur_spec, field, getattr(def_spec, field))

        if self.memory is not None:
            memory_defaults = MemoryLibraryDefaults.model_validate(lib.get("memory", {}))
            for name in ["sram", "hbm", "fabric"]:
                cur_tech = getattr(self.memory, name)
                def_tech = getattr(memory_defaults, name)
                for field in [
                    "read_energy_pj_per_byte",
                    "write_energy_pj_per_byte",
                    "read_bandwidth_GBps",
                    "write_bandwidth_GBps",
                    "read_latency_ns",
                    "write_latency_ns",
                    "area_mm2",
                    "capacity_bytes",
                ]:
                    if field not in cur_tech.model_fields_set:
                        setattr(cur_tech, field, getattr(def_tech, field))

    @property
    def mode(self) -> HardwareMode:
        if self.analog is not None:
            return HardwareMode.knob_based
        return HardwareMode.legacy

    @property
    def selected_library(self) -> str:
        return self.library or self.DEFAULT_LIBRARY

    def resolve_knob_specs(self) -> ResolvedKnobSpecs:
        if self.mode != HardwareMode.knob_based:
            raise ValueError("Cannot resolve knob specs for legacy costs.* config")

        library_name = self.selected_library
        libraries = self.runtime_libraries()
        lib = libraries.get(library_name)
        if lib is None:
            raise ValueError(
                f"Unknown hardware library '{library_name}'. "
                f"Available: {', '.join(sorted(libraries))}"
            )

        assert self.analog is not None
        adc_table: dict[int, dict[str, Any]] = lib["adc"]
        dac_table: dict[int, dict[str, Any]] = lib["dac"]

        draft_bits = self.analog.adc.draft_bits
        residual_bits = self.analog.adc.residual_bits
        dac_bits = self.analog.dac_bits

        if draft_bits not in adc_table:
            raise ValueError(
                f"Requested analog.adc.draft_bits={draft_bits} is not available in library '{library_name}'. "
                f"Available ADC bits: {sorted(adc_table)}"
            )
        if residual_bits not in adc_table:
            raise ValueError(
                f"Requested analog.adc.residual_bits={residual_bits} is not available in library '{library_name}'. "
                f"Available ADC bits: {sorted(adc_table)}"
            )
        if dac_bits not in dac_table:
            raise ValueError(
                f"Requested analog.dac_bits={dac_bits} is not available in library '{library_name}'. "
                f"Available DAC bits: {sorted(dac_table)}"
            )

        return ResolvedKnobSpecs(
            library=library_name,
            dac_bits=dac_bits,
            adc_draft_bits=draft_bits,
            adc_residual_bits=residual_bits,
            dac=PeripheralSpec.model_validate(dac_table[dac_bits]),
            adc_draft=PeripheralSpec.model_validate(adc_table[draft_bits]),
            adc_residual=PeripheralSpec.model_validate(adc_table[residual_bits]),
            array=AnalogArraySpec.model_validate(lib["array"]),
            digital=DigitalCostDefaults.model_validate(lib["digital"]),
        )

    def resolved_library_payload(self) -> dict[str, Any] | None:
        if self.mode != HardwareMode.knob_based:
            return None
        specs = self.resolve_knob_specs()
        _resolved_feature_costs, feature_mapping = specs.digital.resolve_feature_costs()
        payload: dict[str, Any] = {
            "name": specs.library,
            "dac": {"bits": specs.dac_bits, **specs.dac.model_dump(mode="json")},
            "adc_draft": {"bits": specs.adc_draft_bits, **specs.adc_draft.model_dump(mode="json")},
            "adc_residual": {"bits": specs.adc_residual_bits, **specs.adc_residual.model_dump(mode="json")},
            "digital": specs.digital.model_dump(mode="json"),
            "digital_feature_mapping": feature_mapping,
        }
        lib = self.runtime_libraries().get(specs.library)
        if lib is not None:
            if "soc" in lib:
                payload["soc"] = SocLibraryDefaults.model_validate(lib["soc"]).model_dump(mode="json")
            if "memory" in lib:
                payload["memory"] = MemoryLibraryDefaults.model_validate(lib["memory"]).model_dump(mode="json")
            if "analog_periphery" in lib:
                payload["analog_periphery"] = AnalogPeripheryKnobs.model_validate(lib["analog_periphery"]).model_dump(
                    mode="json"
                )
        return payload

    @classmethod
    def paper_library_extract(cls, name: str = "science_adi9405_2024") -> dict[str, Any]:
        extract = cls.PAPER_LIBRARY_EXTRACTS.get(name)
        if extract is None:
            raise ValueError(
                f"Unknown paper extract '{name}'. "
                f"Available: {', '.join(sorted(cls.PAPER_LIBRARY_EXTRACTS))}"
            )
        return deepcopy(extract)

    @classmethod
    def paper_library_missing_specs(cls, name: str = "science_adi9405_2024") -> list[str]:
        extract = cls.paper_library_extract(name)
        missing = extract.get("missing_specs", [])
        return [str(path) for path in missing]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HardwareConfig":
        p = Path(path)
        data = _load_yaml(p)
        raw_library_file = data.get("library_file")
        if raw_library_file is not None:
            lf = Path(str(raw_library_file))
            if not lf.is_absolute():
                lf = (p.parent / lf).resolve()
            data["library_file"] = str(lf)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid hardware config: {path}\n{exc}") from exc


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(path))
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse YAML: {p}") from exc


class InputPaths(BaseModel):
    model: str
    hardware: str
    stats: str

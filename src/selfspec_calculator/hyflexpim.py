from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import FfnType, ModelConfig
from .sota_baselines import DEFAULT_MODEL_PATHS, WorkloadOps, estimate_workload_ops


PowerMode = Literal["active_pu", "module_utilized"]
ThroughputMode = Literal["single_stream", "paper_pipeline"]


class HyFlexPIMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    slc_rate: float = Field(0.20, ge=0.0, le=1.0)
    svd_rank_scale: float = Field(1.0, gt=0.0)
    activation_bits: int = Field(8, ge=1)
    weight_bits: int = Field(8, ge=1)
    value_bits: int = Field(12, ge=1)
    analog_rows: int = Field(64, ge=1)
    analog_columns: int = Field(128, ge=1)
    analog_arrays_per_module: int = Field(512, ge=1)
    analog_modules_per_pu: int = Field(24, ge=1)
    analog_cycle_ns: float = Field(100.0, gt=0.0)
    digital_modules_per_pu: int = Field(8, ge=1)
    digital_ops_per_cycle_per_module: float = Field(273.0, gt=0.0)
    sfu_inputs_per_cycle_per_module: float = Field(256.0, gt=0.0)
    digital_frequency_hz: float = Field(1.0e9, gt=0.0)
    pus_per_chip: int = Field(24, ge=1)
    analog_area_per_pu_mm2: float = Field(11.24, ge=0.0)
    analog_power_per_pu_w: float = Field(22.33659, ge=0.0)
    digital_area_per_pu_mm2: float = Field(64.05, ge=0.0)
    digital_power_per_pu_w: float = Field(52.25641, ge=0.0)
    inter_layer_transfer_cycles: int = Field(24, ge=0)
    inter_chip_bandwidth_gbs: float = Field(128.0, gt=0.0)
    hidden_transfer_bytes_per_element: int = Field(1, ge=1)
    tensor_pus_per_layer: int | None = Field(default=None, ge=1)
    include_digital_attention: bool = True
    include_nonlinear_ops: bool = True
    analog_power_mode: PowerMode = "active_pu"
    throughput_mode: ThroughputMode = "single_stream"

    @property
    def effective_cell_bits(self) -> float:
        return self.slc_rate * 1.0 + (1.0 - self.slc_rate) * 2.0

    @property
    def analog_capacity_cells_per_pu(self) -> int:
        return (
            self.analog_modules_per_pu
            * self.analog_arrays_per_module
            * self.analog_rows
            * self.analog_columns
        )

    @property
    def analog_output_values_per_array(self) -> float:
        return self.analog_columns * self.effective_cell_bits / float(self.weight_bits)


class LinearBranch(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    group: str
    input_dim: int = Field(..., ge=1)
    output_dim: int = Field(..., ge=1)
    hard_threshold_rank: int = Field(..., ge=1)
    rank: int = Field(..., ge=1)
    original_macs: float = Field(..., ge=0.0)
    svd_macs: float = Field(..., ge=0.0)
    array_tasks: float = Field(..., ge=0.0)
    array_waves: int = Field(..., ge=1)
    array_utilization: float = Field(..., ge=0.0, le=1.0)
    assigned_modules: int = Field(..., ge=1)
    latency_s: float = Field(..., ge=0.0)


class LinearGroupProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    branches: list[LinearBranch]
    assigned_modules_per_branch: int = Field(..., ge=1)
    latency_s: float = Field(..., ge=0.0)
    energy_j: float = Field(..., ge=0.0)
    svd_macs: float = Field(..., ge=0.0)
    original_macs: float = Field(..., ge=0.0)
    power_w: float = Field(..., ge=0.0)


class LinearLayerProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    groups: list[LinearGroupProfile]
    latency_s: float = Field(..., ge=0.0)
    energy_j: float = Field(..., ge=0.0)
    original_macs: float = Field(..., ge=0.0)
    svd_macs: float = Field(..., ge=0.0)
    svd_params: float = Field(..., ge=0.0)


class HyFlexPIMRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    model_path: str | None = None
    prompt_len: int = Field(..., ge=0)
    generated_tokens: int = Field(..., ge=1)
    slc_rate: float = Field(..., ge=0.0, le=1.0)
    svd_rank_scale: float = Field(..., gt=0.0)
    effective_cell_bits: float = Field(..., gt=0.0)
    tensor_pus_per_layer: int = Field(..., ge=1)
    capacity_pus_per_layer: int = Field(..., ge=1)
    chips: int = Field(..., ge=1)
    allocated_pus: int = Field(..., ge=1)
    layers_per_chip: int = Field(..., ge=1)
    original_linear_macs: float = Field(..., ge=0.0)
    hyflex_svd_linear_macs: float = Field(..., ge=0.0)
    attention_qk_macs: float = Field(..., ge=0.0)
    attention_pv_macs: float = Field(..., ge=0.0)
    attention_softmax_ops: float = Field(..., ge=0.0)
    digital_nonlinear_ops: float = Field(..., ge=0.0)
    analog_latency_s: float = Field(..., ge=0.0)
    digital_attention_latency_s: float = Field(..., ge=0.0)
    digital_nonlinear_latency_s: float = Field(..., ge=0.0)
    transfer_latency_s: float = Field(..., ge=0.0)
    total_latency_s: float = Field(..., ge=0.0)
    latency_s_per_token: float = Field(..., ge=0.0)
    reported_throughput_mode: ThroughputMode
    single_stream_total_latency_s: float = Field(..., ge=0.0)
    single_stream_latency_s_per_token: float = Field(..., ge=0.0)
    single_stream_throughput_tokens_per_s: float = Field(..., ge=0.0)
    paper_pipeline_total_latency_s: float = Field(..., ge=0.0)
    paper_pipeline_latency_s_per_token: float = Field(..., ge=0.0)
    paper_pipeline_throughput_tokens_per_s: float = Field(..., ge=0.0)
    paper_pipeline_fill_latency_s: float = Field(..., ge=0.0)
    analog_energy_j: float = Field(..., ge=0.0)
    digital_attention_energy_j: float = Field(..., ge=0.0)
    digital_nonlinear_energy_j: float = Field(..., ge=0.0)
    total_energy_j: float = Field(..., ge=0.0)
    energy_j_per_token: float = Field(..., ge=0.0)
    throughput_tokens_per_s: float = Field(..., ge=0.0)
    tokens_per_joule: float = Field(..., ge=0.0)
    active_layer_area_mm2: float = Field(..., ge=0.0)
    allocated_area_mm2: float = Field(..., ge=0.0)
    analog_capacity_cells_per_pu: int = Field(..., ge=1)
    analog_cells_required_per_layer: float = Field(..., ge=0.0)
    layer_profile: LinearLayerProfile
    basis: str
    caveats: list[str] = Field(default_factory=list)


class HyFlexPIMReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    generated_at: str
    prompt_len: int = Field(..., ge=0)
    generated_tokens: int = Field(..., ge=1)
    model_paths: list[str]
    config: HyFlexPIMConfig
    workloads: list[WorkloadOps]
    rows: list[HyFlexPIMRow]
    notes: list[str] = Field(default_factory=list)


def _model_name(model: ModelConfig, fallback: Path | None = None) -> str:
    if model.name:
        return model.name
    if fallback is not None:
        return fallback.stem
    return "unnamed-model"


def _ceil_div_float(lhs: float, rhs: float) -> int:
    return int(math.ceil(lhs / rhs))


def _linear_groups(model: ModelConfig) -> list[tuple[str, list[tuple[str, int, int]]]]:
    d_model = model.d_model
    d_ff = model.effective_d_ff
    groups: list[tuple[str, list[tuple[str, int, int]]]] = [
        (
            "qkv",
            [
                ("q", d_model, d_model),
                ("k", d_model, d_model),
                ("v", d_model, d_model),
            ],
        ),
        ("wo", [("wo", d_model, d_model)]),
    ]
    if model.ffn_type == FfnType.mlp:
        groups.extend(
            [
                ("ffn_in", [("ffn_in", d_model, d_ff)]),
                ("ffn_out", [("ffn_out", d_ff, d_model)]),
            ]
        )
    else:
        groups.extend(
            [
                (
                    "ffn_gate_up",
                    [
                        ("ffn_gate", d_model, d_ff),
                        ("ffn_up", d_model, d_ff),
                    ],
                ),
                ("ffn_down", [("ffn_down", d_ff, d_model)]),
            ]
        )
    return groups


def _hard_threshold_rank(input_dim: int, output_dim: int) -> int:
    rank = math.ceil((input_dim * output_dim) / float(input_dim + output_dim))
    return max(1, min(rank, input_dim, output_dim))


def _svd_rank(input_dim: int, output_dim: int, config: HyFlexPIMConfig) -> int:
    hard_rank = _hard_threshold_rank(input_dim, output_dim)
    rank = math.ceil(hard_rank * config.svd_rank_scale)
    return max(1, min(rank, input_dim, output_dim))


def _mvm_array_stats(
    *,
    input_dim: int,
    output_dim: int,
    assigned_modules: int,
    config: HyFlexPIMConfig,
) -> tuple[float, int, float, float]:
    input_tiles = _ceil_div_float(float(input_dim), float(config.analog_rows))
    output_tiles = _ceil_div_float(float(output_dim), config.analog_output_values_per_array)
    array_tasks = float(input_tiles * output_tiles)
    parallel_arrays = assigned_modules * config.analog_arrays_per_module
    array_waves = max(1, _ceil_div_float(array_tasks, float(parallel_arrays)))
    utilization = min(1.0, array_tasks / float(array_waves * parallel_arrays))
    latency_s = array_waves * config.activation_bits * config.analog_cycle_ns * 1.0e-9
    return array_tasks, array_waves, utilization, latency_s


def _branch_profile(
    *,
    name: str,
    group: str,
    input_dim: int,
    output_dim: int,
    assigned_modules: int,
    config: HyFlexPIMConfig,
) -> LinearBranch:
    hard_rank = _hard_threshold_rank(input_dim, output_dim)
    rank = _svd_rank(input_dim, output_dim, config)
    svd_macs = float(input_dim * rank + rank * output_dim)
    original_macs = float(input_dim * output_dim)

    tasks_a, waves_a, util_a, latency_a = _mvm_array_stats(
        input_dim=input_dim,
        output_dim=rank,
        assigned_modules=assigned_modules,
        config=config,
    )
    tasks_b, waves_b, util_b, latency_b = _mvm_array_stats(
        input_dim=rank,
        output_dim=output_dim,
        assigned_modules=assigned_modules,
        config=config,
    )
    total_tasks = tasks_a + tasks_b
    total_waves = waves_a + waves_b
    utilization = (util_a * waves_a + util_b * waves_b) / float(total_waves)

    return LinearBranch(
        name=name,
        group=group,
        input_dim=input_dim,
        output_dim=output_dim,
        hard_threshold_rank=hard_rank,
        rank=rank,
        original_macs=original_macs,
        svd_macs=svd_macs,
        array_tasks=total_tasks,
        array_waves=total_waves,
        array_utilization=utilization,
        assigned_modules=assigned_modules,
        latency_s=latency_a + latency_b,
    )


def _group_power_w(
    *,
    active_pus: int,
    assigned_modules_per_branch: int,
    branch_count: int,
    config: HyFlexPIMConfig,
) -> float:
    active_power_w = active_pus * config.analog_power_per_pu_w
    if config.analog_power_mode == "active_pu":
        return active_power_w
    used_modules = min(assigned_modules_per_branch * branch_count, active_pus * config.analog_modules_per_pu)
    module_fraction = used_modules / float(active_pus * config.analog_modules_per_pu)
    return active_power_w * module_fraction


def estimate_linear_layer_profile(
    model: ModelConfig,
    *,
    config: HyFlexPIMConfig,
    tensor_pus_per_layer: int,
) -> LinearLayerProfile:
    groups: list[LinearGroupProfile] = []
    for group_name, branch_specs in _linear_groups(model):
        total_modules = tensor_pus_per_layer * config.analog_modules_per_pu
        assigned_modules = max(1, total_modules // len(branch_specs))
        branches = [
            _branch_profile(
                name=name,
                group=group_name,
                input_dim=input_dim,
                output_dim=output_dim,
                assigned_modules=assigned_modules,
                config=config,
            )
            for name, input_dim, output_dim in branch_specs
        ]
        group_latency_s = max(branch.latency_s for branch in branches)
        group_power_w = _group_power_w(
            active_pus=tensor_pus_per_layer,
            assigned_modules_per_branch=assigned_modules,
            branch_count=len(branches),
            config=config,
        )
        groups.append(
            LinearGroupProfile(
                name=group_name,
                branches=branches,
                assigned_modules_per_branch=assigned_modules,
                latency_s=group_latency_s,
                energy_j=group_latency_s * group_power_w,
                svd_macs=sum(branch.svd_macs for branch in branches),
                original_macs=sum(branch.original_macs for branch in branches),
                power_w=group_power_w,
            )
        )

    return LinearLayerProfile(
        groups=groups,
        latency_s=sum(group.latency_s for group in groups),
        energy_j=sum(group.energy_j for group in groups),
        original_macs=sum(group.original_macs for group in groups),
        svd_macs=sum(group.svd_macs for group in groups),
        svd_params=sum(group.svd_macs for group in groups),
    )


def _capacity_pus_per_layer(model: ModelConfig, config: HyFlexPIMConfig) -> int:
    one_pu_profile = estimate_linear_layer_profile(model, config=config, tensor_pus_per_layer=1)
    cells_required = one_pu_profile.svd_params * config.weight_bits / config.effective_cell_bits
    return max(1, math.ceil(cells_required / float(config.analog_capacity_cells_per_pu)))


def _auto_tensor_pus_per_layer(model: ModelConfig, config: HyFlexPIMConfig) -> tuple[int, int]:
    capacity_pus = _capacity_pus_per_layer(model, config)
    if config.tensor_pus_per_layer is not None:
        return config.tensor_pus_per_layer, capacity_pus

    spare_pus_for_small_models = 1
    if model.n_layers <= config.pus_per_chip:
        spare_pus_for_small_models = max(1, config.pus_per_chip // model.n_layers)
    return max(capacity_pus, spare_pus_for_small_models), capacity_pus


def _chip_sizing(model: ModelConfig, config: HyFlexPIMConfig, tensor_pus_per_layer: int) -> tuple[int, int, int]:
    if tensor_pus_per_layer <= config.pus_per_chip:
        layers_per_chip = max(1, config.pus_per_chip // tensor_pus_per_layer)
        chips = math.ceil(model.n_layers / float(layers_per_chip))
    else:
        layers_per_chip = 1
        chips = math.ceil(model.n_layers * tensor_pus_per_layer / float(config.pus_per_chip))
    allocated_pus = chips * config.pus_per_chip
    return chips, allocated_pus, layers_per_chip


def _digital_attention_for_context(
    model: ModelConfig,
    *,
    context_len: int,
    tensor_pus_per_layer: int,
    config: HyFlexPIMConfig,
) -> tuple[float, float, float, float]:
    if not config.include_digital_attention:
        return 0.0, 0.0, 0.0, 0.0

    qk_macs = float(model.n_heads * model.d_head * context_len)
    pv_macs = float(model.n_heads * model.d_head * context_len)
    softmax_ops = float(model.n_heads * context_len)

    digital_ops_per_s = (
        tensor_pus_per_layer
        * config.digital_modules_per_pu
        * config.digital_ops_per_cycle_per_module
        * config.digital_frequency_hz
    )
    sfu_ops_per_s = (
        tensor_pus_per_layer
        * config.digital_modules_per_pu
        * config.sfu_inputs_per_cycle_per_module
        * config.digital_frequency_hz
    )
    qk_latency_s = qk_macs / digital_ops_per_s
    pv_latency_s = (pv_macs * (config.value_bits / 8.0)) / digital_ops_per_s
    softmax_latency_s = softmax_ops / sfu_ops_per_s
    latency_s = qk_latency_s + softmax_latency_s + pv_latency_s
    energy_j = latency_s * tensor_pus_per_layer * config.digital_power_per_pu_w
    return latency_s, energy_j, qk_macs, pv_macs


def _digital_nonlinear_for_layer(
    model: ModelConfig,
    *,
    tensor_pus_per_layer: int,
    config: HyFlexPIMConfig,
) -> tuple[float, float, float]:
    if not config.include_nonlinear_ops:
        return 0.0, 0.0, 0.0

    layernorm_ops = 2.0 * model.d_model
    if model.ffn_type == FfnType.mlp:
        ffn_ops = float(model.effective_d_ff)
    else:
        ffn_ops = 2.0 * model.effective_d_ff
    ops = layernorm_ops + ffn_ops
    sfu_ops_per_s = (
        tensor_pus_per_layer
        * config.digital_modules_per_pu
        * config.sfu_inputs_per_cycle_per_module
        * config.digital_frequency_hz
    )
    latency_s = ops / sfu_ops_per_s
    energy_j = latency_s * tensor_pus_per_layer * config.digital_power_per_pu_w
    return latency_s, energy_j, ops


def _inter_layer_transfer_s(config: HyFlexPIMConfig) -> float:
    return config.inter_layer_transfer_cycles / config.digital_frequency_hz


def _inter_chip_transfer_s(model: ModelConfig, *, config: HyFlexPIMConfig) -> float:
    hidden_bytes = model.d_model * config.hidden_transfer_bytes_per_element
    return hidden_bytes / (config.inter_chip_bandwidth_gbs * 1.0e9)


def _single_stream_transfer_latency_per_token(
    model: ModelConfig,
    *,
    chips: int,
    config: HyFlexPIMConfig,
) -> float:
    inter_layer_s = model.n_layers * _inter_layer_transfer_s(config)
    inter_chip_s = max(0, chips - 1) * _inter_chip_transfer_s(model, config=config)
    return inter_layer_s + inter_chip_s


def _pipeline_transfer_stage_s(model: ModelConfig, *, chips: int, config: HyFlexPIMConfig) -> float:
    inter_chip_s = _inter_chip_transfer_s(model, config=config) if chips > 1 else 0.0
    return _inter_layer_transfer_s(config) + inter_chip_s


def estimate_hyflexpim(
    model: ModelConfig,
    *,
    prompt_len: int,
    generated_tokens: int,
    config: HyFlexPIMConfig | None = None,
    model_path: str | None = None,
) -> tuple[WorkloadOps, HyFlexPIMRow]:
    if config is None:
        config = HyFlexPIMConfig()

    workload = estimate_workload_ops(
        model,
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        model_path=model_path,
    )
    tensor_pus_per_layer, capacity_pus = _auto_tensor_pus_per_layer(model, config)
    if tensor_pus_per_layer < capacity_pus:
        raise ValueError(
            "tensor_pus_per_layer is below analog weight capacity requirement: "
            f"{tensor_pus_per_layer} < {capacity_pus}"
        )
    chips, allocated_pus, layers_per_chip = _chip_sizing(model, config, tensor_pus_per_layer)
    layer_profile = estimate_linear_layer_profile(
        model,
        config=config,
        tensor_pus_per_layer=tensor_pus_per_layer,
    )

    analog_latency_s = generated_tokens * model.n_layers * layer_profile.latency_s
    analog_energy_j = generated_tokens * model.n_layers * layer_profile.energy_j

    layer_nonlinear_latency_s, layer_nonlinear_energy_j, layer_nonlinear_ops = _digital_nonlinear_for_layer(
        model,
        tensor_pus_per_layer=tensor_pus_per_layer,
        config=config,
    )
    digital_attention_latency_s = 0.0
    digital_attention_energy_j = 0.0
    digital_nonlinear_latency_s = generated_tokens * model.n_layers * layer_nonlinear_latency_s
    digital_nonlinear_energy_j = generated_tokens * model.n_layers * layer_nonlinear_energy_j
    pipeline_total_latency_s = 0.0
    first_token_layer_compute_s = 0.0
    pipeline_stage_transfer_s = _pipeline_transfer_stage_s(model, chips=chips, config=config)
    for token_idx in range(generated_tokens):
        context_len = prompt_len + token_idx
        layer_digital_latency_s, layer_digital_energy_j, _qk, _pv = _digital_attention_for_context(
            model,
            context_len=context_len,
            tensor_pus_per_layer=tensor_pus_per_layer,
            config=config,
        )
        digital_attention_latency_s += model.n_layers * layer_digital_latency_s
        digital_attention_energy_j += model.n_layers * layer_digital_energy_j
        stage_latency_s = (
            layer_profile.latency_s
            + layer_digital_latency_s
            + layer_nonlinear_latency_s
            + pipeline_stage_transfer_s
        )
        if token_idx == 0:
            first_token_layer_compute_s = layer_profile.latency_s + layer_digital_latency_s + layer_nonlinear_latency_s
        pipeline_total_latency_s += stage_latency_s

    transfer_latency_s = generated_tokens * _single_stream_transfer_latency_per_token(
        model,
        chips=chips,
        config=config,
    )
    single_stream_total_latency_s = (
        analog_latency_s + digital_attention_latency_s + digital_nonlinear_latency_s + transfer_latency_s
    )
    paper_pipeline_latency_s_per_token = pipeline_total_latency_s / generated_tokens
    paper_pipeline_fill_latency_s = (
        model.n_layers * (first_token_layer_compute_s + _inter_layer_transfer_s(config))
        + max(0, chips - 1) * _inter_chip_transfer_s(model, config=config)
    )
    paper_pipeline_throughput_tokens_per_s = (
        generated_tokens / pipeline_total_latency_s if pipeline_total_latency_s > 0 else 0.0
    )
    single_stream_throughput_tokens_per_s = (
        generated_tokens / single_stream_total_latency_s if single_stream_total_latency_s > 0 else 0.0
    )
    if config.throughput_mode == "paper_pipeline":
        total_latency_s = pipeline_total_latency_s
        throughput = paper_pipeline_throughput_tokens_per_s
    else:
        total_latency_s = single_stream_total_latency_s
        throughput = single_stream_throughput_tokens_per_s

    total_energy_j = analog_energy_j + digital_attention_energy_j + digital_nonlinear_energy_j
    tokens_per_joule = generated_tokens / total_energy_j if total_energy_j > 0 else 0.0
    pu_area = config.analog_area_per_pu_mm2 + config.digital_area_per_pu_mm2
    cells_required = layer_profile.svd_params * config.weight_bits / config.effective_cell_bits

    row = HyFlexPIMRow(
        model=_model_name(model, Path(model_path) if model_path is not None else None),
        model_path=model_path,
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        slc_rate=config.slc_rate,
        svd_rank_scale=config.svd_rank_scale,
        effective_cell_bits=config.effective_cell_bits,
        tensor_pus_per_layer=tensor_pus_per_layer,
        capacity_pus_per_layer=capacity_pus,
        chips=chips,
        allocated_pus=allocated_pus,
        layers_per_chip=layers_per_chip,
        original_linear_macs=workload.linear_macs,
        hyflex_svd_linear_macs=layer_profile.svd_macs * model.n_layers * generated_tokens,
        attention_qk_macs=workload.attention_qk_macs,
        attention_pv_macs=workload.attention_pv_macs,
        attention_softmax_ops=workload.attention_softmax_ops,
        digital_nonlinear_ops=layer_nonlinear_ops * model.n_layers * generated_tokens,
        analog_latency_s=analog_latency_s,
        digital_attention_latency_s=digital_attention_latency_s,
        digital_nonlinear_latency_s=digital_nonlinear_latency_s,
        transfer_latency_s=transfer_latency_s,
        total_latency_s=total_latency_s,
        latency_s_per_token=total_latency_s / generated_tokens,
        reported_throughput_mode=config.throughput_mode,
        single_stream_total_latency_s=single_stream_total_latency_s,
        single_stream_latency_s_per_token=single_stream_total_latency_s / generated_tokens,
        single_stream_throughput_tokens_per_s=single_stream_throughput_tokens_per_s,
        paper_pipeline_total_latency_s=pipeline_total_latency_s,
        paper_pipeline_latency_s_per_token=paper_pipeline_latency_s_per_token,
        paper_pipeline_throughput_tokens_per_s=paper_pipeline_throughput_tokens_per_s,
        paper_pipeline_fill_latency_s=paper_pipeline_fill_latency_s,
        analog_energy_j=analog_energy_j,
        digital_attention_energy_j=digital_attention_energy_j,
        digital_nonlinear_energy_j=digital_nonlinear_energy_j,
        total_energy_j=total_energy_j,
        energy_j_per_token=total_energy_j / generated_tokens,
        throughput_tokens_per_s=throughput,
        tokens_per_joule=tokens_per_joule,
        active_layer_area_mm2=tensor_pus_per_layer * pu_area,
        allocated_area_mm2=allocated_pus * pu_area,
        analog_capacity_cells_per_pu=config.analog_capacity_cells_per_pu,
        analog_cells_required_per_layer=cells_required,
        layer_profile=layer_profile,
        basis=(
            "Local HyFlexPIM analytical decode simulator using the paper's 24 analog modules and 8 digital modules per PU, "
            "512 64x128 analog arrays per module, 256 1024x1024 digital arrays per module, 6/7-bit ADC path, "
            "100 ns analog cycle, 273 INT8 digital operations/cycle/module, INT8 linear/QKV, SFU softmax, "
            "and the paper's PU-per-layer pipeline/tensor-parallel scaling rules."
        ),
        caveats=[
            "This is a hardware/dataflow estimator, not a reproduction of HyFlexPIM's training-time SVD fine-tuning or gradient-rank mapper.",
            "The default SVD rank is the hard-threshold rank from the paper, so MAC and parameter count are approximately preserved.",
            "The default 20% SLC rate follows the decoder-model discussion; use --slc-rate for sensitivity.",
            "Residual adds, LM head, host sampling, and detailed RRAM non-ideality accuracy checks are not included.",
        ],
    )
    return workload, row


def build_report(
    *,
    model_paths: list[Path],
    prompt_len: int,
    generated_tokens: int,
    config: HyFlexPIMConfig | None = None,
) -> HyFlexPIMReport:
    if config is None:
        config = HyFlexPIMConfig()

    workloads: list[WorkloadOps] = []
    rows: list[HyFlexPIMRow] = []
    for model_path in model_paths:
        model = ModelConfig.from_yaml(model_path)
        workload, row = estimate_hyflexpim(
            model,
            prompt_len=prompt_len,
            generated_tokens=generated_tokens,
            config=config,
            model_path=str(model_path),
        )
        workloads.append(workload)
        rows.append(row)

    return HyFlexPIMReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_len=prompt_len,
        generated_tokens=generated_tokens,
        model_paths=[str(path) for path in model_paths],
        config=config,
        workloads=workloads,
        rows=rows,
        notes=[
            "The simulator is intended to replace the single-number HyFlexPIM multiplier with a local matched-workload baseline.",
            "Single-stream mode reports autoregressive request latency, where one generated token must traverse the layers in order.",
            "Paper-pipeline mode reports the HyFlexPIM steady-state layer-PU pipeline period for independent streamed inputs.",
            "Area is reported both for the active tensor-parallel layer and for the allocated HyFlexPIM chips needed to place the model.",
        ],
    )


def _float_for_csv(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _float_for_csv(value) for key, value in row.items()})


def write_report_outputs(report: HyFlexPIMReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    (output_dir / "hyflexpim_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "hyflexpim_rows.csv",
        [
            {
                key: value
                for key, value in row.model_dump(mode="json").items()
                if key != "layer_profile"
            }
            for row in report.rows
        ],
    )
    component_rows: list[dict[str, Any]] = []
    for row in report.rows:
        for group in row.layer_profile.groups:
            for branch in group.branches:
                component_rows.append(
                    {
                        "model": row.model,
                        "group": group.name,
                        "branch": branch.name,
                        "input_dim": branch.input_dim,
                        "output_dim": branch.output_dim,
                        "hard_threshold_rank": branch.hard_threshold_rank,
                        "rank": branch.rank,
                        "assigned_modules": branch.assigned_modules,
                        "array_tasks": branch.array_tasks,
                        "array_waves": branch.array_waves,
                        "array_utilization": branch.array_utilization,
                        "latency_s": branch.latency_s,
                        "original_macs": branch.original_macs,
                        "svd_macs": branch.svd_macs,
                    }
                )
    _write_csv(output_dir / "hyflexpim_layer_components.csv", component_rows)
    (output_dir / "summary.md").write_text(report_to_markdown(report), encoding="utf-8")


def _fmt(value: float, *, unit_scale: float = 1.0, digits: int = 3) -> str:
    scaled = value * unit_scale
    if scaled == 0:
        return "0"
    if abs(scaled) >= 100:
        return f"{scaled:.2f}"
    if abs(scaled) >= 1:
        return f"{scaled:.{digits}f}"
    return f"{scaled:.3e}"


def report_to_markdown(report: HyFlexPIMReport) -> str:
    lines: list[str] = []
    lines.append("# HyFlexPIM Local Simulator")
    lines.append("")
    lines.append(f"Prompt length: `{report.prompt_len}`")
    lines.append(f"Generated tokens: `{report.generated_tokens}`")
    lines.append(f"SLC rate: `{report.config.slc_rate:.3f}`")
    lines.append(f"SVD rank scale: `{report.config.svd_rank_scale:.3f}`")
    lines.append(f"Analog power mode: `{report.config.analog_power_mode}`")
    lines.append(f"Reported throughput mode: `{report.config.throughput_mode}`")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| Model | Mode | PUs/layer | Chips | Reported latency/token (us) | Reported tok/s | Single-stream tok/s | Paper-pipeline tok/s | Energy/token (mJ) | tok/J |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.rows:
        lines.append(
            f"| {row.model} | {row.reported_throughput_mode} | {row.tensor_pus_per_layer} | {row.chips} | "
            f"{_fmt(row.latency_s_per_token, unit_scale=1e6)} | "
            f"{_fmt(row.throughput_tokens_per_s)} | "
            f"{_fmt(row.single_stream_throughput_tokens_per_s)} | "
            f"{_fmt(row.paper_pipeline_throughput_tokens_per_s)} | "
            f"{_fmt(row.energy_j_per_token, unit_scale=1e3)} | "
            f"{_fmt(row.tokens_per_joule)} |"
        )

    lines.append("")
    lines.append("## Mapping")
    lines.append("")
    lines.append(
        "| Model | Capacity PUs/layer | SVD linear MACs / original | Cells/layer | Capacity cells/PU | Single-stream latency/token (us) | Pipeline period/token (us) | Active area (mm2) | Allocated area (mm2) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.rows:
        mac_ratio = row.hyflex_svd_linear_macs / row.original_linear_macs if row.original_linear_macs else 0.0
        lines.append(
            f"| {row.model} | {row.capacity_pus_per_layer} | {mac_ratio:.3f} | "
            f"{row.analog_cells_required_per_layer:.4e} | {row.analog_capacity_cells_per_pu:.4e} | "
            f"{_fmt(row.single_stream_latency_s_per_token, unit_scale=1e6)} | "
            f"{_fmt(row.paper_pipeline_latency_s_per_token, unit_scale=1e6)} | "
            f"{_fmt(row.active_layer_area_mm2)} | {_fmt(row.allocated_area_mm2)} |"
        )

    lines.append("")
    lines.append("## Per-Token Serialized Components")
    lines.append("")
    lines.append(
        "| Model | Analog (us) | Digital attention (us) | Digital nonlinear (us) | Transfer (us) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in report.rows:
        lines.append(
            f"| {row.model} | {_fmt(row.analog_latency_s / row.generated_tokens, unit_scale=1e6)} | "
            f"{_fmt(row.digital_attention_latency_s / row.generated_tokens, unit_scale=1e6)} | "
            f"{_fmt(row.digital_nonlinear_latency_s / row.generated_tokens, unit_scale=1e6)} | "
            f"{_fmt(row.transfer_latency_s / row.generated_tokens, unit_scale=1e6)} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report.notes:
        lines.append(f"- {note}")
    if report.rows:
        for caveat in report.rows[0].caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)

## ADDED Requirements

### Requirement: Report includes component-level area breakdown (on-chip vs off-chip)
The estimator SHALL report a machine-readable area breakdown in units of mm^2 that separates:
- on-chip die area (counted toward chip area), and
- off-chip HBM area (reported separately and NOT counted toward chip area).

The on-chip breakdown SHALL include, at minimum, distinct entries for:
- analog arrays (weight storage),
- DAC area,
- ADC-Draft area,
- ADC-Residual area,
- analog periphery area (e.g., TIA/SNH/MUX/IO buffers/switches/write drivers, as configured),
- SRAM area (on-chip),
- fabric/interconnect area (on-chip, if modeled), and
- existing digital-overhead area.

#### Scenario: HBM area is reported but excluded from on-chip totals
- **WHEN** `hardware.yaml` sets a non-zero `memory.hbm.area_mm2`
- **THEN** the report includes a non-zero off-chip HBM area entry and the on-chip total area does not increase due to that HBM area

### Requirement: Component area accounting uses configured area knobs
In knob-based hardware mode, the estimator MUST compute component area values using the configured area knobs:
- DAC/ADC per-unit area from the resolved library entries, and
- periphery per-unit area from `analog.periphery.*.area_mm2_per_unit`, and
- memory/fabric area from `memory.{sram,hbm,fabric}.area_mm2`.

Unspecified or zero-valued area knobs MUST contribute zero area and MUST NOT prevent estimation.

#### Scenario: Non-zero periphery/memory area knobs produce non-zero reported areas
- **WHEN** the user provides a knob-based `hardware.yaml` with non-zero `analog.periphery.*.area_mm2_per_unit` and/or non-zero `memory.*.area_mm2`
- **THEN** the component-level area breakdown contains corresponding non-zero on-chip and/or off-chip area entries

### Requirement: Existing stage-level area report remains backward compatible
The report MUST preserve the existing stage-level area output (e.g., `{qkv, wo, ffn, digital}_mm2`) with stable semantics so that existing report consumers remain compatible.

Any new component-level area breakdown MUST be additive and MUST NOT remove or rename existing area fields.

#### Scenario: Existing reports remain parseable without new knobs
- **WHEN** the user runs the estimator with an existing `hardware.yaml` that omits all new area/capacity knobs
- **THEN** the report still contains the pre-existing stage-level area fields with the same meaning as before

### Requirement: Layer-pipelined schedule reports steady-state bottleneck token period
When `soc.schedule: layer-pipelined` is enabled, the estimator SHALL model a scaled full-chip configuration where stationary weights are resident and layers execute as pipeline stages.

In this mode, the reported `latency_ns_per_token` SHALL represent a steady-state token period bounded by the bottleneck stage and MUST be computed using a bottleneck model (not by dividing serialized latency by `n_layers`).

Energy accounting MUST remain unchanged by the schedule mode.

#### Scenario: Layer-pipelined time is bottleneck-based, not `/ n_layers`
- **WHEN** `soc.schedule` is set to `layer-pipelined`
- **THEN** the estimator computes throughput/latency using a steady-state bottleneck token period model and does not divide the end-to-end serialized latency by `n_layers`

### Requirement: Shared memory bandwidth is a global throughput constraint in pipelined mode
When `soc.schedule: layer-pipelined` is enabled and a memory hierarchy is configured, shared memory service time (e.g., HBM/fabric/SRAM bandwidth and latency as a bytes-moved model) SHALL be treated as a global resource constraint and SHALL NOT be amortized by the number of layers.

#### Scenario: Pipelined token period can be memory-bottlenecked
- **WHEN** the configured shared memory service time for a step exceeds the slowest per-layer compute time for that step
- **THEN** the reported steady-state token period is determined by the shared memory service time (i.e., memory is the bottleneck)

### Requirement: Optional max context capacity constraint is enforced in sweeps
The hardware configuration MAY specify a maximum supported context capacity as an integer token limit `memory.kv_cache.max_context_tokens`.

When `memory.kv_cache.max_context_tokens` is set, for every prompt-length sweep point the estimator MUST validate:
- `L_prompt + K <= max_context_tokens`

If the constraint is violated, the estimator MUST fail that sweep point with a clear error message that includes `L_prompt`, `K`, and `max_context_tokens`.

When `memory.kv_cache.max_context_tokens` is unset, the estimator MUST NOT enforce a max-context constraint.

#### Scenario: Capacity violation produces a clear error
- **WHEN** `memory.kv_cache.max_context_tokens` is set and a sweep point has `L_prompt + K > max_context_tokens`
- **THEN** the estimator errors with a message that includes the violating values and indicates that the sweep exceeds hardware KV capacity

#### Scenario: No capacity knob means no capacity enforcement
- **WHEN** `memory.kv_cache.max_context_tokens` is unset
- **THEN** the estimator does not reject sweep points based on `L_prompt + K` capacity

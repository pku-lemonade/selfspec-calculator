## ADDED Requirements

### Requirement: Hardware config SHALL define leakage power for all modeled components
The hardware schema SHALL support leakage-power coefficients in mW for all modeled components so users can provide component-level static power inputs.

#### Scenario: Leakage coefficients default to zero
- **WHEN** a hardware config omits leakage-power coefficients
- **THEN** all leakage-power coefficients SHALL default to `0.0 mW`

#### Scenario: Per-component leakage coefficients are accepted
- **WHEN** a hardware config provides leakage-power coefficients per component
- **THEN** the config SHALL validate and preserve each provided coefficient without changing unrelated dynamic-energy knobs

### Requirement: Estimator SHALL compute burst leakage energy from summed leakage power and effective burst latency
The estimator SHALL compute burst leakage energy as:
`E_leak_burst_pJ = (sum of component leakage power in mW) * T_burst_effective_ns`.

#### Scenario: Leakage energy formula is applied
- **WHEN** total component leakage power is `P_leak_total_mW` and effective burst latency is `T_burst_ns`
- **THEN** leakage energy SHALL equal `P_leak_total_mW * T_burst_ns` in pJ

### Requirement: Leakage latency source SHALL be schedule-aware
The effective burst latency used for leakage energy SHALL match the schedule used for latency/token reporting.

#### Scenario: Serialized schedule uses serialized burst time
- **WHEN** `soc.schedule` is `serialized`
- **THEN** leakage energy SHALL use serialized burst latency

#### Scenario: Layer-pipelined schedule uses pipelined burst time
- **WHEN** `soc.schedule` is `layer-pipelined`
- **THEN** leakage energy SHALL use layer-pipelined burst latency

### Requirement: Total energy and derived metrics SHALL include leakage energy
Leakage energy SHALL be added to burst total energy before per-token normalization and derived metric computation.

#### Scenario: Zero leakage preserves legacy behavior
- **WHEN** all leakage-power coefficients are `0.0 mW`
- **THEN** energy/token results SHALL match prior dynamic-only behavior

#### Scenario: Positive leakage increases burst and per-token energy
- **WHEN** total leakage power is positive and burst latency is non-zero
- **THEN** total burst energy and energy/token SHALL increase by the leakage contribution

### Requirement: Report SHALL expose leakage contribution for auditability
The report SHALL expose leakage accounting outputs sufficient to audit leakage impact.

#### Scenario: Report includes leakage summary
- **WHEN** a sweep point is produced
- **THEN** output SHALL include leakage summary fields containing at least total leakage power and leakage energy for that point

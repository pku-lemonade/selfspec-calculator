## ADDED Requirements

### Requirement: Analog readout periphery SHALL use streamed output-path latency
For knob-based analog stages, output-stream periphery blocks that operate on ADC results (`TIA`, `SNH`, `MUX`, and `IO` buffers) SHALL be modeled as part of the same streamed readout path as ADC scan rather than as separate serialized full-scan wall-clock phases.

#### Scenario: Output-stream periphery uses bottleneck path timing
- **WHEN** a knob-based analog stage uses non-zero ADC and output-stream periphery latency coefficients
- **THEN** the stage wall-clock latency contribution from ADC/TIA/SNH/MUX/IO is computed from the maximum streamed output-path latency among those blocks, not the sum of all of them

### Requirement: Memory hierarchy latency SHALL use transfer-path bottlenecks
When memory modeling is enabled, the estimator SHALL treat HBM/fabric-backed traffic as a streamed transfer path and SHALL NOT compute wall-clock latency by summing every hierarchy layer's service time unconditionally.

#### Scenario: HBM-backed traffic does not sum HBM and fabric wall-clock latency
- **WHEN** a transfer reads bytes from HBM and traverses the on-chip fabric
- **THEN** the transfer latency is bounded by the bottleneck transfer path rather than `HBM latency + fabric latency`

#### Scenario: Energy accounting remains additive across memory components
- **WHEN** the estimator computes a memory-backed phase
- **THEN** SRAM, HBM, and fabric energies remain summed into the report even though latency uses overlap-aware transfer timing

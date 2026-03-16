## ADDED Requirements

### Requirement: `layer-pipelined` verify SHALL execute a fixed `K+1` burst
When `soc.schedule` is `layer-pipelined` and `K > 0`, the estimator SHALL model verify as a fixed `K+1` pipelined burst:
- `K` verify steps corresponding to drafted tokens, plus
- `1` bonus verify step.

Acceptance outcome SHALL NOT reduce the amount of executed verify work.

#### Scenario: Early mismatch still executes full verify burst
- **WHEN** accepted-prefix length is `a = 0` for a burst with draft length `K`
- **THEN** the estimator still charges verify latency and energy for all `K+1` verify burst steps

#### Scenario: Full acceptance executes the same verify burst length
- **WHEN** accepted-prefix length is `a = K`
- **THEN** the estimator charges the same verify burst latency and energy as any other acceptance outcome with the same `K` and `L_prompt`

### Requirement: Acceptance SHALL only affect normalization and commit-side effects
In `layer-pipelined` mode, acceptance statistics SHALL affect:
- committed tokens per burst,
- per-token normalization,
- commit-only traffic such as HBM writes,
but SHALL NOT change executed verify-burst length.

#### Scenario: Acceptance changes per-token metrics through committed outputs
- **WHEN** two layer-pipelined runs use the same `K` and hardware but different acceptance histograms
- **THEN** verify burst latency is unchanged while `latency_ns_per_token` may differ because committed tokens per burst differ

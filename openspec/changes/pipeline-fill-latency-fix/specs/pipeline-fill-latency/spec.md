## ADDED Requirements

### Requirement: Layer-pipelined verify burst SHALL include first-token fill latency
When `soc.schedule` is `layer-pipelined` and `K > 0`, verify burst latency SHALL include the fully serialized latency of the first verify token before steady-state pipeline periods are applied to subsequent verify tokens.

#### Scenario: Verify burst is not just n times bottleneck period
- **WHEN** the estimator reports a verify burst for multiple tokens in `layer-pipelined` mode
- **THEN** the burst latency is greater than or equal to one fully serialized verify token plus additional steady-state pipeline periods

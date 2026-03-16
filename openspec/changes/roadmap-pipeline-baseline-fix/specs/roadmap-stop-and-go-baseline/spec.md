## ADDED Requirements

### Requirement: `K=0` baseline in `layer-pipelined` mode SHALL use stop-and-go full-precision latency
When `soc.schedule` is `layer-pipelined` and the estimator evaluates a non-speculative configuration (`K=0`), it SHALL use the conventional full-precision stop-and-go token latency rather than reinterpret the point as a pipelined verify-burst token period.

#### Scenario: Pipelined schedule does not accelerate `K=0` baseline
- **WHEN** the estimator evaluates `K=0` for a hardware configuration with `soc.schedule: layer-pipelined`
- **THEN** the reported `latency_ns_per_token` matches the full-precision stop-and-go token latency produced by serialized phase accounting

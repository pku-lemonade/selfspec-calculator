## ADDED Requirements

### Requirement: Legacy mode SHALL use streamed buffer timing parity
When the estimator runs in legacy explicit-cost mode with non-zero `soc.buffers_add` latency, it SHALL model draft capture and verify combine latency using streamed overlap semantics rather than serializing one buffer operation per output element.

#### Scenario: Legacy draft capture does not serialize over raw output count
- **WHEN** the user runs a legacy explicit-cost configuration with non-zero `soc.buffers_add.latency_ns_per_op`
- **THEN** draft `buffers_add` latency is less than the fully serialized `outputs * latency_ns_per_op` value for the same output shape

#### Scenario: Legacy verify combine remains heavier than draft capture
- **WHEN** the user runs a legacy explicit-cost configuration with reuse enabled for drafted tokens
- **THEN** verify-drafted `buffers_add` latency remains greater than draft `buffers_add` latency when verify still performs reuse read / combine work that draft does not

## ADDED Requirements

### Requirement: Draft buffer capture SHALL use streamed lane-parallel timing
For knob-based analog runs with `reuse_policy: reuse`, the estimator SHALL model draft-output capture latency using ADC-lane parallelism rather than serializing one `buffers_add` latency per output element.

The streamed draft-capture latency MUST scale with:
- `base_reads * num_columns_per_adc`, equivalently
- `outputs / (xbar_size / num_columns_per_adc)`.

#### Scenario: Draft buffer latency scales with scan steps instead of outputs
- **WHEN** the estimator accounts for a drafted token in knob-based mode with non-zero `soc.buffers_add.latency_ns_per_op`
- **THEN** the draft `buffers_add` latency is computed from stream steps (`base_reads * num_columns_per_adc`) rather than raw output count (`base_reads * xbar_size`)

### Requirement: Streamed buffer/add work SHALL overlap with ADC output production
For draft capture, verify residual combine, and full-read ADC-output combine, the estimator SHALL treat `buffers_add` work as part of the same streamed output path as ADC production and SHALL charge only incremental latency beyond the already-modeled analog output-stream latency.

#### Scenario: Faster buffer path adds no extra wall-clock latency
- **WHEN** the streamed `buffers_add` path for a stage is faster than the stage's analog output-stream latency
- **THEN** the estimator adds zero incremental `buffers_add` latency for that stage while preserving `buffers_add` energy

#### Scenario: Slower buffer path stretches the stream only by the excess
- **WHEN** the streamed `buffers_add` path for a stage is slower than the stage's analog output-stream latency
- **THEN** the estimator adds only the positive difference between `buffers_add` stream latency and analog output-stream latency to the stage wall-clock time

### Requirement: Draft default mode SHALL not charge final residual-add semantics
For default draft mode (drafting from Array 1 only with reuse enabled), the estimator MUST treat the draft `buffers_add` path as streamed output capture for reuse and MUST NOT charge the `Final = D_reg + C` residual-add path during the draft phase.

#### Scenario: Verify-drafted retains extra combine work relative to draft
- **WHEN** the user estimates a reused drafted token for a block that ran in default draft mode
- **THEN** the verify-drafted `buffers_add` work remains greater than the draft-phase `buffers_add` work for that block because verify still performs reuse read / combine semantics that draft does not

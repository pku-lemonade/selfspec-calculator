## ADDED Requirements

### Requirement: Serialized accounting SHALL use step-indexed context length
Serialized burst accounting SHALL compute step-dependent costs using `L_i = L_prompt + i` for step index `i` within a burst, instead of using a single fixed context length for all steps.

#### Scenario: Draft steps use increasing context
- **WHEN** a burst with `K > 1` is evaluated in serialized mode
- **THEN** draft-step attention-dependent costs SHALL be evaluated across `i = 0..K-1` using `L_prompt + i`

#### Scenario: Verify bonus step uses post-draft context
- **WHEN** the verify bonus step is evaluated
- **THEN** its step-dependent costs SHALL use `L_prompt + K`

### Requirement: Serialized phase totals SHALL be computed from explicit per-step sums
The estimator SHALL compute serialized burst totals as explicit sums for `draft`, `verify_drafted`, and `verify_bonus` phases before converting to per-token metrics.

#### Scenario: Phase decomposition is preserved
- **WHEN** serialized metrics are produced
- **THEN** the report SHALL include separate phase totals and a total phase equal to their sum

#### Scenario: Expected committed-token normalization
- **WHEN** per-token latency and energy are produced
- **THEN** they SHALL be normalized by expected committed tokens per burst

### Requirement: Analog matmul expansion SHALL include tiling, slicing, and interface reuse factors
For serialized accounting, analog matmul work SHALL include tile decomposition, activation slicing, and ADC scan reuse effects.

#### Scenario: Tile and slice expansion
- **WHEN** `xbar_size` and `dac_bits` are configured
- **THEN** analog stage counts SHALL scale with tile count and `ceil(activation_bits / dac_bits)`

#### Scenario: ADC scan reuse scaling
- **WHEN** `num_columns_per_adc` changes
- **THEN** ADC latency accounting SHALL scale with the corresponding scan steps

### Requirement: Serialized accounting SHALL separate compute costs from movement costs
Serialized accounting SHALL keep compute-path and movement-path costs explicitly separable before final aggregation.

#### Scenario: Movement channel visibility
- **WHEN** serialized breakdowns are generated
- **THEN** movement-related costs SHALL be identifiable independently from compute-feature costs

#### Scenario: Ownership without double counting
- **WHEN** a cost item can belong to compute or movement
- **THEN** the accounting contract SHALL assign single ownership to exactly one channel

### Requirement: Non-KV intermediate movement SHALL be either explicitly modeled or explicitly declared excluded
For serialized runs, movement for non-KV intermediate activations SHALL not be silently omitted.

#### Scenario: Explicit modeling path
- **WHEN** non-KV intermediate traffic is supported
- **THEN** read/write costs for those intermediates SHALL be included in serialized movement accounting

#### Scenario: Explicit exclusion path
- **WHEN** non-KV intermediate traffic is not modeled
- **THEN** the report or notes SHALL explicitly declare the exclusion and its scope

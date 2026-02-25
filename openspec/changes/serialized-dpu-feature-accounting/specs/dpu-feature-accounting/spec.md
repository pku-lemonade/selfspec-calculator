## ADDED Requirements

### Requirement: DPU feature accounting SHALL support independent latency and energy coefficients per feature
The estimator SHALL represent DPU digital work as independent features, each with separately configurable latency and energy coefficients.

#### Scenario: Feature-level coefficient definition
- **WHEN** a DPU-enabled hardware library is resolved
- **THEN** each enabled DPU feature SHALL have an independent latency coefficient and energy coefficient

#### Scenario: Feature-level tuning
- **WHEN** one feature coefficient is changed
- **THEN** only that feature's contribution SHALL change in the breakdown, holding operation counts constant

### Requirement: Attention digital path SHALL be decomposed into explicit DPU features
Serialized accounting SHALL model attention digital work as explicit feature terms, including at minimum QK compute support, softmax, and PV compute support.

#### Scenario: Attention feature visibility
- **WHEN** attention work is present
- **THEN** the serialized breakdown SHALL expose separate contributions for attention DPU feature terms

#### Scenario: Attention parallel units
- **WHEN** attention parallel unit count is increased
- **THEN** attention feature latency SHALL scale according to configured parallelism rules

### Requirement: FFN digital path SHALL include FFN-type-specific feature accounting
Serialized accounting SHALL model FFN digital DPU work with feature coverage that distinguishes MLP and SwiGLU behavior.

#### Scenario: MLP feature coverage
- **WHEN** `ffn_type` is `mlp`
- **THEN** FFN digital accounting SHALL include required activation and pointwise digital feature costs for MLP execution

#### Scenario: SwiGLU gate feature coverage
- **WHEN** `ffn_type` is `swiglu`
- **THEN** FFN digital accounting SHALL include explicit gate-multiply-related feature costs in addition to activation-related costs

### Requirement: Compatibility mapping SHALL preserve legacy behavior when feature-level coefficients are absent
The system SHALL support deterministic fallback mapping from legacy/coarse digital coefficients to feature-level accounting so existing configs remain valid.

#### Scenario: Legacy config fallback
- **WHEN** a config provides only coarse digital coefficients
- **THEN** serialized DPU feature accounting SHALL use documented fallback mapping without failing validation

#### Scenario: Mixed explicit and fallback coefficients
- **WHEN** some feature coefficients are explicitly set and others are absent
- **THEN** explicitly set coefficients SHALL override mapped defaults for the corresponding features

### Requirement: Serialized reports SHALL provide feature-level DPU transparency
Reports SHALL provide enough detail to audit per-feature digital DPU contributions in serialized mode.

#### Scenario: Feature-level breakdown output
- **WHEN** serialized results are emitted
- **THEN** report output SHALL include identifiable per-feature DPU latency and energy contributions

#### Scenario: Coarse-to-feature traceability
- **WHEN** fallback mapping is used
- **THEN** report metadata SHALL indicate that mapped defaults were applied for missing feature coefficients

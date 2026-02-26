## ADDED Requirements

### Requirement: Layer-pipelined verify SHALL use wavefront scheduling
When `soc.schedule` is `layer-pipelined`, verify execution SHALL be modeled as a fine-grained wavefront pipeline rather than a fixed phase-multiplier approximation.

#### Scenario: Verify wavefront replaces coarse fixed verify formula
- **WHEN** a burst with draft length `K` is evaluated in layer-pipelined mode
- **THEN** verify timing SHALL be computed from step-level wavefront execution instead of a fixed `K * verify_drafted + 1 * verify_bonus` tail term

### Requirement: Draft stage SHALL remain token-serial
Draft token generation SHALL remain serialized because each next draft token depends on the previous draft output.

#### Scenario: No token-level overlap in draft stage
- **WHEN** draft stage timing is computed
- **THEN** the model SHALL not overlap draft token `i+1` execution with draft token `i` execution

### Requirement: Verify and bonus SHALL preserve step-indexed context growth
Verify-step work and bonus-step work SHALL use step-indexed context length `L_i = L_prompt + i`.

#### Scenario: Bonus step context index
- **WHEN** all `K` drafted tokens are accepted and bonus executes
- **THEN** bonus-step costs SHALL use `L_prompt + K`

#### Scenario: Verify drafted-step context index
- **WHEN** verify step `i` executes for drafted tokens
- **THEN** verify-step costs SHALL use `L_prompt + i`

### Requirement: Bonus token SHALL be modeled as terminal verify wavefront work
The bonus token SHALL be represented as the terminal step in the same verify wavefront resource path.

#### Scenario: Full-accept includes bonus inside same wavefront
- **WHEN** accepted prefix length is `a = K`
- **THEN** the verify wavefront SHALL execute `K + 1` total verify steps with the final step representing bonus-token generation

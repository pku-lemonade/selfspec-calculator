## ADDED Requirements

### Requirement: Verification SHALL stop at first mismatch
In layer-pipelined mode, verify execution SHALL stop immediately after the first mismatching verify step and SHALL not execute remaining verify steps in that burst.

#### Scenario: Early mismatch halts remaining verify work
- **WHEN** mismatch occurs at verify step `j` where `j < K`
- **THEN** verify execution SHALL stop after step `j` and SHALL not charge latency/energy for verify steps `j+1..K` or bonus

### Requirement: Mismatch step output SHALL be the committed correction token
When mismatch occurs, the mismatching verify step output SHALL be committed as the correction token for that burst.

#### Scenario: Mid-burst mismatch commit behavior
- **WHEN** accepted drafted prefix length is `a` with `0 <= a < K`
- **THEN** committed tokens for the burst SHALL be `a + 1` (accepted prefix plus mismatch-step correction token)

### Requirement: Full acceptance SHALL commit one bonus token
When all drafted tokens are accepted, the model SHALL execute and commit one additional bonus token.

#### Scenario: Full-accept commit behavior
- **WHEN** accepted drafted prefix length is `a = K`
- **THEN** committed tokens for the burst SHALL be `K + 1`

### Requirement: Layer-pipelined burst latency SHALL aggregate outcome-conditioned executed work
Layer-pipelined burst latency SHALL be computed as the histogram-weighted expectation over outcome-conditioned executed verify steps.

#### Scenario: Outcome-conditioned latency aggregation
- **WHEN** acceptance histogram contains multiple outcomes `a in [0, K]`
- **THEN** burst latency SHALL be aggregated from per-outcome executed-step latency, weighted by each outcome probability

### Requirement: Layer-pipelined burst energy SHALL aggregate outcome-conditioned executed work
Layer-pipelined burst energy SHALL be computed as the histogram-weighted expectation over outcome-conditioned executed verify steps.

#### Scenario: No post-mismatch energy charge
- **WHEN** mismatch occurs before step `K`
- **THEN** burst energy SHALL exclude non-executed verify steps beyond the mismatch point

### Requirement: Per-token normalization SHALL use expected committed tokens under stop policy
Per-token latency and energy SHALL be normalized by expected committed tokens per burst under the same stop-and-commit semantics.

#### Scenario: Consistent normalization contract
- **WHEN** per-token metrics are produced in layer-pipelined mode
- **THEN** normalization SHALL divide by `E[committed tokens per burst]` consistent with mismatch and full-accept bonus rules

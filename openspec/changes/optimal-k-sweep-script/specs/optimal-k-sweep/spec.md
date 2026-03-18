## ADDED Requirements

### Requirement: Utility SHALL accept candidate `K` values with expected accepted drafted tokens
The system SHALL provide a utility that accepts a set of candidate burst lengths `K` and, for each candidate, the expected number of accepted drafted tokens per burst `E[a]`.

#### Scenario: Mean-acceptance sweep input loads successfully
- **WHEN** the user provides an input file containing candidate `K` values and `expected_accepted_tokens`
- **THEN** the utility parses the file and validates that each `E[a]` lies in `[0, K]`

### Requirement: Utility SHALL evaluate each candidate `K` with the estimator
For each candidate `K` and prompt length, the utility SHALL build estimator-compatible speculation stats and SHALL compute at least:
- latency/token,
- throughput tokens/s,
- tokens/J.

#### Scenario: Candidate set produces one result row per `K`
- **WHEN** the user provides `N` candidate `K` entries and one prompt length
- **THEN** the utility outputs `N` evaluated result rows for that prompt length

### Requirement: Utility SHALL rank optimal `K` values per prompt length
The utility SHALL identify, for each prompt length, the best candidate `K` by:
- throughput, and
- tokens/J.

#### Scenario: Best `K` summaries are included in output
- **WHEN** the utility evaluates multiple candidate `K` values
- **THEN** the output includes per-prompt-length best-`K` summaries for throughput and tokens/J

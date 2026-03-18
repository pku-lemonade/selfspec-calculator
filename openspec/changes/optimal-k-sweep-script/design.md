## Context

The Roadmap-correct estimator now produces sensible speculative results for fixed `(model, hardware, K, prompt length)` points, but model tuning typically yields acceptance measurements across multiple candidate burst lengths. Selecting `K` manually is error-prone because the best `K` depends on:

- draft burst cost growth with `K`,
- fixed verify-burst cost for each `K`,
- the measured expected accepted drafted tokens `E[a_K]`,
- and prompt length.

The new utility should make that comparison directly from measured acceptance summaries.

## Goals / Non-Goals

**Goals:**
- Accept a table of candidate `K` values and corresponding measured `E[a]`.
- Reuse the estimator instead of re-implementing throughput/energy formulas.
- Emit machine-readable output that is easy to inspect or feed into paper plots.
- Rank `K` values by throughput and tokens/J for each prompt length.

**Non-Goals:**
- Do not replace the main `ppa-calculator` CLI.
- Do not require full acceptance histograms when only the mean accepted drafted tokens is available.
- Do not introduce optimizer heuristics beyond ranking the user-provided candidate set.

## Decisions

### 1. Add a dedicated CLI entry point

Add a new console script, tentatively `ppa-k-sweep`, instead of overloading `ppa-calculator`.

Rationale:
- the workflow is distinct from the main estimator;
- input schema differs from the existing `stats.json` histogram format;
- output is a comparison table rather than a standard report sweep.

### 2. Support average accepted drafted tokens as first-class input

The utility input file will support entries of the form:

- `k`
- `expected_accepted_tokens`

For estimator compatibility, the tool will synthesize a valid histogram with the same mean using the minimal two-bin construction between `floor(E[a])` and `ceil(E[a])`.

Rationale:
- the user explicitly asked for "corresponding accepted token";
- under the current fixed-verify-burst pipeline semantics, throughput/latency depends only on `E[a+1]` for a fixed `K`.

### 3. Emit JSON with both raw points and best-`K` summaries

Output structure:
- input metadata,
- one result row per `(prompt_length, K)`,
- best-by-throughput summary per prompt length,
- best-by-tokens/J summary per prompt length.

Rationale:
- JSON is already the repo norm;
- easy to convert into tables/plots later.

## Risks / Trade-offs

- [Mean-only input hides histogram shape] -> Mitigation: document that the tool uses a synthetic histogram that preserves `E[a]`; this is sufficient for the current fixed-verify-burst throughput model.
- [Future estimator changes may depend on full histogram shape again] -> Mitigation: keep the input format extensible for a future per-`K` histogram field.
- [Users may compare different prompt lengths incorrectly] -> Mitigation: rank best `K` separately for each prompt length.

## Migration Plan

1. Add a small input parser / validator for `K -> E[a]` data.
2. Add conversion from mean accepted tokens to estimator-compatible `SpeculationStats`.
3. Add the new CLI entry point and JSON output.
4. Add tests and README usage examples.

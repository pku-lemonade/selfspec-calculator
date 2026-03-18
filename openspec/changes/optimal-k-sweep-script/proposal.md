## Why

Once a tuned model has acceptance well above the current operating point, the next design question is no longer "does speculation help?" but "which burst length `K` is optimal?" Today the calculator can evaluate one `(K, acceptance stats)` point at a time, but there is no utility that takes measured `K -> accepted tokens` results and turns them into an optimal-`K` recommendation.

## What Changes

- Add a utility CLI that sweeps candidate `K` values using user-provided acceptance results for each `K`.
- Support acceptance inputs expressed as average accepted drafted tokens per burst (`E[a]`) and convert them into valid estimator inputs.
- Report per-`K` throughput, latency/token, and tokens/J for each prompt length, plus the best `K` for throughput and energy efficiency.
- Add tests and README documentation for the new sweep utility.

## Capabilities

### New Capabilities
- `optimal-k-sweep`: Sweep candidate burst lengths from measured acceptance summaries and rank the resulting performance/efficiency.

### Modified Capabilities
- None.

## Impact

- CLI / utility code:
  - new module(s) under `src/selfspec_calculator/`
  - `pyproject.toml` console script entry
- Documentation:
  - `README.md`
- Validation:
  - new CLI / utility tests under `tests/`

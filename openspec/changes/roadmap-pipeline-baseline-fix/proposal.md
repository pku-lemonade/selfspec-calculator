## Why

The current `layer-pipelined` estimator semantics do not match `Roadmap.md`. It treats `K=0` as a pipelined full-precision token period instead of a conventional stop-and-go baseline, and it stops verify work at the first mismatch even though the Roadmap says pipelined verification always executes `K+1` burst steps and only discards the unusable suffix.

## What Changes

- Correct the `layer-pipelined` baseline so non-speculative `K=0` uses stop-and-go full-precision token latency rather than speculative verify-burst token period.
- Change pipelined verify timing/accounting to always execute a fixed `K+1` verify burst, independent of acceptance outcome, while keeping acceptance-dependent committed-token normalization and commit traffic.
- Update report notes / pipeline metadata so the reported baseline and verify-burst semantics match the Roadmap execution contract.
- Add regression tests for stop-and-go baseline semantics, fixed `K+1` verify execution, and the resulting break-even behavior against baseline.

## Capabilities

### New Capabilities
- `roadmap-stop-and-go-baseline`: Define the Roadmap-correct non-speculative baseline used against speculative pipelined bursts.
- `roadmap-fixed-verify-burst`: Define Roadmap-correct pipelined verify-burst execution semantics where verify always executes `K+1` steps.

### Modified Capabilities
- None.

## Impact

- Estimator logic:
  - `src/selfspec_calculator/estimator.py`
- Docs / report metadata:
  - `README.md`
- Validation:
  - `tests/test_soc_hardware_model.py`
  - `tests/test_report_shape.py`

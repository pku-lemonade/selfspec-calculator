## Why

Current serialized latency and energy accounting captures the main analog path, but several digital-processing and data-movement parts are still coarse or missing. This makes it hard to trust stage-level breakdowns when evaluating design tradeoffs, especially for attention and FFN DPU behavior.

## What Changes

- Add complete serialized-step accounting checks and requirements for all per-token compute and movement stages.
- Expand digital processing unit (DPU) modeling from coarse buckets into independent feature terms with separate latency/energy coefficients.
- Add explicit per-feature accounting for attention digital path (QK, softmax, PV, and related DPU work) and FFN digital path (activation, gate multiply, and related elementwise work).
- Add explicit requirements for serialized context growth handling across burst steps (not only fixed prompt length).
- Add requirements to distinguish what is explicitly modeled, proxy-modeled, and intentionally excluded in serialized reporting.

## Capabilities

### New Capabilities
- `serialized-accounting-completeness`: Define the full serialized accounting contract per phase and per token-step, including analog matmul expansion, digital compute, and memory/buffer movement coverage.
- `dpu-feature-accounting`: Define independent DPU feature-level latency/energy accounting for attention and FFN digital sub-operations.

### Modified Capabilities
- None.

## Impact

- Estimator logic:
  - `src/selfspec_calculator/estimator.py`
- Report schema/output shape:
  - `src/selfspec_calculator/report.py`
  - `README.md` reporting/modeling sections
- Runtime library schema and defaults for digital feature coefficients:
  - `src/selfspec_calculator/config.py`
  - `src/selfspec_calculator/libraries/runtime_libraries.json`
- Validation/tests:
  - `tests/test_soc_hardware_model.py`
  - new serialized-accounting and DPU-feature coverage tests

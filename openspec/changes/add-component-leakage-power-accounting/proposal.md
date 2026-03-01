## Why

Current energy accounting is operation-only (dynamic energy). This underestimates total energy, especially in long-latency regimes where static leakage can dominate. We need a small, explicit leakage model so energy/token and power results better reflect real hardware behavior.

## What Changes

- Add component-level leakage power specs (default `0`) so every modeled component can contribute static power.
- Add leakage energy accounting in estimator:
  - `P_leak_total = sum(component leakage powers)`
  - `E_leak_burst = P_leak_total * T_burst_effective`
  - `E_total = E_dynamic + E_leak`
- Keep schedule-aware leakage timing:
  - serialized mode uses serialized burst latency
  - layer-pipelined mode uses pipelined burst latency
- Include leakage contribution in reported totals/derived metrics and add tests for correctness/regression.

## Capabilities

### New Capabilities
- `component-leakage-power-accounting`: Define configuration, accounting, and reporting requirements for component-level leakage power and leakage energy aggregation.

### Modified Capabilities
- None.

## Impact

- Estimator and accounting: `src/selfspec_calculator/estimator.py`
- Hardware schema/validation: `src/selfspec_calculator/config.py`
- Report shape/metadata: `src/selfspec_calculator/report.py`
- Documentation: `README.md`
- Validation: `tests/test_soc_hardware_model.py`, `tests/test_report_shape.py`

## 1. Leakage Power Schema

- [ ] 1.1 Add per-component leakage-power fields (mW) to hardware configuration models, covering all modeled components.
- [ ] 1.2 Set default leakage-power values to zero and ensure backward-compatible validation/loading behavior.
- [ ] 1.3 Ensure leakage-power config is preserved in report-visible hardware metadata paths where appropriate.

## 2. Leakage Energy Accounting

- [ ] 2.1 Add estimator helpers to compute total leakage power and burst leakage energy (`mW * ns = pJ`).
- [ ] 2.2 Integrate leakage energy into serialized-mode total burst energy before per-token normalization.
- [ ] 2.3 Integrate leakage energy into layer-pipelined-mode totals using pipelined effective burst latency.
- [ ] 2.4 Keep dynamic-energy accounting unchanged and additive (leakage added on top of existing dynamic totals).

## 3. Reporting and Assumptions

- [ ] 3.1 Extend report models/output to expose leakage summary (at least total leakage power and leakage energy per point).
- [ ] 3.2 Update estimator/report notes and README assumptions to document leakage units, formula, and schedule-aware latency source.

## 4. Validation

- [ ] 4.1 Add tests proving zero leakage reproduces previous energy/token results.
- [ ] 4.2 Add tests proving positive leakage increases burst/per-token energy by the expected amount.
- [ ] 4.3 Add tests proving schedule-aware leakage timing (serialized uses serialized burst time; layer-pipelined uses pipelined burst time).
- [ ] 4.4 Add report-shape tests for leakage summary fields and component-level leakage input acceptance.

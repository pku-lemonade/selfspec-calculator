## 1. Baseline Semantics

- [x] 1.1 Update `layer-pipelined` timing so `K=0` uses stop-and-go full-precision baseline latency
- [x] 1.2 Add regression tests for `K=0` baseline semantics in `layer-pipelined`

## 2. Verify Burst Semantics

- [x] 2.1 Update `layer-pipelined` verify latency/energy accounting to execute a fixed `K+1` burst independent of acceptance
- [x] 2.2 Keep acceptance-dependent committed-token normalization and commit-only writes while removing acceptance-dependent verify execution length
- [x] 2.3 Add regression tests proving verify burst latency/energy are fixed across acceptance outcomes for the same `K`

## 3. Reporting And Verification

- [x] 3.1 Update README/report metadata to describe stop-and-go baseline and fixed `K+1` verify-burst semantics
- [x] 3.2 Run targeted and full test suites covering pipeline semantics, report shape, and break-even comparisons

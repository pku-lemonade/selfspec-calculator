## 1. Science Reference Example Path

- [x] 1.1 Update the shipped Science reference example / smoke-test path to use `science_adi9405_v1_neurosim`
- [x] 1.2 Update README guidance so Science-oriented runs point at the detailed packaged library

## 2. RTL-Aligned Buffer Timing

- [x] 2.1 Refactor buffer/add accounting in `estimator.py` to compute streamed lane-parallel latency from ADC-lane parallelism
- [x] 2.2 Apply incremental overlap rules so draft/verify stream-aligned buffer work does not serialize after ADC output production
- [x] 2.3 Keep draft default-mode capture distinct from verify residual-add semantics in the estimator logic

## 3. Validation

- [x] 3.1 Add or update tests for Science example library selection and resolved-library reporting
- [x] 3.2 Add estimator tests for streamed / overlapped `buffers_add` latency behavior
- [x] 3.3 Run targeted CLI and estimator tests covering Science examples and SoC accounting

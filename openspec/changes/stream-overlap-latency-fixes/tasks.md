## 1. Legacy Stream Parity

- [x] 1.1 Port streamed / overlapped `buffers_add` latency accounting to the legacy estimator path
- [x] 1.2 Add regression tests proving legacy `buffers_add` latency no longer serializes over raw outputs

## 2. Analog Readout Overlap

- [x] 2.1 Refactor knob-based analog readout latency so ADC/TIA/SNH/MUX/IO use a streamed bottleneck path
- [x] 2.2 Add tests proving readout periphery no longer stacks as multiple full-scan wall-clock passes

## 3. Memory Transfer Overlap

- [x] 3.1 Replace serialized `SRAM + HBM + fabric` latency accumulation with a transfer-path bottleneck model
- [x] 3.2 Add tests proving HBM/fabric transfer latency no longer sums both wall-clock delays while energy remains additive

## 4. Verification

- [x] 4.1 Update README/report notes to document the overlap-aware timing rules
- [x] 4.2 Run targeted and full test suites covering accounting, SoC hardware, CLI examples, and legacy mode

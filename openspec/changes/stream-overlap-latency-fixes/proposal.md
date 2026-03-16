## Why

The recent streamed `buffers_add` fix only covered knob-based analog runs. The estimator still has the same serialized-latency failure mode in the legacy path, still sums multiple analog readout periphery blocks as separate full-scan wall-clock passes, and still treats memory hierarchy service time as `SRAM + HBM + fabric` rather than a streamed transfer path.

## What Changes

- Port streamed / overlapped `buffers_add` timing to the legacy estimator path so legacy and knob-based modes obey the same reuse/capture timing contract.
- Rework analog readout periphery timing so TIA/SNH/MUX/IO output-stream work is overlapped with ADC output production instead of added as separate serialized scan passes.
- Replace serialized memory-hierarchy latency accumulation with a streamed service model for memory movement so HBM/SRAM/fabric contribute as a transfer path rather than independent end-to-end phases.
- Add regression tests that catch these overlap bugs in legacy mode, knob-based periphery timing, and memory/fabric timing.

## Capabilities

### New Capabilities
- `legacy-overlap-parity`: Ensure legacy explicit-cost mode follows the same streamed buffer/add timing rules as knob-based mode.
- `streamed-readout-and-memory-latency`: Define overlap-aware timing for analog readout periphery and the memory/fabric transfer path.

### Modified Capabilities
- None.

## Impact

- Estimator logic:
  - `src/selfspec_calculator/estimator.py`
- Docs / report notes:
  - `README.md`
- Validation:
  - `tests/test_accounting.py`
  - `tests/test_soc_hardware_model.py`

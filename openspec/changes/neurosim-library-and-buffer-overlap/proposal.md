## Why

The current Science-oriented example path still defaults to the simplified `science_soc_v1` library, even though the repo already includes a more complete `science_adi9405_v1_neurosim` library with provenance, leakage, and richer SoC defaults. At the same time, the estimator charges `soc.buffers_add` latency as a fully serialized per-output cost, which makes draft-output capture unrealistically slower than the ADC path and does not match the Roadmap reuse contract or the placeholder RTL's one-result-per-cycle streaming behavior.

## What Changes

- Prefer the detailed `science_adi9405_v1_neurosim` library in the shipped Science reference example / documentation path so example-based analysis uses the most complete packaged library.
- Rework `buffers_add` latency accounting for draft / verify streaming paths so output capture and simple adds scale with ADC-lane parallelism instead of serializing over every output element.
- Use the reference RTL (`buffers_add_unit.v`, `quantize_dequantize_unit.v`) and Roadmap reuse description to document which buffer/add operations are modeled in draft, verify-drafted, and verify-bonus phases.
- Add regression tests that prove draft buffering no longer dominates ADC time purely because of per-output serialization and that the Science example path resolves to the detailed library.

## Capabilities

### New Capabilities
- `science-reference-library-profile`: Define the repo's Science-oriented example/profile path around the detailed `science_adi9405_v1_neurosim` packaged library.
- `rtl-aligned-buffer-timing`: Define lane-parallel, stream-oriented timing for draft/verify buffer and add operations using the reference RTL and Roadmap reuse contract.

### Modified Capabilities
- None.

## Impact

- Estimator logic:
  - `src/selfspec_calculator/estimator.py`
- Examples / docs:
  - `examples/hardware_soc_memory.yaml`
  - `examples/hardware_soc_area.yaml`
  - `README.md`
- Validation:
  - `tests/test_soc_hardware_model.py`
  - `tests/test_cli_soc_examples.py`

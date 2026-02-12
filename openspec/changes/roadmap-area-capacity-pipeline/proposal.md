## Why

The current SoC estimator is missing several Roadmap-critical behaviors: (1) area reporting does not include ADC/DAC/periphery/memory (and cannot report off-chip HBM separately), (2) `layer-pipelined` scheduling uses an overly idealized `/ n_layers` latency reduction instead of a fine-grained NPU-level bottleneck token period, and (3) there is no max-context capacity constraint to ensure sweeps satisfy `L_prompt + K` bounds. These gaps make the calculator’s outputs hard to interpret and misaligned with the Roadmap’s intended full-SoC evaluation.

## What Changes

- Extend area modeling and reporting to match Roadmap-style breakdown:
  - report on-chip area for arrays + DAC/ADC + analog periphery + buffers/control + (optional) SRAM/fabric,
  - report **off-chip HBM area separately** (not included in on-chip totals).
- Replace the simplified `layer-pipelined` latency model with a fine-grained **NPU-level pipeline throughput** model:
  - compute a steady-state **token period** bounded by the bottleneck stage (and shared resources like fabric/HBM bandwidth when configured),
  - keep energy accounting unchanged.
- Add an optional **max context capacity** knob and enforce Roadmap’s constraint (`L_prompt + K` must fit) during sweeps with clear error messages.
- Add tests + examples + docs to validate and demonstrate the above, without changing default behavior for existing configs.

## Capabilities

### New Capabilities
- `soc-area-capacity-pipeline`: Report on-chip vs off-chip area breakdown (incl. HBM separately), enforce `L_prompt + K` capacity constraints, and model NPU-level pipelined token period via a bottleneck throughput model.

### Modified Capabilities

<!-- None (no shared/global specs exist yet in openspec/specs/). -->

## Impact

- Config/schema:
  - extend knob-based `hardware.yaml` with optional capacity and area knobs (additive, backward-compatible).
- Estimator:
  - update scheduling math for `soc.schedule: layer-pipelined` to use a bottleneck token period model,
  - add capacity validation for sweeps (`L_prompt + K`).
- Reporting:
  - extend report area breakdown to include per-component area (on-chip) and report HBM area separately (off-chip).
- Validation:
  - add unit + CLI tests and update docs/examples for new knobs and semantics.

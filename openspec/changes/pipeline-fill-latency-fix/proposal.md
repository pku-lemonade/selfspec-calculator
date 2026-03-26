## Why

The current `layer-pipelined` verify timing still underestimates burst latency because it uses only the steady-state step period and omits pipeline fill. That makes multi-token verify bursts unrealistically cheap compared with the Roadmap's intended "first token serialized, subsequent tokens at bottleneck rate" behavior.

## What Changes

- Add pipeline fill latency to `layer-pipelined` verify timing.
- Make verify burst time scale as one full serialized verify token plus additional bottleneck periods, instead of just `n * bottleneck`.
- Add regression tests around verify burst timing.

## Capabilities

### New Capabilities
- `pipeline-fill-latency`: Model fill latency for layer-pipelined verify bursts.

### Modified Capabilities
- None.

## Impact

- `src/selfspec_calculator/estimator.py`
- `README.md`
- `tests/test_soc_hardware_model.py`

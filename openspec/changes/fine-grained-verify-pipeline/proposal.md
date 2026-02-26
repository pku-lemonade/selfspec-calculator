## Why

The current `layer-pipelined` latency model uses a coarse phase-level approximation (`K * draft + K * verify_drafted + 1 * verify_bonus`) that does not represent fine-grained verify wavefront execution or mismatch-stop behavior. We need a pipeline model that matches the intended hardware execution contract: serialized draft generation, pipelined verify, and immediate stop at first mismatch.

## What Changes

- Add a fine-grained verify-stage wavefront pipeline model at CIM-unit granularity for `soc.schedule: layer-pipelined`.
- Keep draft-stage token progression serialized (no token-level overlap in draft).
- Change verify execution policy to stop all further verify work at first mismatch.
- Define bonus token behavior inside the same verify pipeline wave:
  - if mismatch occurs at step `j`, the mismatch step output is the committed correction token;
  - if all `K` drafted tokens are accepted, execute one additional verify bonus step on the full drafted prefix and commit it.
- Replace fixed verify-step counting in pipeline timing with executed-step accounting based on accepted-prefix outcomes.
- Ensure pipeline latency and energy both account only for executed verify work under the mismatch-stop policy.
- Add reporting metadata for pipeline assumptions and stop-policy semantics to keep results auditable.
- Add targeted tests for early mismatch, mid-burst mismatch, full acceptance with bonus, and no post-mismatch residual work.

## Capabilities

### New Capabilities
- `verify-wavefront-pipeline`: Define fine-grained verify/bonus wavefront scheduling and resource bottleneck rules for layer-pipelined mode.
- `pipeline-stop-and-commit-policy`: Define mismatch-stop semantics, executed-step accounting, and token-commit behavior under acceptance outcomes.

### Modified Capabilities
- None.

## Impact

- Estimator timing/accounting logic:
  - `src/selfspec_calculator/estimator.py`
- Runtime/report outputs:
  - `src/selfspec_calculator/report.py`
  - `README.md`
- Validation:
  - `tests/test_soc_hardware_model.py`
  - new pipeline-focused tests (fine-grained wavefront + stop policy)

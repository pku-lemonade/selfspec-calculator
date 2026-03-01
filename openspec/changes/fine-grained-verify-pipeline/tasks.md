## 1. Pipeline Contract Wiring

- [x] 1.1 Add/adjust internal pipeline policy hooks so `layer-pipelined` uses fine-grained verify wavefront logic while draft remains serialized.
- [x] 1.2 Ensure mismatch-stop and full-accept bonus semantics are represented in the pipeline execution contract used by the estimator.

## 2. Verify Wavefront Scheduler

- [x] 2.1 Implement outcome-conditioned verify wavefront step scheduling (executed steps depend on accepted-prefix outcome `a`).
- [x] 2.2 Integrate bonus step into the same verify wavefront path for `a = K`.
- [x] 2.3 Preserve step-indexed context growth (`L_i = L_prompt + i`) for verify and bonus step cost generation.

## 3. Layer-Pipelined Timing and Energy Aggregation

- [x] 3.1 Replace fixed coarse verify timing (`K/K/+1`) with histogram-weighted outcome-conditioned burst latency aggregation.
- [x] 3.2 Update layer-pipelined burst energy accounting to charge only executed verify work under mismatch-stop behavior.
- [x] 3.3 Keep per-token normalization consistent with expected committed tokens under the same stop-and-commit policy.

## 4. Reporting and Assumption Transparency

- [x] 4.1 Add/extend report metadata to expose verify wavefront, mismatch-stop, and bonus semantics used in layer-pipelined mode.
- [x] 4.2 Update estimator notes/readme text describing the new pipeline assumptions and removal of post-mismatch tail charging.

## 5. Validation

- [x] 5.1 Add tests for early mismatch (`a=0`) proving verify stops immediately and no later verify/bonus work is charged.
- [x] 5.2 Add tests for mid-burst mismatch (`0 < a < K`) proving executed-step accounting and commit behavior (`a+1`).
- [x] 5.3 Add tests for full acceptance (`a=K`) proving `K+1` verify steps (including bonus) are executed and committed.
- [x] 5.4 Add regression tests showing draft-stage serialization behavior is unchanged by the fine-grained verify pipeline update.

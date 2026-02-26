## Context

The estimator currently has two timing paths: serialized and `layer-pipelined`. Serialized accounting is now step-indexed and captures full per-step costs, but the `layer-pipelined` timing path still uses a coarse phase-level approximation. It models draft/verify/bonus using fixed phase multipliers and phase bottlenecks, instead of a fine-grained verify wavefront that reflects executed steps under mismatch outcomes.

For this change, the intended execution contract is:
- Draft stage remains token-serial (next draft token depends on previous draft output).
- Verify runs as a fine-grained wavefront pipeline after draft inputs are available.
- Verification stops immediately at first mismatch.
- If mismatch occurs at step `j`, that mismatch step output is the committed correction token.
- If all `K` draft tokens are accepted, verify executes one additional bonus step on the full drafted prefix and commits it.

## Goals / Non-Goals

**Goals:**
- Replace coarse `layer-pipelined` verify timing with fine-grained wavefront scheduling.
- Model verify/bonus execution as outcome-conditioned executed steps (histogram-aware), not fixed always-on tail work.
- Preserve draft-stage serialized behavior.
- Ensure latency and energy in layer-pipelined mode include only work that is actually executed under mismatch-stop semantics.
- Keep pipeline assumptions explicit and auditable in reporting/tests.

**Non-Goals:**
- Redesigning serialized mode formulas.
- Introducing cycle-accurate microarchitectural simulation.
- Changing model topology assumptions (attention/FFN operation definitions).

## Decisions

### 1) Draft and verify use different timing abstractions
- **Decision:** Keep draft token progression serialized; apply fine-grained pipeline modeling only to verify (+ bonus).
- **Rationale:** Draft tokens are autoregressive and input-dependent. Verify inputs are known from drafted sequence and can be wavefront-scheduled.
- **Alternative considered:** Pipeline draft and verify symmetrically. Rejected due to draft data dependency chain.

### 2) Verify uses outcome-conditioned executed steps
- **Decision:** For accepted-prefix outcome `a`:
  - if `a < K`, executed verify steps are `a + 1` and stop;
  - if `a = K`, executed verify steps are `K + 1` (includes bonus).
- **Rationale:** Matches stop-on-mismatch and bonus-on-full-accept contract.
- **Alternative considered:** Always execute `K+1` verify steps regardless of mismatch. Rejected because it overcharges latency/energy under early mismatch.

### 3) Bonus is modeled as part of the verify wavefront
- **Decision:** Treat bonus as the terminal verify step in the same wavefront path (not a separate always-on tail block).
- **Rationale:** Bonus consumes the same verify resources and depends on the full accepted drafted prefix.
- **Alternative considered:** Keep bonus as an independent fixed post-phase term. Rejected as it duplicates tail timing semantics.

### 4) Pipeline timing aggregates by histogram outcomes
- **Decision:** Compute per-outcome burst latency/energy and aggregate by acceptance histogram probabilities.
- **Rationale:** Executed verify length is stochastic; expected metrics must follow `E[T_burst(a)]` and `E[E_burst(a)]`.
- **Alternative considered:** Reuse a single deterministic phase formula with expected `a`. Rejected because nonlinearity in bottlenecks can bias results.

### 5) Step-indexed verify costs are preserved inside pipeline model
- **Decision:** Use `L_i = L_prompt + i` (including `i=K` bonus step) in verify step timing/cost generation.
- **Rationale:** Keeps consistency with serialized accounting and avoids undercounting later verify steps.
- **Alternative considered:** Fixed context for all pipeline steps. Rejected due to systematic underestimation.

## Risks / Trade-offs

- **[Complexity increase]** More detailed scheduling logic is harder to reason about.  
  **Mitigation:** Isolate wavefront scheduler helpers and add table-driven tests for outcome cases.

- **[Resource-overlap assumptions]** Pipeline timing can be optimistic/pessimistic if overlap rules are unclear.  
  **Mitigation:** Define explicit ownership and bottleneck rules (compute vs memory ports) in spec scenarios.

- **[Result discontinuity vs old pipeline mode]** Existing users may see metric shifts compared to coarse model.  
  **Mitigation:** Document behavior change clearly and add regression fixtures that lock new semantics.

- **[Histogram sensitivity]** Latency/energy become more sensitive to acceptance distribution shape.  
  **Mitigation:** Add tests for extreme histograms (`a=0`, `a=K`) and mixed distributions.

## Migration Plan

1. Implement verify wavefront scheduling helpers and outcome-conditioned aggregation in `estimate_point` for `layer-pipelined` mode.
2. Replace fixed `K/K/+1` verify timing formula with executed-step-aware calculations.
3. Ensure layer-pipelined energy path matches executed work policy (no post-mismatch charging).
4. Update reports/notes to expose stop-policy and bonus semantics for auditability.
5. Add focused tests for mismatch-stop, full-accept bonus, and no-tail-work invariants.

Rollback strategy:
- Revert to existing coarse pipeline formula while keeping serialized path unchanged.

## Open Questions

- Should we preserve an optional legacy coarse pipeline mode for comparison/debugging, or fully replace it?
- Do we need a configurable drain/flush overhead constant for mismatch-stop transitions in the first version?

## Context

The current estimator is a closed-form analytical model with phase-level accounting (`draft`, `verify_drafted`, `verify_bonus`) and stage buckets (`qkv`, `wo`, `ffn`, `qk`, `pv`, `softmax`, `elementwise`, `kv_cache`, `buffers_add`, `control`). It already expands analog work into tiled/sliced reads and includes ADC scan reuse factors, but digital accounting is still coarse for serialized-mode validation.

The immediate need is to make serialized accounting auditable end-to-end: every required compute and movement step must either be explicitly modeled, explicitly approximated by a named proxy, or explicitly declared out-of-scope. Attention and FFN digital processing should move from broad buckets to independent DPU features with separate latency/energy coefficients.

## Goals / Non-Goals

**Goals:**
- Define a strict serialized accounting contract per burst phase and per token-step.
- Add independent DPU feature accounting for attention and FFN digital sub-operations.
- Add explicit stepwise context growth rules for serialized burst accounting.
- Separate compute vs movement accounting so missing pieces are visible in reports/specs.
- Preserve backward compatibility by providing default mappings from old coarse coefficients.

**Non-Goals:**
- Implementing layer-pipelined timing changes in this change (handled after serialized is validated).
- Building cycle-accurate event simulation.
- Redesigning hardware architecture assumptions outside serialized accounting completeness.

## Decisions

### 1) Serialized accounting uses an explicit step-indexed burst model

**Decision:** Define serialized burst accounting as explicit sums over step index `i`, with `L_i = L_prompt + i` for all step-dependent costs.

**Rationale:** This matches real autoregressive behavior inside a burst and avoids undercounting attention/softmax work by using a fixed prompt length for all steps.

**Alternatives considered:**
- Keep fixed `L_prompt` for all steps: simpler but systematically undercounts later steps in the burst.

### 2) DPU is decomposed into named feature terms with independent coefficients

**Decision:** Replace broad digital buckets with explicit DPU feature terms (for example: attention qk, attention softmax, attention pv, ffn activation, ffn gate multiply, residual add, norm, quant/dequant, kv pack/unpack).

**Rationale:** The user needs independent latency/energy control per digital feature and clear visibility of what dominates serialized timing.

**Alternatives considered:**
- Keep a single `elementwise` feature: rejected because it cannot represent FFN variants (especially SwiGLU gate path) or calibration differences.

### 3) Compute and movement are accounted separately, then merged

**Decision:** Keep separate accounting channels for:
- digital compute feature costs, and
- memory/buffer movement costs (KV and non-KV intermediates),
before merging into phase totals.

**Rationale:** This makes missing movement costs detectable and avoids hiding memory effects inside compute coefficients.

**Alternatives considered:**
- Fold movement into compute coefficients: rejected because it obscures what is modeled and prevents targeted calibration.

### 4) Legacy compatibility via deterministic mapping

**Decision:** Introduce a deterministic compatibility mapping from legacy coarse digital coefficients to the new DPU feature set when fine-grained values are not provided.

**Rationale:** Existing configs and tests must keep running while new feature-level knobs are introduced.

**Alternatives considered:**
- Hard break requiring all new coefficients: rejected because it would invalidate existing workflows.

## Risks / Trade-offs

- **[Parameter explosion]** More DPU coefficients increase config complexity.  
  Mitigation: provide grouped defaults and compatibility mapping from old fields.
- **[Double counting]** Some operations (for example quant/dequant or buffer writes) may be charged in both compute and movement paths.  
  Mitigation: define single ownership per feature in spec requirements and add invariance tests.
- **[Calibration uncertainty]** Independent features need reliable per-op calibration data.  
  Mitigation: allow explicit provenance tags and fallback coefficients.
- **[Performance impact on maintainability]** More detailed formulas increase code complexity.  
  Mitigation: keep helper functions per feature family and add reference tests per requirement scenario.

## Migration Plan

1. Add spec-level requirements for serialized completeness and DPU feature accounting.
2. Implement data model extensions for feature-level digital coefficients with compatibility defaults.
3. Refactor serialized accounting to step-indexed formulas using context growth.
4. Add missing non-KV movement accounting (or explicit exclusion markers where intentionally not modeled).
5. Add validation and golden tests for all new accounting scenarios.
6. Keep old report fields available; expose feature-level detail additively.

Rollback:
- Revert to coarse digital buckets and fixed-context serialized formulas while keeping new specs in draft change only.

## Open Questions

- Which DPU features should be mandatory vs optional in v1 of this change?
- Should non-KV intermediate movement be represented as explicit memory traffic fields in report output or as component-only costs?
- How should feature coefficients be grouped in library JSON to stay readable while remaining explicit?
- Do we need separate coefficients per model family (MLP vs SwiGLU) or only per feature type with operation counts carrying the difference?

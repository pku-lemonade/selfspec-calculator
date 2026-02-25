## 1. Serialized Step-Indexed Accounting

- [x] 1.1 Refactor serialized phase accounting to compute step-indexed costs using `L_i = L_prompt + i` for draft, verify_drafted, and verify_bonus steps.
- [x] 1.2 Update burst aggregation formulas to sum explicit per-step costs before normalizing by expected committed tokens.
- [x] 1.3 Add unit tests proving attention-dependent serialized costs increase with step index inside a burst.
- [x] 1.4 Add unit tests proving verify bonus uses `L_prompt + K`.

## 2. DPU Feature Model

- [x] 2.1 Define a feature-level DPU coefficient schema (latency/energy per feature) in config and runtime library loading.
- [x] 2.2 Add attention digital features with independent accounting terms (QK support, softmax, PV support).
- [x] 2.3 Add FFN digital features with FFN-type-aware operation counting (MLP vs SwiGLU, including gate multiply coverage).
- [x] 2.4 Add report breakdown fields for per-feature DPU latency/energy transparency.

## 3. Compute vs Movement Separation

- [x] 3.1 Refactor serialized accounting internals to keep compute costs and movement costs separate before final merge.
- [x] 3.2 Add explicit ownership rules to avoid double counting between compute features and movement features.
- [x] 3.3 Add movement accounting coverage for non-KV intermediates, or add explicit structured exclusion markers when not modeled.
- [x] 3.4 Add tests validating compute/movement separation and no-double-count invariants.

## 4. Backward Compatibility

- [x] 4.1 Implement deterministic fallback mapping from existing coarse digital coefficients to feature-level DPU coefficients.
- [x] 4.2 Ensure explicit feature coefficients override mapped defaults for matching features.
- [x] 4.3 Add regression tests to confirm existing example configs still run and produce stable outputs under compatibility mapping.

## 5. Validation and Reporting

- [x] 5.1 Extend serialized report outputs to expose per-feature DPU subtotals and compatibility-mapping provenance metadata.
- [x] 5.2 Add test cases for feature-level sensitivity (changing one feature coefficient only affects that feature contribution).
- [x] 5.3 Add test cases for attention parallelism scaling under feature-level accounting.
- [x] 5.4 Add tests for SwiGLU-specific FFN feature coverage.

## 6. Documentation and Examples

- [x] 6.1 Update README modeling assumptions to describe step-indexed serialized accounting and DPU feature-level coefficients.
- [x] 6.2 Add or update example hardware/library files demonstrating explicit DPU feature coefficients.
- [x] 6.3 Document explicitly modeled vs proxy-modeled vs excluded movement items for serialized mode.

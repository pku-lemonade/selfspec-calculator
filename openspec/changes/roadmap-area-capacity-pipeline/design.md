## Context

The current `selfspec-calculator` is an analytical (closed-form) estimator. The recent SoC-focused extensions added explicit modeling knobs for:
- memory traffic (SRAM buffer + off-chip HBM + on-chip fabric) as bytes-based energy/latency, and
- analog periphery + buffers/add/control overheads as knob-based components.

However, several Roadmap-critical aspects are still missing or simplified:
- **Area reporting is incomplete**: area is currently reported only as `{qkv, wo, ffn}` weight area plus a per-layer “digital overhead” term. The library/schema already contains per-unit area knobs for DAC/ADC and periphery, and `memory.*.area_mm2`, but these are not yet used.
- **`layer-pipelined` schedule is too idealized**: today it is implemented as dividing latency/token by `n_layers`, which does not reflect a fine-grained NPU-level pipeline and does not treat shared memory bandwidth as a potential throughput bottleneck.
- **No max context capacity constraint**: Roadmap expects sweeps to respect a max context capacity check using `L_prompt + K` as the within-burst upper bound, but the estimator currently has no such knob or validation.

This change implements a Roadmap-aligned area breakdown (including off-chip HBM reported separately), a fine-grained pipelined token-period model, and an optional capacity constraint, while preserving backward compatibility for existing configs and report consumers.

## Goals / Non-Goals

**Goals:**
- Add a **component-level area breakdown** using existing area knobs:
  - on-chip: arrays, DAC, ADC-draft, ADC-residual, analog periphery blocks, SRAM, fabric, and existing “digital overhead” area
  - off-chip: HBM area reported separately and **excluded** from on-chip totals
- Improve `soc.schedule: layer-pipelined` to report a **steady-state token period** bounded by the bottleneck:
  - NPU-level per-layer compute as a pipeline stage
  - shared resources (HBM/fabric bandwidth) treated as global throughput constraints when configured
  - energy accounting unchanged
- Add an optional **max context capacity** knob and enforce Roadmap’s constraint `L_prompt + K <= capacity` during sweeps with clear error messages.
- Maintain backward compatibility:
  - old configs remain valid (new knobs optional, default disabled/zero),
  - existing JSON fields remain present with stable semantics; new reporting is additive.

**Non-Goals:**
- Cycle-accurate or event-based simulation of pipeline fill/flush, arbitration, overlap, or instruction scheduling.
- Detailed floorplanning (placement/routing) or modeling chiplet/package constraints.
- Modeling multiple HBM stacks or detailed DRAM topology; HBM is a single aggregated off-chip “resource” for reporting.

## Decisions

### 1) Area reporting is additive and split into on-chip vs off-chip

**Decision:** Keep the existing stage-level `Report.area` output intact for backward compatibility, and add a new report field for a **component-level area breakdown** that explicitly separates:
- `on_chip_mm2` (counted toward chip area)
- `off_chip_hbm_mm2` (reported but not counted toward chip area)

**Rationale:** Existing consumers expect the current `area` shape. The Roadmap needs a richer area breakdown without breaking existing tooling. Off-chip HBM must be reported but not conflated with die area.

**Alternatives considered:**
- Replace `Report.area` with a new structured area object → rejected (breaking change).
- Add HBM into on-chip `digital_mm2` → rejected (mixes off-chip with on-chip).

### 2) Use existing per-unit area knobs with explicit unit-count proxies

**Decision:** Compute area for DAC/ADC/periphery using simple, explainable unit-count proxies derived from the analog tiling parameters.

Proposed unit-count anchors (knob-based analog mode):
- Let `tiles_total` be the total number of crossbar tiles instantiated across **all layers** and **all analog blocks**.
- DAC units per tile ≈ `xbar_size` (one DAC per row).
- ADC units per tile ≈ `xbar_size / num_columns_per_adc` (validated integer).
- Periphery units default to being tied to ADC-unit count (TIA/SNH/MUX) or tile count (switches), with clear documentation and optional future overrides if needed.

Memory area:
- `memory.sram.area_mm2` and `memory.fabric.area_mm2` contribute to on-chip area.
- `memory.hbm.area_mm2` contributes to **off-chip** HBM area reporting only.

**Rationale:** The estimator already has the tiling knobs and library per-unit areas. This yields a first-order Roadmap-aligned area breakdown without adding a floorplanner or requiring new configuration in the common case.

**Alternatives considered:**
- Require users to directly provide a full area breakdown (manual) → rejected (less reusable, defeats “parametric” intent).
- Derive periphery unit counts from dynamic activation counts → rejected (area depends on instantiated units, not how often they are used).

### 3) `layer-pipelined` reports bottleneck token period (NPU-level pipeline)

**Decision:** Interpret `soc.schedule: layer-pipelined` as a **spatially scaled full-chip** where each layer’s stationary weights reside on a dedicated NPU pipeline stage. The metric `latency_ns_per_token` becomes a steady-state **token period** rather than a serialized end-to-end latency.

Implementation approach:
- Compute a per-layer compute latency for each phase step (Draft / Verify-drafted / Verify-bonus) based on the per-layer precision policy and per-layer digital costs.
- Compute a shared memory service latency per step when `memory` is configured (based on bytes moved and configured bandwidth/latency), without dividing by layers.
- Define step token period as:
  - `T_step = max( max_layer_compute_latency, shared_memory_service_latency )`
- Burst time is approximated by:
  - `T_burst = K*T_draft + K*T_verify_drafted + 1*T_verify_bonus + verify_setup_per_burst`
- Divide by `E[a+1]` committed tokens to produce `latency_ns_per_token` and throughput.
- Energy accounting remains unchanged (all work still performed).

**Rationale:** This matches the Roadmap intuition: with sufficient NPUs to hold all weights, throughput is bottlenecked by the slowest pipeline stage and shared resources. It also avoids the incorrect behavior of dividing shared HBM/fabric time by `n_layers`.

**Alternatives considered:**
- Keep `/ n_layers` scaling → rejected (not bottleneck-based; mishandles shared memory).
- Full token timeline simulation (fill/flush, overlap) → rejected (out of scope for an analytical estimator).

### 4) Add an optional max-context capacity knob and enforce `L_prompt + K`

**Decision:** Extend the memory KV-cache config with an optional `max_context_tokens` (default: unset / no constraint). When set, enforce:
- `L_prompt + K <= max_context_tokens`
for every sweep point.

**Rationale:** Roadmap uses `L_prompt + K` as the within-burst maximum context touched and expects sweeps to be bounded by the hardware’s KV capacity.

**Alternatives considered:**
- Enforce capacity unconditionally with a guessed default → rejected (would break existing configs and assumptions).
- Enforce `L_prompt + K + 1` → rejected (Roadmap explicitly uses `L_prompt + K` and the difference is not material for this estimator).

## Risks / Trade-offs

- **[Area proxy mismatch]** Unit-count proxies (e.g., DAC per row, ADC per column-group) may differ from a real macro implementation → Mitigation: document proxies clearly; keep knobs parametric; allow future override knobs for unit counts if needed.
- **[Throughput overlap ignored]** Bottleneck model is conservative and ignores fine overlap/arbitration effects → Mitigation: keep model simple and explainable; extend later with optional overlap factors.
- **[Semantic shift for latency]** In `layer-pipelined`, `latency_ns_per_token` becomes token period, not end-to-end latency → Mitigation: add explicit notes in the report; keep serialized breakdowns available.
- **[Capacity check surprises]** Users may hit errors in existing sweeps when they enable capacity → Mitigation: include a clear error message with `L_prompt`, `K`, and the configured capacity, and suggest reducing the sweep or increasing capacity.

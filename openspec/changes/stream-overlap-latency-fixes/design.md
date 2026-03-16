## Context

The overlap audit found three same-class problems:

1. The legacy estimator path still charges `buffers_add` as a fully serialized per-output pass.
2. The knob-based analog path still sums TIA, SNH, MUX, and IO readout latency as separate full-scan wall-clock phases even though they sit on the same ADC output stream.
3. The memory path sums `SRAM + HBM + fabric` latency, which turns a transfer path into multiple serialized end-to-end phases.

The placeholder RTL is sufficient to justify stream-style throughput assumptions:
- `buffers_add_unit.v` is a one-result-per-cycle streaming operator.
- `memory_fabric_axi_dma.v` is a stream-handshaked DMA/fabric block.

The design goal is not cycle-accurate scheduling. It is to remove obviously over-serialized wall-clock accounting where multiple blocks are on the same stream.

## Goals / Non-Goals

**Goals:**
- Port the streamed `buffers_add` timing rule to legacy mode.
- Collapse analog output-stream timing so ADC + TIA + SNH + MUX + IO are modeled as a single streamed path.
- Model memory movement latency as a transfer path bottleneck instead of summing every hierarchy layer's wall-clock service time.
- Add targeted tests that lock in the corrected timing behavior.

**Non-Goals:**
- Do not split the schema into new standalone report components for every periphery block.
- Do not build a detailed NoC or controller simulator.
- Do not remove conservative energy accounting; this change is about latency semantics.

## Decisions

### 1. Legacy mode gets the same streamed `buffers_add` timing rule

Legacy explicit-cost mode will reuse the same helper semantics as knob-based mode for draft capture, verify residual combine, and bonus-token combine:
- energy remains proportional to output count,
- latency scales with streamed output steps,
- only incremental latency beyond the relevant stream is charged.

Rationale:
- the bug class is identical;
- legacy mode still exposes `soc.buffers_add` and should not regress into a different timing contract.

### 2. Analog output-stream periphery uses `max`, not a sum

For knob-based analog stages, TIA/SNH/MUX/IO will be treated as blocks on the same output stream as ADC scan.

Implementation rule:
- keep component energies separate,
- compute each component's streamed latency from the same scan-step anchor,
- set the stage's streamed output-path latency contribution to the maximum of:
  - ADC streamed latency,
  - TIA streamed latency,
  - SNH streamed latency,
  - MUX streamed latency,
  - IO streamed latency.

Array, DAC, switch, and write-driver terms remain outside that output-stream max because they are anchored to different activity.

Rationale:
- these blocks are serially connected on the same sample stream, not burst-wide post-passes;
- summing them creates the same kind of false slowdown that previously affected `buffers_add`.

### 3. Memory hierarchy latency uses a transfer-path bottleneck

Memory latency will be modeled as:
- local SRAM service time,
- external HBM-to-fabric transfer path time,
- write path analogously,
with streamed transfer latency derived from the bottleneck path rather than summed per layer of the hierarchy.

Implementation rule:
- keep energy summed across SRAM/HBM/fabric bytes,
- for latency, use a bottleneck-style composition:
  - local SRAM-only traffic contributes local SRAM time,
  - HBM-backed traffic contributes `max(hbm_service, fabric_service)` instead of `hbm + fabric`,
  - total read/write latency is the relevant path sum, not the sum of all components regardless of topology.

This preserves the existing analytical character while removing the most obvious over-serialization.

### 4. Preserve conservative semantics unless topology proves overlap

Only blocks that clearly share a stream are overlapped in this change.

Rationale:
- the estimator should remain explainable;
- this avoids replacing one arbitrary simplification with another.

## Risks / Trade-offs

- [Memory topology is still simplified] -> Mitigation: bottleneck transfer-path model is less pessimistic than full summation while staying analytical and auditable.
- [Periphery overlap may undercount if there are hidden staging bubbles] -> Mitigation: only output-stream blocks are overlapped; unrelated setup terms remain additive.
- [Legacy and knob-based paths may diverge again] -> Mitigation: use shared helper logic where possible and add parity-oriented tests.

## Migration Plan

1. Refactor shared stream helpers in `estimator.py`.
2. Apply those helpers to legacy `buffers_add` timing.
3. Rework knob-based analog periphery stage latency to use output-stream `max(...)` instead of full summation.
4. Rework memory latency helpers to use transfer-path bottlenecks while keeping energy unchanged.
5. Add tests for legacy buffer timing, periphery overlap, and memory-path overlap.

## Open Questions

- Should future work distinguish on-core SRAM-local traffic from off-core fabric traffic more explicitly in the schema?
- Should `softmax` / `elementwise` / `kv_cache_datapath` eventually move to a throughput-oriented digital pipeline model instead of `ops * delay`?

## Context

The repo already ships two Science-oriented packaged libraries:

- `science_soc_v1`: a simplified SoC-oriented profile used by current examples.
- `science_adi9405_v1_neurosim`: a more detailed profile with provenance, leakage, analog periphery, memory defaults, and DC / CACTI / Ramulator notes.

The current estimator also treats `soc.buffers_add.latency_ns_per_op` as a fully serialized per-output latency term. That is too conservative for draft-output capture and for streamed add/combine paths:

- the Roadmap says default draft stores `D_reg` for reuse, not `Final = D_reg + C`;
- `reference/dc_rtl/buffers_add_unit.v` is a 1-cycle streaming block with pass/add modes, not a burst-wide serialized engine;
- ADC timing already models lane parallelism through `xbar_size / num_columns_per_adc`, but `buffers_add` latency currently ignores that parallelism.

The change needs to make the Science reference path use the detailed packaged library and revise buffer/add timing so draft capture is modeled as a streamed path aligned with ADC output production.

## Goals / Non-Goals

**Goals:**
- Make the shipped Science reference example / smoke-test path use `science_adi9405_v1_neurosim`.
- Replace per-output serialized `buffers_add` latency with a stream-oriented timing model derived from ADC-lane parallelism.
- Distinguish draft capture semantics from verify residual-add semantics using the Roadmap reuse contract and the placeholder RTL.
- Add regression coverage for both the example-library selection and the new buffer timing behavior.

**Non-Goals:**
- Do not change the repo-wide generic default library (`puma_like_v1`).
- Do not introduce a cycle-accurate simulator or a new event-trace engine.
- Do not fully remodel every analog periphery block in this change.
- Do not add a new standalone `quantize_dequantize` report component in this change.

## Decisions

### 1. Prefer the detailed NeuroSim-backed library in the Science example path

The shipped Science-oriented example / validation path will be updated to use `science_adi9405_v1_neurosim` rather than `science_soc_v1`.

Rationale:
- it is the most complete packaged Science profile in the repo;
- it preserves the existing simplified library for quick experiments;
- it keeps the user-facing "Science reference" path aligned with the richest available source set.

Alternative considered:
- Change `HardwareConfig.DEFAULT_LIBRARY` globally.
  Rejected because it would silently affect generic non-Science configs and unrelated tests.

### 2. Model buffer/add latency as a streamed lane-parallel path

For `buffers_add` operations tied to ADC output streams, latency will no longer scale as `outputs * latency_ns_per_op`.

Instead, derive:
- `adc_lanes = xbar_size / num_columns_per_adc`
- `stream_steps = outputs / adc_lanes = base_reads * num_columns_per_adc`
- `stream_latency = stream_steps * latency_ns_per_op`

This matches the same lane parallelism already assumed by ADC scan latency.

Rationale:
- one output is produced per ADC lane per scan step;
- a per-lane buffer / add datapath should therefore scale with scan steps, not total outputs.

Alternative considered:
- Set draft buffer latency to zero.
  Rejected because capture and add logic still consumes latency when it becomes the slowest streaming stage.

### 3. Charge only incremental latency when buffer/add is on the same output stream as ADC

For draft capture, residual combine, and full-read ADC-output combine, the estimator will treat `buffers_add` as part of the same streaming pipeline as ADC output production.

Implementation rule:
- keep `buffers_add` energy proportional to element count;
- compute streamed `buffers_add` latency from `stream_steps`;
- add only the incremental latency beyond the already-modeled analog output-stream time for that stage.

This means a buffer path that is faster than ADC scan adds no extra wall-clock time, while a slower path stretches the stream by the excess only.

Rationale:
- the placeholder RTL is throughput-oriented (`out_valid <= in_valid` with one result per cycle);
- the user-visible problem is the incorrect serialized post-pass implied by the current additive model.

Alternative considered:
- Keep additive latency but divide only by ADC lanes.
  Rejected because it still models buffering as a separate serialized phase after ADC completion.

### 4. Keep schema changes minimal for now

This change will not split `quantize_dequantize_unit` into a new schema field, even though the reference RTL keeps it separate.

Rationale:
- the user request is focused on example-library selection and buffer timing;
- the current schema/report model can be corrected substantially without introducing a new top-level component;
- a future change can split `soc.buffers_add` into finer-grained components if needed.

## Risks / Trade-offs

- [Detailed library exposes other conservative timings] -> Mitigation: keep the change scoped to example selection plus buffer timing, and preserve simplified libraries for comparison.
- [Placeholder RTL is not full microarchitecture] -> Mitigation: use it only to justify operation classes and streaming throughput, not detailed placement or stall behavior.
- [Overlap model may undercount latency if downstream buffering is actually back-pressured] -> Mitigation: incremental-latency rule still charges the excess whenever buffer/add is slower than the analog output stream.
- [Schema still folds quant/dequant into `soc.buffers_add`] -> Mitigation: document this explicitly and avoid claiming component-level separation that the schema cannot represent yet.

## Migration Plan

1. Update the Science reference example / smoke-test path to load `science_adi9405_v1_neurosim`.
2. Refactor buffer/add accounting helpers in `estimator.py` to use streamed lane-parallel latency and incremental overlap.
3. Add tests for detailed-library example selection and the new buffer-latency scaling rule.
4. Re-run targeted CLI and estimator tests for Science examples and SoC hardware accounting.

## Open Questions

- Should a follow-up change split `quantize_dequantize_unit` out of `soc.buffers_add` in the schema and report?
- Should the same overlap treatment be extended to more analog periphery blocks (for example TIA/SNH/MUX) after this buffer-path correction lands?

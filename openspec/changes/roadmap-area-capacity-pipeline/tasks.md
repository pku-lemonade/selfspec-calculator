## 1. Config / Schema Extensions (Capacity Knob)

- [ ] 1.1 Add optional `memory.kv_cache.max_context_tokens` to knob-based `hardware.yaml` (default: unset / no constraint)
- [ ] 1.2 Add validation/tests that the new knob is backward compatible (old configs still load without new keys)

## 2. Report Schema: Component-Level Area Breakdown

- [ ] 2.1 Extend report models to include a component-level area breakdown with explicit `on_chip_mm2` and `off_chip_hbm_mm2` reporting
- [ ] 2.2 Preserve existing stage-level `Report.area` fields (`qkv/wo/ffn/digital` mm^2) with stable semantics (additive reporting only)

## 3. Estimator: Area Breakdown Computation + Wiring

- [ ] 3.1 Derive instantiated unit-count proxies from tiling knobs (e.g., total tiles, DAC-per-row, ADC-per-column-group) for area accounting
- [ ] 3.2 Compute on-chip component areas (arrays/DAC/ADC/periphery/SRAM/fabric/digital-overhead) from configured per-unit and `memory.*.area_mm2` knobs
- [ ] 3.3 Compute off-chip HBM area from `memory.hbm.area_mm2` and ensure it is excluded from on-chip totals
- [ ] 3.4 Add tests that non-zero periphery/memory area knobs produce non-zero reported areas and that HBM area is reported separately

## 4. Estimator: Max Context Capacity Enforcement

- [ ] 4.1 Enforce `L_prompt + K <= memory.kv_cache.max_context_tokens` for every sweep point when the knob is set (clear error message includes `L_prompt`, `K`, and capacity)
- [ ] 4.2 Add tests for both violation (errors) and disabled knob (no rejection)

## 5. Estimator: Fine-Grained Layer-Pipelined Token Period (Bottleneck Model)

- [ ] 5.1 Replace the current `layer-pipelined` `/ n_layers` latency scaling with a bottleneck token-period model (max per-layer compute vs shared memory service time)
- [ ] 5.2 Ensure shared resource service time (SRAM/HBM/fabric bytes-based latency) is treated as a global throughput constraint (not amortized by `n_layers`)
- [ ] 5.3 Update/add tests: energy unchanged vs serialized; latency <= serialized; add a memory-bottleneck case; and ensure behavior is not equivalent to naive `/ n_layers`

## 6. Examples + Docs

- [ ] 6.1 Update/add example `hardware.yaml` to demonstrate area knobs (periphery `area_mm2_per_unit`, `memory.*.area_mm2`) and off-chip HBM area reporting
- [ ] 6.2 Document new area fields, pipelined-token-period semantics, and the capacity check in `README.md`

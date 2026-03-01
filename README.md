# selfspec-calculator

Parametric hardware PPA estimator for the Self-Speculating Analog Architecture described in `Roadmap.md`.

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
ppa-calculator \
  --model examples/model.yaml \
  --hardware examples/hardware.yaml \
  --stats examples/stats.json \
  --prompt-lengths 64 128 256 \
  --output out/report.json
```

## `model.yaml`

`activation_bits` is required and is used with `analog.dac_bits` to compute serial slicing:

```yaml
activation_bits: 12
```

The calculator uses:

```text
num_slices = ceil(activation_bits / analog.dac_bits)
```

## `hardware.yaml` formats

The tool supports two mutually exclusive formats.

### 1) Knob-based (preferred)

```yaml
reuse_policy: reuse
library: puma_like_v1
# Optional: load libraries from a custom JSON file.
# If omitted, packaged default libraries are used.
library_file: library_custom_minimal.json

analog:
  xbar_size: 128
  num_columns_per_adc: 16
  dac_bits: 4
  adc:
    draft_bits: 4
    residual_bits: 12
```

Optional SoC extensions (all optional; default to zero/disabled so older configs keep working):

```yaml
soc:
  schedule: serialized  # or: layer-pipelined
  attention_cim_units: 1  # parallel SRAM-CIM units for QK/PV digital attention stages
  attention_cim_mac_area_mm2_per_unit: 0.0  # MAC logic area per attention SRAM-CIM unit (DC fill target)
  attention_cim_storage_bits_per_element: null  # optional override; default uses model.activation_bits
  verify_setup:
    energy_pj_per_burst: 0.0  # optional override (otherwise resolved from library, if available)
    latency_ns_per_burst: 0.0 # optional override (otherwise resolved from library, if available)
  buffers_add: {}  # optional override fields (otherwise resolved from library, if available)
  control: {}      # optional override fields (otherwise resolved from library, if available)
  # Any field provided here overrides the library (including explicit `0.0`).

memory:
  # Enables bytes-based KV-cache modeling (SRAM buffer + off-chip HBM + fabric).
  # PPA coefficients (energy/byte, bandwidth, latency, and optional area) can be provided explicitly,
  # but are typically resolved from the selected `library` (e.g., `science_soc_v1`).
  # Any field provided here overrides the library (including explicit `0.0`).
  sram: {}
  hbm: {}
  fabric: {}
  # Optional for area modeling of attention SRAM-CIM storage:
  # sram.capacity_bytes lets the tool derive area-per-byte from CACTI-style area+capacity.
  # If omitted, attention SRAM-CIM storage area defaults to 0.
  kv_cache:
    # Optional capacity check: enforce `L_prompt + K <= max_context_tokens` during sweeps.
    max_context_tokens: null
    hbm:
      value_bytes_per_elem: 1
      scale_bytes: 2
      scales_per_token_per_head: 2
    # sram: { ...optional override... }

analog:
  periphery:
    # Optional override fields (otherwise resolved from library, if available).
    # Any field provided here overrides the library (including explicit `0.0`).
    tia: {}
    snh: {}
    mux: {}
    io_buffers: {}
    subarray_switches: {}
    write_drivers: {}
```

Digital library schema:
- Coarse coefficients remain supported: `digital.attention`, `digital.softmax`, `digital.elementwise`, `digital.kv_cache`.
- Optional feature-level coefficients may be provided in `digital.features.*`:
  - `attention_qk`, `attention_softmax`, `attention_pv`, `ffn_activation`, `ffn_gate_multiply`, `kv_cache_update`.
- If a feature coefficient is missing, the calculator applies deterministic compatibility mapping from the coarse fields.

Examples:
- `examples/hardware_soc_memory.yaml` (SRAM + HBM + fabric KV-cache model)
- `examples/hardware_analog_periphery.yaml` (non-zero analog periphery + buffers/control)
- `examples/hardware_soc_area.yaml` (component area breakdown; off-chip HBM area reported separately)
- `examples/hardware_custom_library.yaml` (custom JSON library source via `library_file`, including explicit `digital.features.*`)

Validation rules:
- `xbar_size`, `num_columns_per_adc`, `dac_bits`, `draft_bits`, and `residual_bits` must be positive integers.
- `xbar_size % num_columns_per_adc == 0`.
- Requested ADC/DAC bit-widths must exist in the selected library.
- If `library` is omitted, default is `puma_like_v1`.
- `library_file` path handling:
  - if absolute, it is used as-is;
  - if relative, it is resolved relative to the `hardware.yaml` file location.
- Library source precedence:
  - if `library_file` is set, libraries are loaded only from that JSON file;
  - otherwise, packaged default libraries are used.
- Invalid custom library files fail fast with explicit errors (for example: file not found, malformed JSON, missing required sections like `adc/dac/array/digital`).

Paper provenance helper:
- `HardwareConfig.paper_library_extract("science_adi9405_2024")` returns a machine-readable extraction from the Science paper + supplement in `reference/`.
- `HardwareConfig.paper_library_missing_specs("science_adi9405_2024")` returns spec paths that are not provided by the paper (for example, missing bit-resolved ADC/DAC tables and `128x128` geometry).
- This extract is for provenance/gap analysis; it is not a complete runnable estimator library.

### 2) Legacy explicit-cost (backward compatible)

Keep existing `costs.*` format (see `examples/hardware_legacy.yaml`).

⚠️ Mixed configs are rejected: do not provide both `analog.*` and `costs.*` in the same file.

## Reporting

The JSON report includes:
- stage-level breakdown (`qkv`, `wo`, `ffn`, `qk`, `pv`, `softmax`, `elementwise`, `kv_cache`),
- component-level breakdown (`arrays`, `dac`, `adc_draft`, `adc_residual`, attention/digital components),
- optional SoC components (`buffers_add`, `control`, and analog periphery blocks),
- optional memory components (`sram`, `hbm`, `fabric`) and `memory_traffic` bytes,
- DPU feature subtotals per phase (`dpu_features`) with independent latency/energy for:
  - attention (`attention_qk`, `attention_softmax`, `attention_pv`)
  - FFN digital work (`ffn_activation`, `ffn_gate_multiply`)
  - KV update (`kv_cache_update`, when memory model is disabled)
- compute-vs-movement channel totals (`channels`) for each phase,
- DPU compatibility provenance (`dpu_feature_mapping`) indicating `explicit:*` vs `mapped:*` feature coefficients,
- movement coverage metadata (`movement_accounting`) with modeled/proxy/excluded movement items and ownership rules,
- resolved library entries (for knob-based runs),
- analog activation counts (`dac_conversions`, `adc_*_conversions`, etc.) for knob-based runs,
- baseline/delta and break-even fields compatible with previous outputs.
- area reporting:
  - stage-level `area` (`qkv/wo/ffn/digital` mm^2, backward compatible), and
  - component-level `area_breakdown_mm2` with `on_chip_mm2`, `off_chip_hbm_mm2`, and `on_chip_components` (arrays/DAC/ADC/periphery/SRAM/fabric/digital-overhead).

## Modeling assumptions

- This project is an analytical calculator (closed-form counting), not an event/instruction simulator.
- Serialized phase accounting is step-indexed:
  - draft and verify-drafted steps use `L_i = L_prompt + i` for `i = 0..K-1`,
  - verify-bonus uses `L_prompt + K`.
- `soc.schedule` affects how `latency_ns_per_token` is reported:
  - `serialized`: end-to-end serialized time per committed token.
  - `layer-pipelined`: draft remains serialized; verify uses outcome-conditioned wavefront execution.
    - mismatch (`a < K`): execute verify steps `0..a`, stop immediately, and commit `a+1` tokens.
    - full accept (`a = K`): execute `K` drafted verify steps plus one bonus step (`K+1` total committed).
    - no post-mismatch verifier tail work is charged.
- DPU feature coefficients support backward-compatible fallback:
  - missing `digital.features.*` entries are mapped from coarse digital coefficients.
- When `memory` is configured:
  - Serialized mode keeps fixed-phase KV reads (not mismatch-gated).
  - Layer-pipelined mode mismatch-gates verify-stage KV reads via outcome-conditioned executed-step accounting.
  - HBM KV writes are commit-only (Policy B).
  - ownership rule: KV update compute is excluded from DPU features and owned by movement accounting (`kv_cache` stage + `sram/hbm/fabric` components).
  - non-KV intermediate movement is currently excluded and explicitly reported under `movement_accounting.excluded`.
  - Optional capacity check: when `memory.kv_cache.max_context_tokens` is set, sweeps must satisfy `L_prompt + K <= max_context_tokens`.
- Full-read dual-ADC latency uses parallel timing:
  - energy sums ADC-Draft + ADC-Residual,
  - latency uses `max(adc_draft_scan, adc_residual_scan)`.

## Context

The `inputs/` directory contains simulator outputs keyed by `K`, with mean accepted drafted tokens already computed under `results_by_k.<K>.mean_accepted`. The calculator already has a reusable `evaluate_k_sweep` utility that can rank candidate `K` values once given `(K, E[a])` pairs. What is missing is the glue:

- scan the input directory,
- infer the model config for each file,
- map ADC settings from the input into a hardware config,
- run the `K` sweep,
- write one output file per input,
- and emit a compact summary table.

## Goals / Non-Goals

**Goals:**
- Reuse the current `ppa-k-sweep` machinery instead of re-implementing scoring logic.
- Support the current `inputs/*.json` shape directly.
- Produce both detailed per-run JSON outputs and a compact summary table.
- Keep the workflow configurable but defaulted for the current paper setup.

**Non-Goals:**
- Do not change the simulator input format.
- Do not require users to manually list files one by one.
- Do not build a new report schema for the main calculator.

## Decisions

### 1. Add a dedicated batch CLI

Add a separate CLI, tentatively `ppa-batch-inputs`, instead of overloading `ppa-k-sweep`.

Rationale:
- input discovery, model inference, and summary writing are distinct from the single-sweep workflow;
- easier to automate on a folder.

### 2. Use `results_by_k.<K>.mean_accepted` as the acceptance source

For each input file:
- extract candidate `K` values from `results_by_k`,
- use `mean_accepted` as `E[a]`,
- build a `KSweepInput`,
- run `evaluate_k_sweep`.

Rationale:
- this is the directly relevant signal for current calculator semantics;
- avoids inventing another intermediate format.

### 3. Infer model config from checkpoint/path patterns

For the current repo workflow, map inputs to packaged model YAMLs via checkpoint path / file name patterns:
- `Qwen3-0.6B` -> `examples/model_qwen3_0p6b.yaml`
- `Qwen3-1.7B` -> `examples/model_qwen3_1p7b.yaml`
- `Llama-3.2-1B` -> `examples/model_llama3_2_1b.yaml`

Rationale:
- current inputs already encode the model identity in stable strings;
- this avoids duplicating model descriptors inside input files.

### 4. Apply ADC-bit overrides on top of a hardware template

Use a hardware template path (defaulting to the paper workflow template) and override:
- `analog.adc.draft_bits`
- `analog.adc.residual_bits`

from the input file knobs.

Rationale:
- all current inputs vary ADC settings while sharing the same estimator hardware context;
- this keeps the batch tool simple and auditable.

### 5. Summary table uses best-throughput candidate per file

The summary file will use the `best_throughput` candidate for each input file and report:
- model
- ADC bits
- selected `K`
- acceptance rate (`E[a]/K`)
- final PPA

`final PPA` will be rendered as a compact metrics string containing latency/token, throughput, tokens/J, and on-chip area.

Rationale:
- "selected K" implies one winning row per input file;
- throughput is the default optimization target for the current speculative-decoding discussion.

## Risks / Trade-offs

- [Model inference is pattern-based] -> Mitigation: fail fast on unknown input files with a clear error.
- [Summary collapses to one winning row] -> Mitigation: detailed per-run JSON outputs still preserve the full candidate sweep.
- [PPA string is compact rather than fully structured] -> Mitigation: include detailed JSON outputs per file; keep summary table concise for paper use.

## Migration Plan

1. Add a parser for the current `inputs/*.json` format.
2. Add model/hardware resolution helpers.
3. Add the batch CLI and per-run output writing.
4. Add summary table generation.
5. Add tests and README usage notes.

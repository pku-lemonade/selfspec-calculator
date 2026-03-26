## Why

The repo now has a directory of simulator `K`-sweep inputs under `inputs/`, but there is no batch utility that consumes those files, runs the calculator for each one, and emits a consolidated paper-friendly summary. Doing that manually is repetitive and error-prone.

## What Changes

- Add a batch runner that scans an input directory of simulator `K`-sweep JSON files and evaluates each file with the calculator.
- Write one output JSON per input file containing the per-`K` calculator results.
- Generate a summary file with a compact table containing: model, ADC bits, selected `K`, acceptance rate, and final PPA.
- Allow the batch runner to use a configurable hardware template, with a sensible default for the current paper workflow.

## Capabilities

### New Capabilities
- `batch-input-runner-summary`: Batch-process simulator `K`-sweep input files into per-run outputs and a compact summary table.

### Modified Capabilities
- None.

## Impact

- New batch CLI / utility code under `src/selfspec_calculator/`
- `pyproject.toml` console script entry
- `README.md`
- tests for input parsing, batch output generation, and summary table generation

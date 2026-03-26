## ADDED Requirements

### Requirement: Batch utility SHALL read simulator `K`-sweep inputs from a directory
The system SHALL provide a batch utility that scans a directory of simulator `K`-sweep JSON files and loads candidate `K` values plus mean accepted drafted tokens from each file.

#### Scenario: Input directory produces one batch job per JSON file
- **WHEN** the user runs the batch utility on a directory containing multiple valid simulator input files
- **THEN** the utility evaluates each file as a separate batch job

### Requirement: Batch utility SHALL write one detailed output per input file
For each valid input file, the utility SHALL write one output JSON file containing the calculator results for all candidate `K` values in that input.

#### Scenario: Per-run output file is created
- **WHEN** the batch utility evaluates an input file
- **THEN** it writes a corresponding output JSON file under the requested output directory

### Requirement: Batch utility SHALL emit a compact summary table
The batch utility SHALL emit a summary file containing a table with the following columns:
- model
- ADC bits
- selected `K`
- acceptance rate
- final PPA

#### Scenario: Summary file includes one row per input file
- **WHEN** the batch utility evaluates multiple input files
- **THEN** the summary table contains one row per input file using the selected best candidate for that file

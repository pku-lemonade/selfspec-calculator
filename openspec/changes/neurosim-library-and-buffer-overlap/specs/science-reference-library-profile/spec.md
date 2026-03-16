## ADDED Requirements

### Requirement: Science reference example SHALL use the detailed packaged library
The system SHALL ship a Science-oriented example / validation path that resolves the packaged `science_adi9405_v1_neurosim` library instead of the simplified `science_soc_v1` profile.

#### Scenario: Science memory example resolves detailed library
- **WHEN** the user loads the shipped Science memory example hardware configuration
- **THEN** `HardwareConfig.selected_library` is `science_adi9405_v1_neurosim`

#### Scenario: Science example report exposes detailed resolved library metadata
- **WHEN** the user runs the CLI with the shipped Science reference hardware configuration
- **THEN** the JSON report's `resolved_library.name` is `science_adi9405_v1_neurosim`

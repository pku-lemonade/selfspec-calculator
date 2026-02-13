from selfspec_calculator.config import HardwareConfig


def test_science_adi9405_extract_has_expected_array_geometries() -> None:
    extract = HardwareConfig.paper_library_extract("science_adi9405_2024")
    geoms = extract["extracted_specs"]["array_geometry"]

    assert any(g["rows"] == 256 and g["cols"] == 256 for g in geoms)
    assert any(g["rows"] == 128 and g["cols"] == 64 for g in geoms)


def test_science_adi9405_extract_reports_missing_128x128_and_adc_tables() -> None:
    missing = HardwareConfig.paper_library_missing_specs("science_adi9405_2024")

    assert "array_geometry.128x128" in missing
    assert "adc.energy_pj_per_conversion_by_bits" in missing
    assert "dac.energy_pj_per_conversion_by_bits" in missing

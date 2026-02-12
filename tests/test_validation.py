from pathlib import Path

import pytest

from selfspec_calculator.config import HardwareConfig, ModelConfig


def test_model_yaml_missing_activation_bits_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
n_layers: 2
d_model: 64
n_heads: 8
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="activation_bits"):
        ModelConfig.from_yaml(path)


def test_model_yaml_invalid_activation_bits_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
n_layers: 2
d_model: 64
n_heads: 8
activation_bits: 0
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="activation_bits"):
        ModelConfig.from_yaml(path)


def test_hardware_yaml_invalid_reuse_policy(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        """
reuse_policy: invalid
costs:
  analog_draft: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  analog_full: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  analog_verify_reuse: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  digital_attention: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  digital_softmax: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  digital_elementwise: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  kv_cache: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  analog_weight_area: { area_mm2_per_weight: 0.0 }
  digital_overhead_area_mm2_per_layer: 0.0
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid hardware config"):
        HardwareConfig.from_yaml(path)


def test_draft_policy_invalid_layer_index_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
n_layers: 2
d_model: 64
n_heads: 8
activation_bits: 8
draft_policy:
  per_layer:
    3:
      qkv: full
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid layer index"):
        ModelConfig.from_yaml(path)


def test_knob_config_divisibility_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        """
reuse_policy: reuse
analog:
  xbar_size: 128
  num_columns_per_adc: 24
  dac_bits: 4
  adc:
    draft_bits: 4
    residual_bits: 12
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be divisible"):
        HardwareConfig.from_yaml(path)


def test_mixed_knob_and_legacy_config_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        """
reuse_policy: reuse
analog:
  xbar_size: 128
  num_columns_per_adc: 16
  dac_bits: 4
  adc:
    draft_bits: 4
    residual_bits: 12
costs:
  analog_draft: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  analog_full: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  analog_verify_reuse: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  digital_attention: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  digital_softmax: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  digital_elementwise: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  kv_cache: { energy_pj_per_mac: 0.0, latency_ns_per_mac: 0.0 }
  analog_weight_area: { area_mm2_per_weight: 0.0 }
  digital_overhead_area_mm2_per_layer: 0.0
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        HardwareConfig.from_yaml(path)


def test_unknown_library_bit_width_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        """
reuse_policy: reuse
library: puma_like_v1
analog:
  xbar_size: 128
  num_columns_per_adc: 16
  dac_bits: 7
  adc:
    draft_bits: 4
    residual_bits: 12
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Available DAC bits"):
        HardwareConfig.from_yaml(path)


def test_kv_cache_max_context_tokens_is_optional(tmp_path: Path) -> None:
    path = tmp_path / "hardware.yaml"
    path.write_text(
        """
reuse_policy: reuse
library: puma_like_v1
analog:
  xbar_size: 128
  num_columns_per_adc: 16
  dac_bits: 4
  adc:
    draft_bits: 4
    residual_bits: 12
memory:
  kv_cache:
    hbm:
      value_bytes_per_elem: 1
""".lstrip(),
        encoding="utf-8",
    )
    hw = HardwareConfig.from_yaml(path)
    assert hw.memory is not None
    assert hw.memory.kv_cache.max_context_tokens is None

    path.write_text(
        """
reuse_policy: reuse
library: puma_like_v1
analog:
  xbar_size: 128
  num_columns_per_adc: 16
  dac_bits: 4
  adc:
    draft_bits: 4
    residual_bits: 12
memory:
  kv_cache:
    max_context_tokens: 4096
    hbm:
      value_bytes_per_elem: 1
""".lstrip(),
        encoding="utf-8",
    )
    hw = HardwareConfig.from_yaml(path)
    assert hw.memory is not None
    assert hw.memory.kv_cache.max_context_tokens == 4096

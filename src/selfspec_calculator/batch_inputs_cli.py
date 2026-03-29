from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .batch_inputs import run_batch_inputs


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path not found: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppa-batch-inputs", add_help=True)
    parser.add_argument(
        "--inputs-dir",
        type=_existing_path,
        default=Path("inputs"),
        help="Directory containing simulator K-sweep JSON inputs (default: inputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out") / "batch_inputs",
        help="Directory to write per-run outputs and summary files (default: out/batch_inputs)",
    )
    parser.add_argument(
        "--hardware-template",
        type=_existing_path,
        default=Path("examples") / "hardware_soc_area.yaml",
        help="Hardware template YAML used for all runs before ADC-bit overrides (default: examples/hardware_soc_area.yaml)",
    )
    parser.add_argument(
        "--force-verify-adc-bits",
        type=int,
        default=None,
        help="Override verify ADC bits for all runs (default: use input file value)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_batch_inputs(
            inputs_dir=args.inputs_dir,
            output_dir=args.output_dir,
            hardware_template_path=args.hardware_template,
            repo_root=Path.cwd(),
            force_verify_adc_bits=args.force_verify_adc_bits,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    processed_files = len({row.input_file for row in summary.rows})
    print(f"processed {processed_files} input files")
    print(f"wrote outputs to {args.output_dir}")
    print(f"summary table: {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

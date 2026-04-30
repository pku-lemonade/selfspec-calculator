from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .hyflexpim import DEFAULT_MODEL_PATHS, HyFlexPIMConfig, build_report, write_report_outputs


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppa-hyflexpim", add_help=True)
    parser.add_argument(
        "--models",
        nargs="+",
        type=_existing_path,
        default=list(DEFAULT_MODEL_PATHS),
        help="One or more model YAMLs (default: current Qwen/Llama example models)",
    )
    parser.add_argument("--prompt-length", type=int, default=64, help="Decode prompt length (default: 64)")
    parser.add_argument("--generated-tokens", type=int, default=32, help="Generated tokens per request (default: 32)")
    parser.add_argument("--slc-rate", type=float, default=0.20, help="Fraction of SVD ranks mapped to SLC (default: 0.20)")
    parser.add_argument(
        "--svd-rank-scale",
        type=float,
        default=1.0,
        help="Scale applied to HyFlexPIM's hard-threshold SVD rank (default: 1.0)",
    )
    parser.add_argument(
        "--tensor-pus-per-layer",
        type=int,
        default=None,
        help="Override automatic PUs per layer derived from capacity and spare PUs",
    )
    parser.add_argument(
        "--analog-power-mode",
        choices=["active_pu", "module_utilized"],
        default="active_pu",
        help="Analog energy model: full active PU power or only assigned-module fraction (default: active_pu)",
    )
    parser.add_argument(
        "--throughput-mode",
        choices=["single_stream", "paper_pipeline"],
        default="single_stream",
        help="Reported tok/s mode: autoregressive single request or HyFlexPIM paper-style layer pipeline",
    )
    parser.add_argument(
        "--no-digital-attention",
        action="store_true",
        help="Disable digital QK/softmax/PV timing and energy for sensitivity checks",
    )
    parser.add_argument(
        "--no-nonlinear-ops",
        action="store_true",
        help="Disable SFU nonlinear timing and energy for sensitivity checks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write JSON/CSV/Markdown outputs to this directory instead of printing JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = HyFlexPIMConfig(
            slc_rate=args.slc_rate,
            svd_rank_scale=args.svd_rank_scale,
            tensor_pus_per_layer=args.tensor_pus_per_layer,
            analog_power_mode=args.analog_power_mode,
            throughput_mode=args.throughput_mode,
            include_digital_attention=not args.no_digital_attention,
            include_nonlinear_ops=not args.no_nonlinear_ops,
        )
        report = build_report(
            model_paths=args.models,
            prompt_len=args.prompt_length,
            generated_tokens=args.generated_tokens,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        write_report_outputs(report, args.output_dir)
        print(f"wrote HyFlexPIM simulator outputs to {args.output_dir}")
        print(f"summary table: {args.output_dir / 'summary.md'}")
        return 0

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .sota_baselines import DEFAULT_MODEL_PATHS, build_report, write_report_outputs


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppa-sota-baselines", add_help=True)
    parser.add_argument(
        "--models",
        nargs="+",
        type=_existing_path,
        default=list(DEFAULT_MODEL_PATHS),
        help="One or more model YAMLs (default: current Qwen/Llama example models)",
    )
    parser.add_argument("--prompt-length", type=int, default=64, help="Decode prompt length (default: 64)")
    parser.add_argument("--generated-tokens", type=int, default=32, help="Generated tokens per request (default: 32)")
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
        report = build_report(
            model_paths=args.models,
            prompt_len=args.prompt_length,
            generated_tokens=args.generated_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        write_report_outputs(report, args.output_dir)
        print(f"wrote SOTA baseline outputs to {args.output_dir}")
        print(f"summary table: {args.output_dir / 'summary.md'}")
        return 0

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

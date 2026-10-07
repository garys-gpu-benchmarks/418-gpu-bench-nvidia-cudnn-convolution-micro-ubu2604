#!/usr/bin/env python3
"""Harness adapter. run_benchmark.sh calls scripts/collect_workload.py."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HELPER_NAMES = (
    'collect_cudnn_conv.py',
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--config", default="config/benchmark_config.yaml")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--output-format", default="csv")
    args, extra = parser.parse_known_args()
    here = Path(__file__).resolve().parent
    helper = next((here / name for name in HELPER_NAMES if (here / name).is_file()), None)
    if helper is None:
        raise SystemExit("[FAIL] component collector not found beside collect_workload.py")
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    probed = subprocess.run([sys.executable, str(helper), "--help"], capture_output=True, text=True)
    help_text = (probed.stdout or "") + (probed.stderr or "")
    cmd = [sys.executable, str(helper), "--profile", args.profile, "--config", args.config]
    wants_run_dir = "--run-dir" in help_text
    wants_raw = "--raw-file" in help_text
    wants_output = "--output" in help_text
    if wants_output and not wants_run_dir:
        cmd.extend(["--output", args.raw_file])
    else:
        if wants_run_dir:
            cmd.extend(["--run-dir", args.run_dir])
        if wants_raw:
            cmd.extend(["--raw-file", args.raw_file])
        if wants_output:
            cmd.extend(["--output", args.raw_file])
    if "--output-format" in help_text:
        cmd.extend(["--output-format", args.output_format])
    # Workbook model_config is display text ("custom"). Collectors that take
    # the flag want the harness token. Real shapes stay in --config.
    if "--model-config" in help_text and "--model-config" not in extra:
        cmd.extend(["--model-config", "true"])
    cmd.extend(extra)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

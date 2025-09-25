#!/usr/bin/env python3
"""Unified entry point for temporal accuracy evaluation benchmarks.

Usage:
    python temporal_accuracy.py --benchmark one-object
    python temporal_accuracy.py --benchmark two-object --videos_path path/to/videos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict


BENCHMARK_OPTIONS: Dict[str, Dict[str, str]] = {
    "one-object": {
        "videos_path": "outputs_one_object_benchmark",
        "output_path": "outputs_one_object_benchmark",
        "csv_file": "data/one_object.csv",
        "result_filename": "temporal_accuracy_one_object.json",
    },
    "two-object": {
        "videos_path": "outputs_two_objects_benchmark",
        "output_path": "outputs_two_objects_benchmark",
        "csv_file": "data/two_objects.csv",
        "result_filename": "temporal_accuracy_two_objects.json",
    },
    "action": {
        "videos_path": "outputs_action_benchmark",
        "output_path": "outputs_action_benchmark",
        "csv_file": "data/action.csv",
        "result_filename": "temporal_accuracy_action.json",
    },
}


def run_one_object(videos_path: str, output_path: str, csv_file: str) -> None:
    from metrics.temporal_accuracy_one_object import main as eval_main

    # Delay import of argparse.Namespace until needed to avoid polluting global namespace
    namespace = argparse.Namespace(
        videos_path=videos_path,
        output_path=output_path,
        csv_file=csv_file,
    )
    eval_main(namespace)


def run_two_object(videos_path: str, output_path: str, csv_file: str) -> None:
    from metrics.temporal_accuracy_two_objects import main as eval_main

    namespace = argparse.Namespace(
        videos_path=videos_path,
        output_path=output_path,
        csv_file=csv_file,
    )
    eval_main(namespace)


def run_action(videos_path: str, output_path: str, csv_file: str) -> None:
    from metrics.temporal_accuracy_action import process_videos_from_csv

    results = process_videos_from_csv(videos_path, csv_file)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "temporal_accuracy_action.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved temporal accuracy action metric results to {output_file}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal accuracy evaluation for a specified benchmark.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=BENCHMARK_OPTIONS.keys(),
        help="Benchmark to evaluate (one-object, two-object, action)",
    )
    parser.add_argument(
        "--videos_path",
        help="Directory containing generated videos to evaluate. Defaults to benchmark-specific path.",
    )
    parser.add_argument(
        "--output_path",
        help="Directory where evaluation JSON will be written. Defaults to videos_path or benchmark-specific path.",
    )
    parser.add_argument(
        "--csv_file",
        help="CSV file describing prompts and control signals. Defaults to benchmark-specific CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    defaults = BENCHMARK_OPTIONS[args.benchmark]

    videos_path = args.videos_path or defaults["videos_path"]
    output_path = args.output_path or defaults["output_path"] or videos_path
    csv_file = args.csv_file or defaults["csv_file"]

    if output_path is None:
        output_path = videos_path

    if not os.path.exists(videos_path):
        raise FileNotFoundError(f"Videos path does not exist: {videos_path}")

    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file does not exist: {csv_file}")

    os.makedirs(output_path, exist_ok=True)

    if args.benchmark == "one-object":
        run_one_object(videos_path, output_path, csv_file)
    elif args.benchmark == "two-object":
        run_two_object(videos_path, output_path, csv_file)
    elif args.benchmark == "action":
        run_action(videos_path, output_path, csv_file)
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - propagate for CLI visibility
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

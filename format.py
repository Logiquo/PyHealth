import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SKIP_KEYS = {"experiment", "trial_number", "seeds", "params"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print compact avg/std summaries from benchmark JSON files."
    )
    parser.add_argument("filenames", nargs="+", help="JSON result file(s) to format.")
    return parser.parse_args()


def metric_summaries(run: Mapping[str, Any]) -> list[str]:
    summaries = []
    for key, value in run.items():
        if key in SKIP_KEYS or not isinstance(value, Mapping):
            continue
        if "avg" not in value or "std" not in value:
            continue
        summaries.append(key)
    return summaries


def format_file(filename: str) -> None:
    path = Path(filename)
    with path.open() as f:
        data = json.load(f)

    runs = data.get("runs", [])
    metric_keys = []
    for run in runs:
        for metric_key in metric_summaries(run):
            if metric_key not in metric_keys:
                metric_keys.append(metric_key)

    for i, metric_key in enumerate(metric_keys):
        if i:
            print()
        print(metric_key)
        for run in runs:
            value = run.get(metric_key)
            if not isinstance(value, Mapping):
                continue
            print(f"{float(value['avg']):.4f}±{float(value['std']):.4f}")


def main() -> None:
    args = parse_args()
    for filename in args.filenames:
        format_file(filename)


if __name__ == "__main__":
    main()

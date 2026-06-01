import argparse
import json
from pathlib import Path


DEFAULT_KEYS = [
    "subject_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "imaging_quality",
    "aesthetic_quality",
    "dynamic_degree",
]


def scalarize(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        for item in value:
            try:
                return scalarize(item)
            except ValueError:
                continue
    if isinstance(value, dict):
        for key in ("score", "mean", "overall"):
            if key in value:
                return scalarize(value[key])
        for item in value.values():
            try:
                return scalarize(item)
            except ValueError:
                continue
    raise ValueError(f"Cannot scalarize {value!r}")


def main(args):
    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    keys = args.keys or DEFAULT_KEYS
    summary = {}
    for key in keys:
        if key not in data:
            continue
        try:
            summary[key] = scalarize(data[key])
        except ValueError:
            summary[key] = data[key]
    output = {
        "source": str(result_path),
        "summary": summary,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--keys", nargs="*")
    main(parser.parse_args())

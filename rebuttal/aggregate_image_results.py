import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(args):
    run_root = Path(args.run_root)
    rows = []
    for method_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        run_metrics = load_json(method_dir / "run_metrics.json")
        eval_metrics = load_json(method_dir / "eval_metrics.json")
        image_count = len(list(method_dir.glob("*.jpg")))
        rows.append(
            {
                "method": method_dir.name,
                "images": image_count,
                "seconds_per_image": run_metrics.get("seconds_per_image"),
                "total_generation_seconds": run_metrics.get("total_generation_seconds"),
                "peak_memory_gb": run_metrics.get("peak_memory_gb"),
                "mean_actual_nfe": run_metrics.get("mean_actual_nfe"),
                "mean_skip_count": run_metrics.get("mean_skip_count"),
                "clip_score": eval_metrics.get("clip_score"),
                "image_reward": eval_metrics.get("image_reward"),
            }
        )

    output = {"run_root": str(run_root), "rows": rows}
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")

    headers = ["method", "images", "sec/img", "NFE", "skip", "peak GB", "CLIPScore", "ImageReward"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| {method} | {images} | {seconds_per_image:.3f} | {mean_actual_nfe:.1f} | {mean_skip_count:.1f} | {peak_memory_gb:.2f} | {clip_score} | {image_reward} |".format(
                method=row["method"],
                images=row["images"],
                seconds_per_image=row["seconds_per_image"] or 0,
                mean_actual_nfe=row["mean_actual_nfe"] or 0,
                mean_skip_count=row["mean_skip_count"] or 0,
                peak_memory_gb=row["peak_memory_gb"] or 0,
                clip_score=(
                    f"{row['clip_score']:.4f}" if isinstance(row["clip_score"], float) else "TBD"
                ),
                image_reward=(
                    f"{row['image_reward']:.4f}" if isinstance(row["image_reward"], float) else "TBD"
                ),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json")
    main(parser.parse_args())

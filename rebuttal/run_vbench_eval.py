import argparse
import json
from pathlib import Path

import torch
from vbench import VBench
from vbench.distributed import dist_init


def patch_torch_load_for_legacy_checkpoints() -> None:
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def main(args):
    patch_torch_load_for_legacy_checkpoints()
    dist_init()

    prompt_list = {}
    if args.prompt_file:
        prompt_list = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))

    kwargs = {"imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing_mode}
    evaluator = VBench(torch.device("cuda"), args.full_json_dir, args.output_path)
    evaluator.evaluate(
        videos_path=args.videos_path,
        name=args.name,
        prompt_list=prompt_list,
        dimension_list=args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.mode,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--name", default="results")
    parser.add_argument("--full_json_dir", default="/home/zs89/AeroWorld/.venv_vbench/lib/python3.12/site-packages/vbench/VBench_full_info.json")
    parser.add_argument("--dimension", nargs="+", required=True)
    parser.add_argument("--mode", choices=["custom_input", "vbench_standard", "vbench_category"], default="custom_input")
    parser.add_argument("--prompt_file")
    parser.add_argument("--load_ckpt_from_local", action="store_true")
    parser.add_argument("--read_frame", action="store_true")
    parser.add_argument("--imaging_quality_preprocessing_mode", default="longer")
    main(parser.parse_args())

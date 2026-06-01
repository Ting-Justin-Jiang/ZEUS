import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vbench_prompts(full_info_json: Path, limit: int) -> list[str]:
    with full_info_json.open("r", encoding="utf-8") as f:
        full_info = json.load(f)

    prompts = []
    seen = set()
    preferred_dims = {
        "subject_consistency",
        "motion_smoothness",
        "temporal_flickering",
        "imaging_quality",
        "aesthetic_quality",
        "dynamic_degree",
    }
    for row in full_info:
        if not preferred_dims.intersection(row["dimension"]):
            continue
        prompt = row["prompt_en"]
        if prompt in seen:
            continue
        prompts.append(prompt)
        seen.add(prompt)
        if len(prompts) >= limit:
            break
    return prompts


def load_prompt_jsonl(prompt_jsonl: Path, limit: int) -> list[str]:
    prompts = []
    with prompt_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            if len(prompts) >= limit:
                break
    return prompts


def make_pipeline(args):
    if args.model_family == "wan":
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(args.model, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe

    if args.model_family == "cogvideo":
        from diffusers import CogVideoXPipeline
        from diffusers.schedulers import DPMSolverMultistepScheduler

        pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe

    raise ValueError(f"Unsupported model family: {args.model_family}")


def apply_zeus(pipe, args) -> None:
    if args.method == "original":
        return
    from zeus import patch

    patch.apply_patch(
        pipe,
        acc_range=(args.acc_start, args.acc_end),
        interp_mode="psi",
        denominator=args.denominator,
        modular=tuple(args.modular),
        lagrange_int=args.lagrange_int,
        lagrange_step=args.lagrange_step,
        lagrange_term=args.lagrange_term,
        max_interval=args.max_interval,
    )


def reset_cache(pipe, method: str) -> None:
    if method == "zeus":
        from zeus import patch

        patch.reset_cache(pipe)


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_random_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_jsonl:
        prompts = load_prompt_jsonl(Path(args.prompt_jsonl), args.num_samples)
    else:
        prompts = load_vbench_prompts(Path(args.vbench_full_info), args.num_samples)
    selected = [(idx, prompt) for idx, prompt in enumerate(prompts)]
    if args.start_index is not None:
        selected = [(idx, prompt) for idx, prompt in selected if idx >= args.start_index]
    if args.end_index is not None:
        selected = [(idx, prompt) for idx, prompt in selected if idx < args.end_index]
    if not selected:
        raise ValueError("No prompts selected for the requested index range")
    prompt_map = {}
    prompt_rows = []

    pipe = make_pipeline(args)
    apply_zeus(pipe, args)

    if not args.skip_warmup:
        warmup_prompt = selected[0][1]
        _ = pipe(
            prompt=warmup_prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
        ).frames[0]
        reset_cache(pipe, args.method)
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_time = 0.0
    generated = 0
    batch_times = []

    for idx, prompt in tqdm(selected, desc=f"{args.method}:{args.model_family}"):
        video_path = out_dir / f"{idx:06d}.mp4"
        prompt_map[str(video_path.resolve())] = prompt
        prompt_rows.append({"id": idx, "prompt": prompt, "seed": args.seed + idx, "video": str(video_path)})
        if args.resume and video_path.exists():
            continue

        set_random_seed(args.seed + idx)
        t0 = time.perf_counter()
        output = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
        ).frames[0]
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        batch_times.append({"id": idx, "seconds": elapsed})

        export_to_video(output, str(video_path), fps=args.fps)
        generated += 1
        reset_cache(pipe, args.method)

    (out_dir / "prompts_for_vbench.json").write_text(json.dumps(prompt_map, indent=2), encoding="utf-8")
    (out_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prompt_rows) + "\n",
        encoding="utf-8",
    )

    summary = {
        "method": args.method,
        "model_family": args.model_family,
        "model": args.model,
        "num_samples": len(prompts),
        "selected_samples": len(selected),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "generated_this_run": generated,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "total_generation_seconds": total_time,
        "seconds_per_video": total_time / max(generated, 1),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "acc_range": [args.acc_start, args.acc_end],
        "denominator": args.denominator,
        "modular": args.modular,
        "lagrange": {
            "term": args.lagrange_term,
            "step": args.lagrange_step,
            "int": args.lagrange_int,
        },
        "max_interval": args.max_interval,
        "batch_times": batch_times,
    }
    (out_dir / "run_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["original", "zeus"], required=True)
    parser.add_argument("--model-family", choices=["wan", "cogvideo"], default="wan")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--vbench-full-info", default="/home/zs89/AeroWorld/.venv_vbench/lib/python3.12/site-packages/vbench/VBench_full_info.json")
    parser.add_argument("--prompt-jsonl")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="bad quality, static")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)

    parser.add_argument("--acc-start", type=int, default=8)
    parser.add_argument("--acc-end", type=int, default=47)
    parser.add_argument("--denominator", type=int, default=3)
    parser.add_argument("--modular", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lagrange-term", type=int, default=4)
    parser.add_argument("--lagrange-step", type=int, default=24)
    parser.add_argument("--lagrange-int", type=int, default=4)
    parser.add_argument("--max-interval", type=int, default=6)
    main(parser.parse_args())

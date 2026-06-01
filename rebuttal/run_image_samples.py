import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SADA_ROOT = REPO_ROOT / "sada-icml"
sys.path.insert(0, str(REPO_ROOT))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(dataset_name: str, limit: int) -> list[str]:
    if dataset_name == "coco2017":
        dataset = load_dataset("phiyodr/coco2017", split="validation")
        prompts = [sample["captions"][0] for sample in dataset]
    elif dataset_name == "parti":
        dataset = load_dataset("nateraw/parti-prompts", split="train")
        prompts = [sample["Prompt"] for sample in dataset]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return prompts[:limit]


def make_pipeline(args):
    if args.model_family == "flux":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        return pipe

    if args.model_family == "sdxl":
        from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
        ).to("cuda")
        if args.solver == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif args.solver == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe

    raise ValueError(f"Unsupported model family: {args.model_family}")


def apply_method_patch(pipe, args) -> None:
    if args.method == "original":
        return

    if args.method == "zeus":
        from zeus import patch

        patch.apply_patch(
            pipe,
            acc_range=(args.acc_start, args.acc_end),
            interp_mode=args.interp_mode,
            caching_mode=args.caching_mode,
            denominator=args.denominator,
            modular=tuple(args.modular),
            lagrange_int=args.lagrange_int,
            lagrange_step=args.lagrange_step,
            lagrange_term=args.lagrange_term,
            max_interval=args.max_interval,
        )
        return

    if args.method == "sada":
        sys.path.insert(0, str(SADA_ROOT))
        from sada import patch

        patch_kwargs = dict(
            sx=args.sada_sx,
            sy=args.sada_sy,
            acc_range=(args.acc_start, args.acc_end),
            lagrange_int=args.lagrange_int,
            lagrange_step=args.lagrange_step,
            lagrange_term=args.lagrange_term,
            max_fix=args.sada_max_fix,
            max_interval=args.max_interval,
        )
        if args.model_family == "flux":
            patch_kwargs.update(max_downsample=0, latent_size=(args.height // 16, args.width // 16))
        elif args.model_family == "sdxl":
            patch_kwargs.update(max_downsample=0)
        patch.apply_patch(pipe, **patch_kwargs)
        return

    raise ValueError(f"Unsupported method: {args.method}")


def reset_method_cache(pipe, method: str) -> None:
    if method == "zeus":
        from zeus import patch

        patch.reset_cache(pipe)
    elif method == "sada":
        sys.path.insert(0, str(SADA_ROOT))
        from sada import patch

        patch.reset_cache(pipe)


def run_batch(pipe, prompts: list[str], seeds: list[int], args):
    generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
    common = dict(
        prompt=prompts,
        num_inference_steps=args.steps,
        output_type="np",
        return_dict=True,
        generator=generators,
    )

    if args.model_family == "flux":
        common.update(
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            max_sequence_length=512,
        )
    elif args.model_family == "sdxl":
        common.update(
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
        )

    return pipe(**common).images


def get_skip_stats(pipe, method: str, steps: int) -> dict:
    if method == "original":
        return {
            "skip_count": 0,
            "actual_nfe": steps,
            "observed_max_consecutive_skips": 0,
            "skipping_path": [],
        }

    diffusion_model = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
    bus = getattr(diffusion_model, "_cache_bus", None)
    skipping_path = list(getattr(bus, "skipping_path", [])) if bus is not None else []
    unique_steps = sorted(set(skipping_path))
    max_run = 0
    current_run = 0
    previous_step = None
    for step in unique_steps:
        if previous_step is None or step == previous_step + 1:
            current_run += 1
        else:
            current_run = 1
        max_run = max(max_run, current_run)
        previous_step = step

    skip_count = len(skipping_path)
    return {
        "skip_count": skip_count,
        "actual_nfe": steps - skip_count,
        "observed_max_consecutive_skips": max_run,
        "skipping_path": skipping_path,
    }


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_random_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "run_metrics.json"
    prompts_path = out_dir / "prompts.jsonl"

    prompts = load_prompts(args.dataset, args.num_samples)
    with prompts_path.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            f.write(json.dumps({"id": idx, "prompt": prompt, "seed": args.seed + idx}) + "\n")

    pipe = make_pipeline(args)
    apply_method_patch(pipe, args)

    # Warmup is only for stable timing; it is not saved.
    warmup_prompt = prompts[0]
    _ = run_batch(pipe, [warmup_prompt], [args.seed], args)
    reset_method_cache(pipe, args.method)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_time = 0.0
    generated = 0
    batch_times = []
    nfe_records = []

    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"{args.method}:{args.model_family}"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_seeds = [args.seed + idx for idx in range(start, end)]

        pending = []
        pending_ids = []
        pending_seeds = []
        for local_idx, prompt in enumerate(batch_prompts, start=start):
            image_path = out_dir / f"{local_idx:06d}.jpg"
            if args.resume and image_path.exists():
                continue
            pending.append(prompt)
            pending_ids.append(local_idx)
            pending_seeds.append(args.seed + local_idx)

        if not pending:
            continue

        t0 = time.perf_counter()
        images = run_batch(pipe, pending, pending_seeds, args)
        elapsed = time.perf_counter() - t0
        skip_stats = get_skip_stats(pipe, args.method, args.steps)
        total_time += elapsed
        batch_record = {"start_id": pending_ids[0], "count": len(pending), "seconds": elapsed, **skip_stats}
        batch_times.append(batch_record)
        for image_id in pending_ids:
            nfe_records.append({"id": image_id, **skip_stats})

        for image_id, image in zip(pending_ids, images):
            pil_image = to_pil_image((image * 255).astype(np.uint8))
            pil_image.save(out_dir / f"{image_id:06d}.jpg", quality=95)
            generated += 1

        reset_method_cache(pipe, args.method)

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    summary = {
        "method": args.method,
        "model_family": args.model_family,
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": len(prompts),
        "generated_this_run": generated,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "total_generation_seconds": total_time,
        "seconds_per_image": total_time / max(generated, 1),
        "peak_memory_gb": peak_memory_gb,
        "mean_actual_nfe": (
            sum(record["actual_nfe"] for record in nfe_records) / len(nfe_records)
            if nfe_records
            else args.steps
        ),
        "mean_skip_count": (
            sum(record["skip_count"] for record in nfe_records) / len(nfe_records)
            if nfe_records
            else 0
        ),
        "mean_observed_max_consecutive_skips": (
            sum(record["observed_max_consecutive_skips"] for record in nfe_records) / len(nfe_records)
            if nfe_records
            else 0
        ),
        "max_observed_max_consecutive_skips": (
            max(record["observed_max_consecutive_skips"] for record in nfe_records)
            if nfe_records
            else 0
        ),
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
        "nfe_records": nfe_records,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["original", "zeus", "sada"], required=True)
    parser.add_argument("--model-family", choices=["flux", "sdxl"], default="flux")
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--solver", choices=["euler", "dpm"], default="dpm")
    parser.add_argument("--dataset", choices=["coco2017", "parti"], default="coco2017")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--acc-start", type=int, default=10)
    parser.add_argument("--acc-end", type=int, default=45)
    parser.add_argument("--denominator", type=int, default=3)
    parser.add_argument("--modular", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--interp-mode", choices=["psi", "x_0"], default="psi")
    parser.add_argument("--caching-mode", choices=["reuse_all", "interp_all", "reuse_interp"], default="reuse_interp")
    parser.add_argument("--lagrange-term", type=int, default=3)
    parser.add_argument("--lagrange-step", type=int, default=24)
    parser.add_argument("--lagrange-int", type=int, default=6)
    parser.add_argument("--max-interval", type=int, default=6)

    parser.add_argument("--sada-sx", type=int, default=3)
    parser.add_argument("--sada-sy", type=int, default=3)
    parser.add_argument("--sada-max-fix", type=int, default=0)
    main(parser.parse_args())

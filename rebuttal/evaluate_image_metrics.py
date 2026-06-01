import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_prompts(path: Path) -> dict[str, str]:
    prompts = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts[f"{row['id']:06d}"] = row["prompt"]
    return prompts


def load_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.jpg"))


@torch.inference_mode()
def compute_clip_score(image_dir: Path, prompts_path: Path, batch_size: int, device: str) -> float:
    prompts = load_prompts(prompts_path)
    image_paths = load_images(image_dir)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="CLIPScore"):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        batch_prompts = [prompts[path.stem] for path in batch_paths]
        inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        image_features = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_features = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores.append((image_features * text_features).sum(dim=-1).detach().cpu())

    return torch.cat(scores).mean().item()


@torch.inference_mode()
def compute_image_reward(image_dir: Path, prompts_path: Path, device: str) -> float:
    import ImageReward as RM

    prompts = load_prompts(prompts_path)
    image_paths = load_images(image_dir)
    model = RM.load("ImageReward-v1.0", device=device)

    scores = []
    for path in tqdm(image_paths, desc="ImageReward"):
        scores.append(float(model.score(prompts[path.stem], str(path))))
    return sum(scores) / max(len(scores), 1)


def main(args):
    image_dir = Path(args.image_dir)
    prompts_path = Path(args.prompts_jsonl)
    output_path = Path(args.output_json)

    result = {
        "image_dir": str(image_dir),
        "prompts_jsonl": str(prompts_path),
        "num_images": len(load_images(image_dir)),
    }

    if args.clip:
        result["clip_score"] = compute_clip_score(image_dir, prompts_path, args.batch_size, args.device)

    if args.image_reward:
        try:
            result["image_reward"] = compute_image_reward(image_dir, prompts_path, args.device)
        except ModuleNotFoundError as exc:
            result["image_reward_error"] = f"Missing dependency: {exc}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--image-reward", action="store_true")
    main(parser.parse_args())

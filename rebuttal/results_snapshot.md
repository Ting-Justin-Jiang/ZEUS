# Rebuttal Results Snapshot

## FLUX COCO-200: CLIPScore / ImageReward

Run root: `rebuttal/runs/flux_coco200`

| Method | Images | CLIPScore ↑ | ImageReward ↑ | Notes |
|---|---:|---:|---:|---|
| Full | 200 | 0.3116 | 1.0991 | Quality reference |
| ZEUS | 200 | 0.3112 | 1.0930 | Nearly identical CLIP, small ImageReward drop vs Full |
| SADA | 200 | 0.3113 | 1.0808 | Similar CLIP, lower ImageReward than ZEUS |

Speed numbers in `summary.json` are useful for rough orientation, but Full was resumed across multiple runs while tuning batch size. Use a clean timing pass before quoting exact latency in the rebuttal.

## FLUX Same-GPU Timing

Run root: `rebuttal/runs/flux_timing20_gpu2`

Setting: single GPU2, sequential runs, 20 COCO prompts, FLUX.1-dev, 50 steps, batch size 1.

| Method | Seconds / image | Relative speedup | Peak memory |
|---|---:|---:|---:|
| Full | 25.84 | 1.00x | 33.85 GB |
| ZEUS | 10.51 | 2.46x | 33.85 GB |
| SADA | 10.55 | 2.45x | 33.85 GB |

This is the cleaner latency table to quote, because all methods were run on the same GPU sequentially. The 40-image timing run in `rebuttal/runs/flux_timing40` is a useful sanity check but was run concurrently across different GPUs.

## SDXL Matched-Budget SADA Comparison

Quality run root: `rebuttal/runs/sdxl_coco200_matched`

Timing run root: `rebuttal/runs/sdxl_timing50_gpu2`

Setting: SDXL base 1.0, COCO2017 prompts, `1024x1024`, `50` steps, DPM solver, guidance scale `5.0`, batch size 1. Quality metrics use 200 prompts. Timing is a separate single-GPU sequential pass on GPU2 with 50 prompts, so use the timing numbers for latency.

| Method | Quality images | Actual NFE | sec/img ↓ | Speedup ↑ | Peak memory | CLIPScore ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 200 | 50.0 | 3.366 | 1.00x | 11.50 GB | 0.3170 | 0.7550 |
| SADA | 200 | 25.2 | 1.556 | 2.16x | 11.50 GB | 0.3171 | 0.7417 |
| ZEUS | 200 | 24.0 | 1.464 | 2.30x | 11.50 GB | 0.3166 | 0.7339 |

Takeaway: ZEUS and SADA are matched closely in actual NFE and same-GPU latency. SADA has slightly higher ImageReward on this SDXL run, while ZEUS is about 6% faster with nearly identical CLIPScore. This should be written as a deployment tradeoff, not as pointwise quality dominance.

## SDXL Predictor Ablation

Run root: `rebuttal/runs/sdxl_ablation50`

Setting: SDXL base 1.0, 50 COCO prompts, `1024x1024`, `50` steps, DPM solver, guidance scale `5.0`.

| Predictor setting | Actual NFE | sec/img ↓ | CLIPScore ↑ | ImageReward ↑ | Note |
|---|---:|---:|---:|---:|---|
| ZEUS no Lagrange tail | 27.0 | 2.058 | 0.3139 | 0.8082 | More conservative; slower but better preference score |
| ZEUS default Lagrange-3 | 24.0 | 1.871 | 0.3132 | 0.7414 | Default speed-quality point |
| ZEUS Lagrange-4 | 24.0 | 1.864 | 0.3122 | 0.7289 | Richer tail predictor does not improve quality |

Takeaway: a richer Lagrange tail predictor does not improve quality under the same construction. The more conservative no-Lagrange variant improves ImageReward but uses more NFE and latency, which supports presenting ZEUS as a controllable speed-quality tradeoff.

## SDXL Window Sensitivity

Run root: `rebuttal/runs/sdxl_window50`

Setting: SDXL base 1.0, 50 COCO prompts, `1024x1024`, `50` steps, DPM solver, guidance scale `5.0`.

| Window | Actual NFE | sec/img ↓ | CLIPScore ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|
| 10/80/10 | 21.0 | 1.664 | 0.3141 | 0.7746 |
| 15/70/15 | 25.0 | 1.924 | 0.3125 | 0.7414 |
| 20/70/10 | 24.0 | 1.868 | 0.3132 | 0.7414 |
| 25/60/15 | 28.0 | 2.143 | 0.3138 | 0.8148 |

Takeaway: CLIPScore is stable across nearby windows. ImageReward shifts with the expected speed-quality tradeoff: the more conservative 25/60/15 setting is slower and improves ImageReward, while more aggressive settings are faster. This supports describing 20/70/10 as a deterministic default rather than a fragile hidden tuning choice.

## SDXL Acceleration Stress Test

Run root: `rebuttal/runs/sdxl_stress50`

Setting: SDXL base 1.0, 50 COCO prompts, `1024x1024`, `50` steps, DPM solver, guidance scale `5.0`.

| Setting | Actual NFE | sec/img ↓ | CLIPScore ↑ | ImageReward ↑ | Note |
|---|---:|---:|---:|---:|---|
| r=3 | 24.0 | 1.890 | 0.3132 | 0.7414 | Default |
| r=4 | 23.0 | 1.806 | 0.3120 | 0.7188 | More aggressive |
| r=5 | 21.0 | 1.672 | 0.3136 | 0.7386 | Stress setting |
| r=6 | 22.0 | 1.750 | 0.3129 | 0.7101 | Boundary setting |

Takeaway: CLIPScore remains stable even under more aggressive skipping, but ImageReward drops at the boundary. This is useful for stating a practical acceleration regime instead of implying unlimited skip aggressiveness.

## SDXL Consecutive-Skip Boundary Stress

Run root: `rebuttal/runs/sdxl_max_interval200`

Setting: SDXL base 1.0, 200 COCO prompts, `1024x1024`, `50` steps, DPM solver, guidance scale `5.0`, batch size 1. All generation and evaluation were run sequentially on the same GPU2. This experiment targets the reviewer concern about the maximum number of consecutive skipped/cached steps, so the table reports the observed maximum consecutive skips per sample.

The default-schedule rows keep the ZEUS schedule fixed: `acc_range=(10,45)`, `denominator=3`, `modular=[0,1]`, `lagrange_term=3`, `lagrange_step=24`, and `lagrange_int=6`, while sweeping only `max_interval`. The extended-anchor rows keep the same settings except for `lagrange_int`, allowing longer observed consecutive skip runs.

| Schedule | max_interval | lagrange_int | Observed max consecutive skips | Actual NFE | sec/img ↓ | CLIPScore ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 2 | 6 | 2 | 27.0 | 1.625 | 0.3169 | 0.7450 |
| Default | 4 | 6 | 4 | 27.0 | 1.636 | 0.3171 | 0.7631 |
| Default | 6 | 6 | 5 | 24.0 | 1.463 | 0.3166 | 0.7339 |
| Default | 8 | 6 | 5 | 24.0 | 1.469 | 0.3166 | 0.7339 |
| Extended anchor | 8 | 8 | 7 | 23.0 | 1.432 | 0.3167 | 0.7366 |
| Extended anchor | 12 | 12 | 11 | 22.0 | 1.346 | 0.3133 | 0.5599 |

Takeaway: under the default ZEUS anchor schedule, increasing `max_interval` beyond 6 does not create longer consecutive skips; the observed run length saturates at 5 and quality is unchanged. When we explicitly extend the anchor interval, 7 consecutive skips remains close to default quality, while 11 consecutive skips causes a large ImageReward drop. This gives a concrete failure boundary and supports stating that ZEUS targets the practical moderate-acceleration regime rather than arbitrary long cache intervals.

## SDXL Sparse-Attention Complementarity Pilot

Run root: `rebuttal/runs/sdxl_sparse_complementarity200`

Setting: SDXL base 1.0, 200 COCO prompts, `1024x1024`, `50` steps, DPM solver, guidance scale `5.0`, batch size 1. Generation and evaluation were run sequentially on GPU2. The sparse variant is a naive top-k cross-attention sparsity pilot, not an optimized sparse-attention implementation and not a speed baseline. It is intended only to test whether ZEUS approximation error catastrophically compounds with attention sparsity.

Baseline Full/ZEUS rows are from `rebuttal/runs/sdxl_coco200_matched`; sparse rows are from this pilot.

| Method | Cross-attn sparsity | Actual NFE | sec/img | CLIPScore ↑ | ImageReward ↑ |
|---|---:|---:|---:|---:|---:|
| Full | 0% | 50.0 | 3.366 | 0.3170 | 0.7550 |
| ZEUS | 0% | 24.0 | 1.464 | 0.3166 | 0.7339 |
| Sparse only | 20% | 50.0 | 4.607 | 0.3176 | 0.7411 |
| ZEUS + sparse | 20% | 24.0 | 2.323 | 0.3170 | 0.7114 |
| Sparse only | 50% | 50.0 | 4.563 | 0.3173 | 0.7717 |
| ZEUS + sparse | 50% | 24.0 | 2.291 | 0.3171 | 0.7591 |

Takeaway: CLIPScore remains stable for sparse-only and ZEUS+sparse variants, suggesting no prompt-alignment collapse. ImageReward changes are modest except for the naive 20% sparse composition case, and the 50% sparse composition remains close to sparse-only/full quality. This supports a conservative statement that a lightweight sparse-attention pilot does not show catastrophic error compounding, while optimized joint sparse-attention speedups remain future work.

## Wan2.1 Balanced VBench-24

Run root: `rebuttal/runs/wan_vbench24_balanced`

Setting: Wan2.1-T2V-14B, `480x832`, `81` frames, `50` steps, VBench `0.1.5`, custom balanced subset with 24 prompts: 8 from `subject_consistency`, 8 from `motion_smoothness`, and 8 from `temporal_flickering`. Metrics were evaluated with VBench dimensions `subject_consistency`, `motion_smoothness`, `temporal_flickering`, and `imaging_quality`.

Videos were generated by parallelizing different prompt ranges across GPUs to finish the rebuttal run; each individual video was still generated on a single GPU. Use this table for VBench quality, not latency.

| Method | Videos | Subject consistency ↑ | Motion smoothness ↑ | Temporal flickering ↑ | Imaging quality ↑ |
|---|---:|---:|---:|---:|---:|
| Full | 24 | 0.9619 | 0.9810 | 0.9737 | 0.6118 |
| ZEUS | 24 | 0.9628 | 0.9820 | 0.9755 | 0.6008 |
| Delta | - | +0.0010 | +0.0010 | +0.0018 | -0.0110 |

Result files:

- Full: `rebuttal/runs/wan_vbench24_balanced/original_vbench/wan_original_24_eval_results.json`
- ZEUS: `rebuttal/runs/wan_vbench24_balanced/zeus_vbench/wan_zeus_24_eval_results.json`
- Prompt subset: `rebuttal/runs/vbench24_balanced/prompts.jsonl`

Takeaway: on the balanced VBench subset, ZEUS preserves or slightly improves the three temporal metrics that directly address the reviewer concern, with a small imaging-quality drop of about 0.011 absolute.

## Wan2.1 VBench-8 Pilot

Run root: `rebuttal/runs/wan_vbench8`

Setting: Wan2.1-T2V-14B, `480x832`, `81` frames, `50` steps, first 8 selected VBench prompts. Full videos were generated by parallelizing different prompts across GPUs to finish the pilot; each individual video was still generated on a single GPU. Use the VBench metrics for quality comparison. For latency, use the single-video probe below.

| Method | Subject consistency ↑ | Motion smoothness ↑ | Temporal flickering ↑ | Imaging quality ↑ |
|---|---:|---:|---:|---:|
| Full | 0.9820 | 0.9965 | 0.9966 | 0.6067 |
| ZEUS | 0.9832 | 0.9967 | 0.9969 | 0.5867 |

Takeaway: ZEUS preserves or slightly improves the three temporal metrics in this pilot, while imaging quality drops by about 0.020 absolute.

## Wan2.1 Single-Video Latency Probe

Actual setting: `480x832`, `81` frames, `50` steps, first VBench prompt.

Run root: `rebuttal/runs/wan_vbench32`

| Method | Seconds / video | Peak memory | Subject consistency ↑ | Motion smoothness ↑ | Temporal flickering ↑ | Imaging quality ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full | 619.5 | 46.64 GB | 0.9965 | 0.9971 | 0.9981 | 0.6130 |
| ZEUS | 293.4 | 46.65 GB | 0.9973 | 0.9971 | 0.9983 | 0.6031 |

This is only a one-video probe. It validates the Wan2.1 generation pipeline and VBench evaluator, but should not be quoted as the final VBench result. Estimated runtime for 32 videos:

| Method | Estimated 32-video generation time |
|---|---:|
| Full | ~5.5 hours |
| ZEUS | ~2.6 hours |

Use the balanced VBench-24 table above as the main video-metric result for the rebuttal.

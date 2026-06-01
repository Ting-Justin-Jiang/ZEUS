# Rebuttal Argument Notes

## Whether VBench Needs TeaCache / Other Baselines

不建议把 TeaCache 作为 VBench 的必跑 baseline，除非已有同设置视频输出或 reviewer 明确要求视频 TeaCache 对比。VBench 的主要 rebuttal 目的不是重新证明 ZEUS beat every baseline，而是回应：

> Does ZEUS harm temporal coherence, motion smoothness, or subject consistency after changing the ODE trajectory?

因此最小强实验是：

1. Full sampling vs ZEUS on Wan2.1 with the same prompts and seeds.
2. 如果已有论文中的视频 baseline 输出，再加一个 existing best baseline。
3. TeaCache / TaylorSeer 优先放在 FLUX text-to-image CLIPScore + ImageReward 表里，因为 reviewer 对 image alignment / preference 的质疑更直接。

## Response to Adaptive Methods / SADA Fairness

Reviewer concern:

> Empirical knock on third-order Lagrange-style predictors does not dismiss richer forecasts or adaptive skipping. Related adaptive budget methods require matching effective forward counts rather than nominal labels.

Recommended stance:

- 不要写“this concern is invalid”。
- 要写：ZEUS 和 SADA / AdaptiveDiffusion are different axes, and our main results already report end-to-end A5000 latency rather than nominal labels.
- 为了移除 ambiguity，再补 matched-NFE / matched-latency 表。

Suggested wording:

> We agree that adaptive-budget methods are a different axis from fixed higher-order extrapolation. Our claim is not that ZEUS subsumes adaptive schedulers; rather, ZEUS studies what output-level information is sufficient under evaluation scarcity and provides a fixed, architecture-agnostic recipe. Importantly, our main comparisons report end-to-end latency on A5000 rather than nominal acceleration labels. To remove any ambiguity, we will add matched-NFE and matched-latency comparisons against SADA, together with peak-memory and implementation-overhead statistics.

## ZEUS vs SADA Positioning

Core distinction:

| Axis | SADA | ZEUS |
|---|---|---|
| Main question | When is the ODE stable enough to skip/adapt? | What minimal output information should be reused under scarcity? |
| Mechanism | Adaptive stability-guided acceleration, feature/cache operations | Fixed output-level predictor + interleaved reuse |
| Deployment | More architecture-specific implementation knobs | Few-line output-level patch, no training/calibration |
| Fairness metric | Must compare by end-to-end latency and effective forward count | Same |

Use this to answer:

> SADA gets better quality on some SD/SDXL points.

Suggested wording:

> On SD-family models, SADA can achieve stronger single-metric quality at nearby speedups, which we will state clearly. The advantage of ZEUS is not that every point dominates SADA, but that it offers a simpler quality-speed-memory tradeoff with no adaptive policy, calibration, or architecture-specific cache design. The added matched-budget results clarify this deployment-oriented contribution.

## Complementarity with Token / Attention Sparsity

Reviewer concern:

> ZEUS is positioned as complementary to token/attention sparsity, but joint measurements are needed.

Recommended stance:

- “Architecturally complementary” is defensible.
- “Empirically complementary” needs joint experiment; if not run, soften wording.

Suggested wording:

> Our complementarity claim is architectural: ZEUS modifies only the sampling schedule and output-level denoiser reuse, whereas token/attention sparsification reduces per-call computation inside the denoiser. We will temper the wording to “potentially complementary” and add joint ZEUS + sparse/cache evaluation as future work. If space permits, we will include a lightweight pilot result.

## Training-Free Wording

Reviewer concern:

> Fixed schedule parameters weaken the training-free claim.

Suggested wording:

> We use “training-free” to mean no model retraining, no learned policy, and no per-model calibration. The 20/70/10 window and r preset are deterministic default recipes. We will add window-split sensitivity results to show the method is not fragile to this choice.


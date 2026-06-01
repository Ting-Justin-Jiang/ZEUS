# ZEUS Rebuttal Experiment Plan

目标：在 5000-character rebuttal 里用最少但最有杀伤力的新增实验，回应 reviewer 对实验完整性、SADA 公平性、schedule heuristic、理论假设边界的主要质疑。

## 总体策略

优先补 reviewer 明确点名的标准 benchmark 和公平对比。Rebuttal 不需要把所有实验细节展开，而是要提供几张高信息密度小表，证明：

1. ZEUS 不只提升 perceptual / reconstruction metric，也保持 prompt alignment、preference quality 和视频时序质量。
2. 相比 SADA，ZEUS 的优势主要是 end-to-end latency、memory、实现复杂度和跨架构通用性，而不是每个单点质量指标都压过 SADA。
3. 20/70/10 window 和 r preset 不是 fragile hand-tuning，方法在合理范围内稳定。
4. 理论假设是理想化的，但 empirical covariance / GLS 检查不改变二阶 predictor 的实践结论。

## 优先级计划表

| Priority | Reviewer concern | Experiment | Minimal setup | Metrics / outputs | Rebuttal claim |
|---|---|---|---|---|---|
| P0 | G6c9: 缺 FLUX text-to-image 标准 alignment / preference 指标 | FLUX / SDXL 上补 CLIPScore + ImageReward | 使用现有 prompt set；比较 Full、SADA、ZEUS、ZEUS-Fast、ZEUS-Turbo；优先 FLUX，时间够再 SDXL | CLIPScore、ImageReward、可选 Aesthetic / HPSv2；附 latency | ZEUS 的加速没有显著损害 prompt adherence 或 human preference，并在相近质量下给出更高真实速度 |
| P0 | G6c9: 视频只测 perceptual / frame-level，缺 temporal benchmark | Wan2.1 / CogVideoX 上跑 VBench | 优先选择论文中已有视频生成设置；比较 Full、baseline、ZEUS preset；如果算力紧张，先跑一个视频模型 | VBench overall、subject consistency、motion smoothness、temporal flickering、imaging quality | ZEUS 对 ODE trajectory 的修改没有破坏 temporal coherence，视频收益不只是单帧感知指标 |
| P0 | Rer9 / G6c9 / mH77: ZEUS 在 SD family 上不总是优于 SADA；公平性需要 matched budget | SADA vs ZEUS matched NFE / matched latency | 在 SDXL 或 SD-2.1 上做两种对齐：相同 NFE、相同 wall-clock latency；记录 memory | FID / CLIPScore / ImageReward、latency、NFE、peak memory、extra code / cache requirement | ZEUS 是 quality-speed-memory Pareto tradeoff；即使某些质量指标 SADA 更高，ZEUS 真实速度、内存和部署成本更优 |
| P1 | d3ux / mH77: 20/70/10 window 看起来 hand-tuned | Window split sensitivity | 选 FLUX 或 SDXL 单模型；测试 15/70/15、20/70/10、10/80/10、25/60/15 | FID 或 perceptual metric、CLIPScore、latency | ZEUS 对中间窗口比例不敏感，20/70/10 是固定 recipe 而不是针对 benchmark 调参 |
| P1 | G6c9: 没有 r>=5 stress test，缺 failure boundary | Extreme r stress test | 同一模型上测试 r=4,5,6,8；可只测 ZEUS | Quality metric、CLIP/ImageReward、failure examples、latency | ZEUS 的 practical range 是约 1.5x-3.2x；极端 skip 会自然退化，我们补充边界而不是声称无限可扩展 |
| P1 | nWwb / mH77: BLUE / i.i.d. noise 假设在 accelerated sampling 中可能不成立 | Residual covariance + empirical GLS check | 采样若干 trajectories；计算 adjacent residual covariance / correlation；比较 fixed weights (2,-1) 与 empirical GLS weights | Residual correlation heatmap 或小表；quality delta between ZEUS and GLS-ZEUS | Theorem 是 idealized lens；实测 covariance deviation 不会带来显著不同的 predictor choice，二阶 predictor 仍是 practical optimum |
| P2 | d3ux: 想看 UniPC / EVODiff 等 newer training-free methods | 补 UniPC，视接入成本决定 EVODiff | UniPC 更优先，因为是常用 solver baseline；EVODiff 若 setting 不一致可文字解释 | Quality / latency under comparable NFE | ZEUS 和 solver acceleration 是不同轴；可作为 solver-agnostic output extrapolation，与 UniPC/DPM-Solver++ 兼容 |
| P2 | mH77: ZEUS 是否能和 cache / sparse attention 叠加 | ZEUS + lightweight cache / sparse attention pilot | 只做一组小规模 sanity check，除非代码已现成 | Quality、latency、memory | 初步结果显示 errors 不明显 compound；完整 joint optimization 留作 future work |

## 建议实际执行顺序

1. 先跑 P0-1：CLIPScore + ImageReward。成本最低，直接补 reviewer 点名缺口。
2. 并行启动 P0-2：VBench。这个最可能耗时，但 rebuttal 价值最高。
3. 跑 P0-3：SADA matched NFE / latency / memory。重点是把 SADA 问题从“质量谁高”改写成“真实部署 tradeoff”。
4. 如果还有时间，跑 P1-1 和 P1-2。它们是小 ablation，适合放一张 compact table。
5. 理论 rebuttal 被 nWwb 明确抓住时，至少做 P1-3 的 residual correlation 小实验；如果结果一般，也可以诚实写成 limitation + empirical robustness。

## 5000-character Rebuttal 推荐结构

| Section | 字符预算 | 内容 |
|---|---:|---|
| Opening | 400-600 | 感谢 reviewer；一句话总结新增实验覆盖 semantic alignment、video temporal quality、fair SADA comparison、schedule sensitivity、theory limitation |
| Standard metrics | 900-1100 | 放 CLIPScore / ImageReward + VBench 的核心数值；直接回应 G6c9 |
| SADA comparison | 900-1100 | 用 matched NFE / matched latency / memory 表说明 ZEUS 的 Pareto 优势；回应 Rer9、G6c9、mH77 |
| Sensitivity / boundary | 700-900 | window split + r stress test；回应 hand-tuned schedule 和 failure boundary |
| Theory caveat | 600-800 | 承认 BLUE 假设是 idealized；报告 residual covariance / GLS-ZEUS 差异；补 limitation wording |
| Presentation fixes | 300-500 | 承诺加 pseudocode、修 LaTeX leakage、figure labels、notation、duplicated paragraph |

## 可直接填数的结果表模板

### Table A: Text-to-image semantic and preference quality

| Method | NFE | Latency | CLIPScore ↑ | ImageReward ↑ | FID / LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Full | TBD | TBD | TBD | TBD | TBD |
| SADA | TBD | TBD | TBD | TBD | TBD |
| ZEUS | TBD | TBD | TBD | TBD | TBD |
| ZEUS-Fast | TBD | TBD | TBD | TBD | TBD |
| ZEUS-Turbo | TBD | TBD | TBD | TBD | TBD |

Rebuttal sentence target:

> We added CLIPScore and ImageReward on FLUX/SDXL. ZEUS preserves semantic alignment and preference quality while delivering higher end-to-end speedup, showing that the trajectory modification does not merely optimize low-level perceptual metrics.

### Table B: Video temporal quality

| Method | Latency | VBench overall ↑ | Subject consistency ↑ | Motion smoothness ↑ | Temporal flickering ↑ |
|---|---:|---:|---:|---:|---:|
| Full | TBD | TBD | TBD | TBD | TBD |
| Baseline | TBD | TBD | TBD | TBD | TBD |
| ZEUS | TBD | TBD | TBD | TBD | TBD |

Rebuttal sentence target:

> The new VBench results confirm that ZEUS maintains temporal coherence and subject consistency, addressing the concern that altered ODE integration may harm video dynamics.

### Table C: Fair SADA comparison

| Method | Matching rule | NFE | Latency | Peak memory | Quality metric |
|---|---|---:|---:|---:|---:|
| SADA | matched NFE | TBD | TBD | TBD | TBD |
| ZEUS | matched NFE | TBD | TBD | TBD | TBD |
| SADA | matched latency | TBD | TBD | TBD | TBD |
| ZEUS | matched latency | TBD | TBD | TBD | TBD |

Rebuttal sentence target:

> Under matched NFE and matched latency, ZEUS offers a favorable quality-speed-memory tradeoff. Unlike SADA, it requires no adaptive policy, feature cache, or architecture-specific implementation, which is central to our deployment claim.

### Table D: Schedule sensitivity and stress boundary

| Setting | Quality metric | CLIP / ImageReward | Latency | Note |
|---|---:|---:|---:|---|
| 15/70/15 | TBD | TBD | TBD | window sensitivity |
| 20/70/10 | TBD | TBD | TBD | default |
| 10/80/10 | TBD | TBD | TBD | window sensitivity |
| 25/60/15 | TBD | TBD | TBD | window sensitivity |
| r=5 | TBD | TBD | TBD | stress |
| r=6 | TBD | TBD | TBD | stress |
| r=8 | TBD | TBD | TBD | stress |

Rebuttal sentence target:

> The default 20/70/10 window is not a tuned hidden hyperparameter: nearby splits yield similar quality-speed tradeoffs. Larger r values define the expected failure boundary, so we now clarify the practical acceleration regime.

## 哪些不用狠狠做实验

| Issue | Response type |
|---|---|
| LaTeX leakage / visible itemsep / duplicated paragraph | 直接承诺 camera-ready 修复 |
| Missing pseudocode | 加 Algorithm 1，rebuttal 里一句话说明 |
| Notation confusion between true / predicted values | 承诺统一 notation，尤其是 `psi_t` vs `hat{psi}_t` |
| Figure axes / blocked labels / alignment | 承诺重画 Fig. 1 / Fig. 2 / Fig. 4 |
| Training-free wording | 文字澄清：no training, no calibration, no learned policy；fixed schedule 是 deterministic recipe |
| Joint with sparse attention / cache | 如无现成代码，不必主打；写 complementary and future work |

## 最小可交付组合

如果时间只够三组实验，做：

1. VBench。
2. CLIPScore + ImageReward。
3. SADA matched NFE / matched latency / memory。

如果时间够五组，追加：

4. Window split sensitivity。
5. r>=5 stress test。

如果 nWwb 的理论质疑需要重点扭转，追加：

6. Residual covariance + empirical GLS check。

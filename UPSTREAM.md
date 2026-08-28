# Ling-3.0-flash MXFP4 on DGX Spark — W4A8 + Humming MoE + online FP8 LM head

这是 [sgl-project/sglang](https://github.com/sgl-project/sglang) 的一个分支快照，在 NVIDIA DGX Spark（GB10 / SM121）上部署 [Ling-3.0-flash-fp4](https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4)，并在上游基础上加了 3 处改动，使得 **Humming MXFP4 MoE + 在线 FP8 LM head + MTP(NEXTN) 投机解码** 三项优化可以同时生效。

在 GSM8K 全量 1319 题上，三项优化合计带来 **+42.5%** 的解码吞吐，而**准确率无统计显著变化**（McNemar 精确检验 p = 1.0）。

## 上游来源

| | |
|---|---|
| 仓库 | `https://github.com/sgl-project/sglang` |
| PR | [#33561](https://github.com/sgl-project/sglang/pull/33561) — Ling-3.0-flash (BailingMoeV3) 支持 |
| 分支 | `ling3-flash-dspark` |
| commit | `814d2d1680e7c431277656f1a48d1d9680a17fd3` |

本仓库为该 commit 的完整快照 + 下述改动，以单个 root commit 形式提交（不含上游历史）。
上游代码遵循 Apache License 2.0，`LICENSE` 文件随树保留。

改动的完整 diff 见 [`patches/ling3-w4a8-humming.patch`](patches/ling3-w4a8-humming.patch)（3 个文件，26 行新增，2 行修改）。

## 改动内容

### 1. 修复上游启动崩溃 — `bailing_moe_v3.py`

```diff
-                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
+                    use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
```

上游在这一处漏改了并行状态重构（全树 49 处都用带 `.config` 的写法，只有这一处没有）。
不修的话模型初始化直接抛异常，服务起不来：

```
AttributeError: 'enable_dp_lm_head' is a parallel config leaf, not live topology;
read it as get_parallel().config.enable_dp_lm_head
```

这是一个独立于其余改动的上游 bug 修复，可单独 cherry-pick。

### 2. lm_head 在线 FP8 量化 — `environ.py` + `bailing_moe_v3.py`

新增环境变量 `SGLANG_ENABLE_FP8_LM_HEAD=1`。checkpoint 中 lm_head 存的是 BF16，
开启后仅对这一层在线量化为动态 FP8，模型其余部分仍按 checkpoint 自带的
FP8/MXFP4 配置加载。

实测 decode 吞吐约 **+11%**。

### 3. MTP draft 共享 target 的 lm_head 模块 — `bailing_moe_nextn.py`

新增 `set_lm_head_from_target()`。**这是让上面两项能与 MTP 共存的关键。**

框架侧 `eagle_worker_v2.py` 用 `hasattr` 判断是否调用该方法：

```python
and hasattr(self.draft_runner.model, "set_lm_head_from_target")
    self.draft_runner.model.set_lm_head_from_target(target_lm_head)
```

`BailingMoeForCausalLMNextN` 原本只有 `set_embed_and_head()`，它把 target 的
**权重张量**赋给 draft。一旦 lm_head 被在线量化为 FP8，draft 拿到的是 FP8 张量，
却仍按自己的 BF16 `quant_method` 解释 —— draft 输出全错，投机接受率归零，
而 MTP 每步照样白跑 3 次 draft 前向，**吞吐反而比不开任何优化还低**。

补上该方法后，draft 共享 target 的**整个 lm_head 模块**（连带正确的 `quant_method`）。
实现与同仓库的 `qwen3_5_mtp.py` 一致。

| 配置 | MTP accept len | decode |
|---|---|---|
| 三项全开（缺此修复） | 1.00 | 24.0 tok/s |
| 三项全开（含此修复） | **2.68** | **53.9 tok/s** |

## 实测结果

**硬件**：NVIDIA DGX Spark（GB10，SM121，48 SM，121 GB 统一内存 LPDDR5X，273 GB/s）
**模型**：Ling-3.0-flash-fp4（124B 总参 / 5.1B 激活，42 层 = 35 KDA 线性注意力 + 7 MLA，512 专家选 8）

**评测协议**：GSM8K 全量 **1319** 题，5-shot（示范取自 train 集，评测用 test 集全量，无污染），
`temperature=0.6, top_p=0.95, top_k=29, max_tokens=32768`，`enable_thinking=True`

两次评测除三项优化外**所有变量完全对齐**：相同的采样参数、reasoning/tool parser、
5-shot 协议、random seed、`mem-fraction-static`。

| 指标 | 基线 | 优化版 | 差异 |
|---|---|---|---|
| MoE backend | `flashinfer_mxfp4` | `humming` | |
| LM head | FP32 (`--enable-fp32-lm-head`) | 在线 FP8 | |
| 投机解码 | 无 | MTP(NEXTN) 3 步 | |
| **GSM8K 准确率** | **96.89%** (1278/1319) | **96.82%** (1277/1319) | −0.08 pt（不显著） |
| **decode 吞吐** | **37.81** tok/s | **53.89** tok/s | **+42.5%** |
| MTP accept len | — | **2.68** | |
| 平均延迟 | 15.21 s | 11.49 s | −24.5% |
| 延迟 P50 / P95 | 8.36 / 50.20 s | 6.11 / 38.60 s | −26.9% / −23.1% |
| TTFT 均值 | 0.207 s | 0.214 s | |
| 输出长度 P50 / P95 | 309 / 1889 tokens | 318 / 2060 tokens | |
| 请求错误 / 截断 | 0 / 0 | 0 / 0 | |

### 精度：McNemar 配对检验

对同样 1319 道题逐题配对：

| | 优化版对 | 优化版错 |
|---|---|---|
| **基线对** | 1264 | 14 |
| **基线错** | 13 | 28 |

不一致对数 27（基线独对 14，优化独对 13），精确二项检验双侧 **p = 1.0000**。

**结论：三项优化没有带来可检出的精度损失。** 那 0.08 个百分点的差距等价于 1 道题，
且两个方向的翻转数量几乎相同（14 vs 13），完全符合 `temperature=0.6` 随机采样下的抖动。

### 一个容易被误报的点：MTP 收益强烈依赖工作负载

同一套代码，只换采样方式和输出长度，MTP 的收益差别很大：

| 工作负载 | 采样 | 平均输出长度 | 提升 |
|---|---|---|---|
| 短输出（`max_tokens=512`） | greedy | 322 tokens | **+74.7%** |
| 长思维链（`max_tokens=32768`, `enable_thinking`） | `temp=0.6/top_p=0.95/top_k=29` | 607 tokens | **+42.5%** |

两者接受率几乎相同（2.71 vs 2.68），差异主要来自**随机采样让每次 draft/verify 都更贵**：
MTP 每产出 1 个 token 要做约 2.6 次采样（3 次 draft + 4 个 verify 位置 ÷ 2.68 accept），
而无投机时只做 1 次。greedy 的 `argmax` 远比 `top_k=29` + `top_p` + 拒绝采样便宜，
这个成本被放大了 2.6 倍。

同代码对照实测：greedy 55.0 tok/s → 推荐采样 46.4 tok/s（**−15.6%**），
剥离接受率影响后采样本身仍占 **−9.3%**。

**对外报数建议用 +42.5%** —— Ling-3.0-flash 是推理模型，思维链是其正常用法，
`max_tokens=512` 会直接截断思考过程。

## 使用

部署步骤见 [`dgx-spark-sglang-ling3-w4a8-humming.ipynb`](dgx-spark-sglang-ling3-w4a8-humming.ipynb)，
启动脚本见 [`scripts/dgx-spark/`](scripts/dgx-spark/)。

关键点：

- **必须先预编译 FlashInfer CUTLASS MoE kernel**。首次启动会在加载权重的同时 JIT 编译，
  GB10 的统一内存下极易 OOM（实测触发过两次 `cudafe++ invoked oom-killer` 硬重启）。
  正确做法是先停掉所有服务，独占内存用 `ninja -j4` 编译，约 20 分钟；`-j1` 实测
  7 分钟才编完 1/97。
- FlashInfer 的缓存路径可能是 `~/.cache/flashinfer/<版本>/121a/cached_ops/fused_moe_120/`
  或 `~/.cache/sglang/.cache/flashinfer/<版本>/121a/cached_ops/fused_moe_120/`
  （随 SGLang 版本而变），且**与 FlashInfer 版本绑定**——升级后旧缓存全部失效，需重编。
- Humming kernel 由 NVRTC 按实际 shape 编译，缓存在 `~/.humming/cache`，
  单个 kernel 秒级完成，不需要单独预编译。但 GB10 用统一 LPDDR5X 内存，NVML 读不到
  显存时钟，而 Humming 依赖内存带宽做 roofline 选型，需要给 `humming/utils/device.py`
  补一个 273 GB/s 的 GB10 fallback（notebook 步骤 5 会自动处理）。
- `--mem-fraction-static` 建议从 **0.68** 起步。0.75 在本机触发过 GPU OOM 并导致
  nvidia 驱动锁死（`rmapiLockAcquire`），只能硬断电恢复。

启动后建议确认三项优化确实生效（**静默降级是最常见的坑**——`SGLANG_ENABLE_FP8_LM_HEAD`
不是上游变量，设置一个 SGLang 不认识的环境变量不会有任何报错）：

```bash
grep -oE "moe_runner_backend=[a-z_]*"                 server.log   # humming
grep -c  "Online FP8 quantization enabled for lm_head" server.log   # 1
grep -oE "accept len: [0-9.]+"                        server.log   # 应远大于 1.00
```

Notebook 的步骤 8 已内置这三项检查，任一失败会直接抛异常。

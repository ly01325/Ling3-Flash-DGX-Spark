"""Paired McNemar test: baseline vs optimized on the same 1319 GSM8K questions."""

import json
import math
from pathlib import Path

BASE = Path("/home/liuyang/Ling/benchmarks/gsm8k-baseline-1319-20260828/results")
OPT = Path("/home/liuyang/Ling/benchmarks/gsm8k-full-1319-20260827/results")


def load(d):
    """Return {index: correct} from the per-question jsonl."""
    files = sorted(p for p in d.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no jsonl under {d}")
    out = {}
    for line in open(files[0]):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        idx = r.get("index", r.get("id", len(out)))
        out[idx] = bool(r.get("correct"))
    return files[0].name, out


bn, base = load(BASE)
on, opt = load(OPT)
common = sorted(set(base) & set(opt))

print(f"baseline  file: {bn}  n={len(base)}")
print(f"optimized file: {on}  n={len(opt)}")
print(f"配对题数: {len(common)}\n")

both = sum(1 for i in common if base[i] and opt[i])
b = sum(1 for i in common if base[i] and not opt[i])   # baseline 对，优化版错
c = sum(1 for i in common if not base[i] and opt[i])   # baseline 错，优化版对
neither = sum(1 for i in common if not base[i] and not opt[i])

print("           优化版对  优化版错")
print(f"基线对      {both:6d}   {b:7d}")
print(f"基线错      {c:6d}   {neither:7d}")
print()
print(f"基线准确率  : {sum(base[i] for i in common)}/{len(common)} = "
      f"{sum(base[i] for i in common)/len(common)*100:.2f}%")
print(f"优化版准确率: {sum(opt[i] for i in common)}/{len(common)} = "
      f"{sum(opt[i] for i in common)/len(common)*100:.2f}%")
print()

n = b + c
print(f"不一致对数 (discordant): {n}   (基线独对 {b} / 优化独对 {c})")

if n == 0:
    print("两次结果完全一致，p = 1.0")
else:
    # exact binomial two-sided test
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    print(f"精确二项检验 (双侧): p = {p:.4f}")
    print("结论:", "差异不显著，无法认为两者精度不同 (p >= 0.05)" if p >= 0.05
          else "差异显著 (p < 0.05)")

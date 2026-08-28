#!/usr/bin/env bash
# 阶段3: Humming MoE + 在线 FP8 LM head + MTP3，三项优化全开
# 基于 cookbook 4b.1 (W4A8 / FlashInfer MXFP4)，改动：
#   --moe-runner-backend flashinfer_mxfp4 -> humming
#   --enable-fp32-lm-head 去掉，改用 SGLANG_ENABLE_FP8_LM_HEAD=1（二者互斥）
#   新增 --speculative-* (NEXTN, 3 步)
#   --mem-fraction-static 0.75 -> 0.68（0.75 在本机触发过 GPU OOM 死机）
set -euo pipefail
ROOT=/home/liuyang/Ling
export PATH="${ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${ROOT}/sglang/python"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_DSV4_FP4_DEQUANT=0
export SGLANG_FP8_IGNORED_LAYERS=""
export SGLANG_ENABLE_FP8_LM_HEAD=1
export MAX_JOBS=1
export NINJA_NUM_JOBS=1
MODEL=/home/liuyang/models/Ling-3.0-flash-fp4
exec "${ROOT}/.venv/bin/python" -m sglang.launch_server \
  --model-path "${MODEL}" --served-model-name ling-v3-flash-fp4 \
  --trust-remote-code --dtype bfloat16 --tp-size 1 --ep-size 1 \
  --host 0.0.0.0 --port 30000 --api-key sk-ling-cookbook-test \
  --max-running-requests 1 --max-mamba-cache-size 64 \
  --chunked-prefill-size 8192 --page-size 64 --context-length 262144 \
  --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1 --cuda-graph-bs-decode 1 \
  --cuda-graph-backend-prefill disabled --random-seed 308534008 \
  --reasoning-parser deepseek-r1 --tool-call-parser qwen25 \
  --attention-backend flashinfer --disable-flashinfer-autotune \
  --mem-fraction-static 0.68 --fp8-gemm-backend cutlass \
  --moe-runner-backend humming --flashinfer-mxfp4-moe-precision default \
  --disable-shared-experts-fusion \
  --speculative-algorithm NEXTN --speculative-draft-model-path "${MODEL}" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --json-model-override-args '{"max_position_embeddings":262144,"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'

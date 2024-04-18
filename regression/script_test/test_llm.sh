#!/bin/bash
# test case: test addr_mode, base, io_tag, io_alone
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# block cache
# llama2_7b
model_transform.py \
  --model_name llama2_block_cache_0 \
  --model_def ${NNMODELS_PATH}/llm_models/llama2_block_cache_0.onnx \
  --mlir llama2_block_cache_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/llama2_block_cache_0_input.npz \
  --model llama2_block_cache_0.mlir \
  --output llama2_block_cache_0_top_outputs.npz

model_deploy.py \
  --mlir llama2_block_cache_0.mlir \
  --quantize W4F16 \
  --q_group_size 64 \
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --addr_mode io_alone \
  --test_input ${NNMODELS_PATH}/llm_models/llama2_block_cache_0_input.npz \
  --test_reference llama2_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.84 \
  --model llama2_block_cache_0_static.bmodel

# model_deploy.py \
#   --mlir llama2_block_cache_0.mlir \
#   --quantize W4F16 \
#   --q_group_size 64 \
#   --chip bm1684x \
#   --quant_input \
#   --quant_output \
#   --addr_mode io_alone \
#   --dynamic \
#   --test_input ${NNMODELS_PATH}/llm_models/llama2_block_cache_0_input.npz \
#   --test_reference llama2_block_cache_0_top_outputs.npz \
#   --tolerance 0.98,0.84 \
#   --model llama2_block_cache_0_dynamic.bmodel

# block
# qwen_0.5b
model_transform.py \
  --model_name qwen_block_0 \
  --model_def ${NNMODELS_PATH}/llm_models/qwen_block_0.onnx \
  --mlir qwen_block_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0.mlir \
  --output qwen_block_0_top_outputs.npz

model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --debug \
  --model qwen_block_0.bmodel

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0_bm1684x_w4bf16_tpu.mlir \
  --output qwen_block_0_tpu_outputs.npz

npz_tool.py compare \
  qwen_block_0_top_outputs.npz \
  qwen_block_0_tpu_outputs.npz \
  --tolerance 0.98,0.90 -v

model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --dynamic \
  --quant_input \
  --quant_output \
  --model qwen_block_0_dynamic.bmodel

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0_dynamic.bmodel \
  --output qwen_block_0_model_outputs.npz

npz_tool.py compare \
  qwen_block_0_model_outputs.npz \
  qwen_block_0_tpu_outputs.npz \
  --tolerance 0.98,0.90 -v

# block cache
# qwen_0.5b
model_transform.py \
  --model_name qwen_block_cache_0 \
  --model_def ${NNMODELS_PATH}/llm_models/qwen_block_cache_0.onnx \
  --mlir qwen_block_cache_0.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --model qwen_block_cache_0.mlir \
  --output qwen_block_cache_0_top_outputs.npz

model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model qwen_block_cache_0_static.bmodel

# dynamic
model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --dynamic \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model qwen_block_cache_0_dynamic.bmodel

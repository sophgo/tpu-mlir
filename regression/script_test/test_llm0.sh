#!/bin/bash
# test case: test addr_mode, base, io_tag, io_alone
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"


# ===------------------------------------------------------------===
# Llama2
# ===------------------------------------------------------------===
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
  --model llama2_block_cache_0_bm1684x_static.bmodel

model_deploy.py \
  --mlir llama2_block_cache_0.mlir \
  --quantize W4F16 \
  --q_group_size 64 \
  --chip bm1688 \
  --quant_input \
  --quant_output \
  --addr_mode io_alone \
  --test_input ${NNMODELS_PATH}/llm_models/llama2_block_cache_0_input.npz \
  --test_reference llama2_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.84 \
  --model llama2_block_cache_0_bm1688_1core_static.bmodel

model_deploy.py \
  --mlir llama2_block_cache_0.mlir \
  --quantize W4F16 \
  --q_group_size 64 \
  --chip bm1688 \
  --num_core 2 \
  --quant_input \
  --quant_output \
  --addr_mode io_alone \
  --test_input ${NNMODELS_PATH}/llm_models/llama2_block_cache_0_input.npz \
  --test_reference llama2_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.84 \
  --model llama2_block_cache_0_bm1688_2core_static.bmodel

# dynamic
model_deploy.py \
  --mlir llama2_block_cache_0.mlir \
  --quantize W4F16 \
  --q_group_size 64 \
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --addr_mode io_alone \
  --dynamic \
  --test_input ${NNMODELS_PATH}/llm_models/llama2_block_cache_0_input.npz \
  --test_reference llama2_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.84 \
  --model llama2_block_cache_0_dynamic.bmodel

# parallel
model_deploy.py \
  --mlir llama2_block_cache_0.mlir \
  --quantize W4F16 \
  --q_group_size 64 \
  --chip bm1684x \
  --quant_input \
  --quant_output \
  --addr_mode io_alone \
  --num_device 6 \
  --model llama2_block_cache_0_dev6.bmodel

rm -rf *.npz
rm -rf *.onnx

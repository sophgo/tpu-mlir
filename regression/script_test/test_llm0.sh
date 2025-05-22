#!/bin/bash
# test case: test addr_mode, base, io_tag, io_alone
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

check_fattention() {
    grep -q "tpu.FAttention" "$1" || { echo "Error: no fattention!!!" >&2; exit 1; }
}

mkdir -p test_llm0
pushd test_llm0

# ===------------------------------------------------------------===
# Qwen
# ===------------------------------------------------------------===
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
  --debug \
  --model qwen_block_0.bmodel

check_fattention qwen_block_0_bm1684x_w4bf16_final.mlir

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0_bm1684x_w4bf16_tpu.mlir \
  --output qwen_block_0_tpu_outputs.npz

npz_tool.py compare \
  qwen_block_0_top_outputs.npz \
  qwen_block_0_tpu_outputs.npz \
  --tolerance 0.98,0.90 -v

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0.bmodel \
  --output qwen_block_0_model_outputs.npz

npz_tool.py compare \
  qwen_block_0_model_outputs.npz \
  qwen_block_0_tpu_outputs.npz \
  --tolerance 0.98,0.90 -v

# convert

model_convert.py \
  --model_name qwen_block_0 \
  --model_def ${NNMODELS_PATH}/llm_models/qwen_block_0.onnx \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --model qwen_block_0_v2.bmodel

model_runner.py \
  --input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --model qwen_block_0_v2.bmodel \
  --output qwen_block_0_v2_model_outputs.npz

npz_tool.py compare \
  qwen_block_0_model_outputs.npz \
  qwen_block_0_v2_model_outputs.npz \
  --tolerance 0.99,0.99 -v

# dynamic
model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --dynamic \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --test_reference qwen_block_0_top_outputs.npz \
  --model qwen_block_0_dynamic.bmodel

# parallel
model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --num_device 8 \
  --model qwen_block_0_dev8.bmodel

# bf16
model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize BF16 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --test_reference qwen_block_0_top_outputs.npz \
  --model qwen_block_0_bf16.bmodel

model_deploy.py \
  --mlir qwen_block_0.mlir \
  --quantize BF16 \
  --chip bm1684x \
  --high_precision \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_0_input.npz \
  --test_reference qwen_block_0_top_outputs.npz \
  --model qwen_block_0_bf16_v2.bmodel

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
  --quantize W4BF16 \
  --q_group_size 64 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model qwen_block_cache_0.bmodel

check_fattention qwen_block_cache_0_bm1684x_w4bf16_final.mlir

model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --tolerance 0.98,0.90 \
  --model qwen_block_cache_0_w8bf16.bmodel

# dynamic
model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --dynamic \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --model qwen_block_cache_0_dynamic.bmodel

# parallel
model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize W8BF16 \
  --chip bm1684x \
  --num_device 2 \
  --model qwen_block_cache_0_dev2.bmodel

# bf16
model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize BF16 \
  --chip bm1684x \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --model qwen_block_cache_0_bf16.bmodel

model_deploy.py \
  --mlir qwen_block_cache_0.mlir \
  --quantize BF16 \
  --chip bm1684x \
  --high_precision \
  --test_input ${NNMODELS_PATH}/llm_models/qwen_block_cache_0_input.npz \
  --test_reference qwen_block_cache_0_top_outputs.npz \
  --model qwen_block_cache_0_bf16_v2.bmodel

# combine prefill and decode, the size should be same

check_merge() {
  local a="$1"
  local b="$2"

  model_tool --combine $a $b -o llm_merge.bmodel

  # Get the sizes of the files in bytes
  size_a=$(stat -c%s $a)
  size_b=$(stat -c%s $b)
  size_m=$(stat -c%s llm_merge.bmodel)

  # Calculate the size of size_m minus 2 MB (2 * 1024 * 1024 bytes)
  size_m_minus=$((size_m - 2 * 1024 * 1024))

  # Compare the sizes
  if [ "$size_a" -le "$size_m_minus" ]; then
    echo "Error: qwen merge size is uncorrect"
    return 1
  fi

  if [ "$size_b" -le "$size_m_minus" ]; then
    echo "Error: qwen merge size is uncorrect"
    return 1
  fi
  return 0
}

check_merge qwen_block_0.bmodel qwen_block_cache_0.bmodel
check_merge qwen_block_0_bf16.bmodel qwen_block_cache_0_bf16.bmodel

rm -rf *.npz *.onnx *.bmodel

popd

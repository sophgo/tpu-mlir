#!/bin/bash

set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p test_pruning
pushd test_pruning

# test matmul case
python3 $DIR/test_pruning.py

model_transform.py \
    --model_name mymatmul \
    --model_def mymatmul.onnx \
    --mlir mymatmul.mlir \
    --pruning test_pruning.json

model_runner.py --input mymatmul_in_f32.npz --model mymatmul.mlir --output matmul_top_outputs.npz  

model_deploy.py \
    --mlir mymatmul.mlir \
    --quantize W4F16 \
    --q_group_size 64 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --test_input mymatmul_in_f32.npz \
    --test_reference matmul_output.npz \
    --model mymatmul.bmodel

rm -rf *.npz *.onnx *.bmodel
popd


# # cp block_input.npz qwen_block_0_in_f32.npz
# model_transform.py \
#     --model_name qwen_block_0 \
#     --model_def qwen_block_0.onnx \
#     --mlir qwen_block_0.mlir \
#     --pruning block_pruning_config.json
# model_runner.py --input qwen_block_0_input.npz --model qwen_block_0.mlir --output qwen_block_0_top_outputs.npz  
# # model_runner.py --input qwen_block_0_input.npz --model qwen_block_0.onnx --output qwen_block_0_ref_outputs.npz  
# model_deploy.py \
#     --mlir qwen_block_0.mlir \
#     --quantize BF16 \
#     --q_group_size 64 \
#     --quant_input \
#     --quant_output \
#     --chip bm1684x \
#     --test_input qwen_block_0_input.npz \
#     --test_reference qwen_block_0_top_outputs.npz \
#     --model mymatmul.bmodel


# test_block
# model_transform.py \
#     --model_name block_model \
#     --model_def pruned_model.onnx \
#     --mlir block_model.mlir \
#     --pruning data.json

# model_runner.py --input block_model_in_f32.npz --model block_model.mlir --output block_model_top_outputs.npz  
# # model_runner.py --input block_model_in_f32.npz --model pruned_model.onnx --output block_model_ref_outputs.npz  

# model_deploy.py \
#     --mlir block_model.mlir \
#     --quantize W4F16 \
#     --q_group_size 64 \
#     --quant_input \
#     --quant_output \
#     --chip bm1684x \
#     --test_input block_model_in_f32.npz \
#     --test_reference block_model_ref_outputs.npz \
#     --model mymatmul.bmodel

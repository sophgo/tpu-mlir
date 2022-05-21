#!/bin/bash
set -ex

#########################
# transform to top mlir
#########################
INPUT=../resnet18_in_f32_b4.npz
mkdir -p tmp
pushd tmp
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --input_shapes [[4,3,224,224]] \
    --model_def  ../resnet18.onnx \
    --test_input $INPUT \
    --test_result resnet18_top_outputs.npz \
    --mlir resnet18.mlir

#########################
# deploy to bmodel
#########################
# to fp32 model
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize F32 \
  --chip bm1686 \
  --test_input $INPUT \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet18_1686_f32.bmodel

# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize F16 \
#   --chip bm1686 \
#   --test_input $INPUT \
#   --test_reference resnet18_top_outputs.npz \
#   --tolerance 0.99,0.99 \
#   --model resnet18_1686_f16.bmodel

# model_deploy.py \
#   --mlir resnet18.mlir \
#   --quantize BF16 \
#   --chip bm1686 \
#   --test_input $INPUT \
#   --test_reference resnet18_top_outputs.npz \
#   --tolerance 0.99,0.99 \
#   --model resnet18_1686_bf16.bmodel

popd

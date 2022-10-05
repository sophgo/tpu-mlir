#!/bin/bash
set -ex

mkdir -p step_by_step
pushd step_by_step

#################################
# Convert to top mlir
#################################
model_transform.py \
  --model_name resnet18 \
  --model_def  ${REGRESSION_PATH}/model/resnet18.onnx \
  --input_shapes [[1,3,224,224]] \
  --resize_dims 256,256 \
  --mean 123.675,116.28,103.53 \
  --scale 0.0171,0.0175,0.0174 \
  --pixel_format rgb \
  --test_input ${REGRESSION_PATH}/image/cat.jpg \
  --test_result resnet18_top_outputs.npz \
  --mlir resnet18.mlir

#################################
# Mlir to f32 bmodel
#################################

# lowering
tpuc-opt resnet18.mlir \
    --convert-top-to-tpu="mode=F32 chip=bm1684x" \
    --save-weight \
    -o resnet18_tpu_f32.mlir

# weigth reorder
tpuc-opt resnet18_tpu_f32.mlir \
    --weight-reorder \
    --save-weight \
    -o resnet18_tpu_f32_reordered.mlir

# divide subnet
tpuc-opt resnet18_tpu_f32_reordered.mlir \
    --subnet-divide \
    --save-weight \
    -o resnet18_tpu_f32_subnet.mlir

# layer group
tpuc-opt resnet18_tpu_f32_subnet.mlir \
    --layer-group \
    --save-weight \
    -o resnet18_tpu_f32_lg.mlir

# address-assign gmem
tpuc-opt resnet18_tpu_f32_lg.mlir \
    --address-assign \
    --save-weight \
    -o resnet18_tpu_f32_addr.mlir

# codegen
tpuc-opt resnet18_tpu_f32_addr.mlir \
    --codegen="model_file=resnet18_f32_1684x.bmodel" \
    -o resnet18_tpu_f32_final.mlir

# inference tpu mlir and bmodel, and check result
model_runner.py \
    --model resnet18_tpu_f32.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_tpu_f32_outputs.npz

npz_tool.py compare \
    resnet18_tpu_f32_outputs.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.99,0.99 -v

model_runner.py \
    --model resnet18_f32_1684x.bmodel \
    --input resnet18_in_f32.npz \
    --output resnet18_model_f32_outputs.npz

npz_tool.py compare \
    resnet18_tpu_f32_outputs.npz \
    resnet18_model_f32_outputs.npz \
    --tolerance 0.99,0.99 -v


#################################
# Mlir to int8 symmetric bmodel
#################################
# calibration
run_calibration.py resnet18.mlir \
  --dataset $REGRESSION_PATH/image \
  --input_num 2 \
  -o resnet18_cali_table

# lowering to symetric int8
tpuc-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=false' \
    --convert-top-to-tpu="mode=INT8 asymmetric=false chip=bm1684x" \
    --save-weight \
    -o resnet18_tpu_int8_sym.mlir

# to symmetric bmodel （all pass in one)
tpuc-opt resnet18_tpu_int8_sym.mlir \
   --weight-reorder \
   --subnet-divide \
   --layer-group \
   --address-assign \
   --save-weight \
   --codegen="model_file=resnet18_int8_sym_1684x.bmodel" \
   -o resnet18_tpu_int8_sym_final.mlir

# inference tpu mlir and bmodel, and check result
model_runner.py \
    --model resnet18_tpu_int8_sym.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_tpu_int8_sym_outputs.npz

npz_tool.py compare \
    resnet18_tpu_int8_sym_outputs.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.95,0.70 -v

model_runner.py \
    --model resnet18_int8_sym_1684x.bmodel \
    --input resnet18_in_f32.npz \
    --output resnet18_model_int8_sym_outputs.npz

npz_tool.py compare \
    resnet18_model_int8_sym_outputs.npz \
    resnet18_tpu_int8_sym_outputs.npz \
    --tolerance 0.99,0.95 -v

#################################
# Mlir to int8 asymmetric bmodel
#################################

# lowering to asymmetric int8
tpuc-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=true' \
    --convert-top-to-tpu="mode=INT8 asymmetric=true chip=bm1684x" \
    --save-weight \
    -o resnet18_tpu_int8_asym.mlir

# to asymmetric bmodel （all pass in one)
tpuc-opt resnet18_tpu_int8_asym.mlir \
   --weight-reorder \
   --subnet-divide \
   --layer-group \
   --address-assign \
   --save-weight \
   --codegen="model_file=resnet18_int8_asym_1684x.bmodel" \
   -o resnet18_tpu_int8_asym_final.mlir

model_runner.py \
    --model resnet18_tpu_int8_asym.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_tpu_int8_asym_outputs.npz

npz_tool.py compare \
    resnet18_tpu_int8_asym_outputs.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.90,0.54 -v

model_runner.py \
    --model resnet18_int8_asym_1684x.bmodel \
    --input resnet18_in_f32.npz \
    --output resnet18_model_int8_asym_outputs.npz

npz_tool.py compare \
    resnet18_model_int8_asym_outputs.npz \
    resnet18_tpu_int8_asym_outputs.npz \
    --tolerance 0.99,0.95 -v

#################################
# Mlir to f16 bmodel
#################################

tpuc-opt resnet18.mlir \
    '--convert-top-to-tpu=mode=F16 chip=bm1684x' \
    --save-weight \
    -o resnet18_f16_1684x.mlir

model_runner.py \
    --model resnet18_f16_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_f16_outputs_1684x.npz

# npz_tool.py compare \
#     resnet18_f16_outputs_1684x.npz \
#     resnet18_top_outputs.npz \
#     --tolerance 0.99,0.90 -v

# ToDo (convert to f16 bmodel)
# tpuc-opt resnet18_tpu_f16.mlir \
#    --weight-reorder \
#    --subnet-divide \
#    --layer-group \
#    --address-assign \
#    --save-weight \
#    --codegen="model_file=resnet18_f16_1684x.bmodel" \
#    -o resnet18_tpu_f16_final.mlir

#################################
# Mlir to bf16 bmodel
#################################

tpuc-opt resnet18.mlir \
    '--convert-top-to-tpu=mode=BF16 chip=bm1684x' \
    --save-weight \
    -o resnet18_bf16_1684x.mlir

model_runner.py \
    --model resnet18_bf16_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_bf16_outputs_1684x.npz

# npz_tool.py compare \
#     resnet18_bf16_outputs_1684x.npz \
#     resnet18_top_outputs.npz \
#     --tolerance 0.99,0.80 -v

# ToDo (convert to bf16 bmodel)
# tpuc-opt resnet18_tpu_bf16.mlir \
#    --weight-reorder \
#    --subnet-divide \
#    --layer-group \
#    --address-assign \
#    --save-weight \
#    --codegen="model_file=resnet18_bf16_1684x.bmodel" \
#    -o resnet18_tpu_bf16_final.mlir

popd

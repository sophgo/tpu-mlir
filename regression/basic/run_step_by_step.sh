#!/bin/bash
set -ex

mkdir -p step_by_step
pushd step_by_step

model_transform.py \
  --model_type onnx \
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

run_calibration.py resnet18.mlir \
  --dataset $REGRESSION_PATH/image \
  --input_num 2 \
  -o resnet18_cali_table

#########################
# BM1684x
#########################

# convert to int8
tpuc-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=true' \
    --save-weight \
    -o resnet18_cali_1684x.mlir

# quantize mlir for 1684x asymmetric
tpuc-opt resnet18_cali_1684x.mlir \
    --lowering="mode=INT8 asymmetric=true chip=bm1684x" \
    --save-weight \
    -o resnet18_int8_1684x_asym.mlir

model_runner.py \
    --model resnet18_int8_1684x_asym.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_int8_outputs_1684x_asym.npz

npz_tool.py compare \
    resnet18_int8_outputs_1684x_asym.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.90,0.54 -v

#ILSVRC2012_img_val_with_subdir is converted by valprep.sh
#model_eval.py \
#    --mlir_file resnet18_int8_1684x_asym.mlir \
#    --dataset /data/ILSVRC2012_img_val_with_subdir/ \
#    --count 1000

#model_eval.py \
#    --mlir_file resnet18_int8_1684x_asym.mlir \
#    --data_list /data/list_val.txt  \
#    --label_file /data/list_val.txt \
#    --dataset_type user_define \
#    --count 1000


tpuc-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=false' \
    --save-weight \
    -o resnet18_cali_1684x_sym.mlir

# quantize mlir for 1684x symmetric
tpuc-opt resnet18_cali_1684x_sym.mlir \
    --lowering="mode=INT8 asymmetric=false chip=bm1684x" \
    --save-weight \
    -o resnet18_int8_1684x_sym.mlir

model_runner.py \
    --model resnet18_int8_1684x_sym.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_int8_outputs_1684x_sym.npz

npz_tool.py compare \
    resnet18_int8_outputs_1684x_sym.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.95,0.70 -v

# convert f16
tpuc-opt resnet18.mlir \
    '--lowering=mode=F16 chip=bm1684x' \
    --save-weight \
    -o resnet18_f16_1684x.mlir

model_runner.py \
    --model resnet18_f16_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_f16_outputs_1684x.npz

npz_tool.py compare \
    resnet18_f16_outputs_1684x.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.99,0.90 -v

# convert bf16
tpuc-opt resnet18.mlir \
    '--lowering=mode=BF16 chip=bm1684x' \
    --save-weight \
    -o resnet18_bf16_1684x.mlir

model_runner.py \
    --model resnet18_bf16_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_bf16_outputs_1684x.npz

npz_tool.py compare \
    resnet18_bf16_outputs_1684x.npz \
    resnet18_top_outputs.npz \
    --tolerance 0.99,0.85 -v
popd

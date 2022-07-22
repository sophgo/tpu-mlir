#!/bin/bash
set -ex
rm -rf resnet50_v1
mkdir -p resnet50_v1
pushd resnet50_v1

model_transform.py \
    --model_name resnet50_v1 \
    --model_def  ${NNMODELS_PATH}/onnx_models/resnet50-v1-7.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result resnet50_v1_top_outputs.npz \
    --mlir resnet50_v1.mlir

model_eval.py \
   --mlir_file resnet50_v1.mlir \
   --dataset /data/ILSVRC2012_img_val_with_subdir/ \
   --count 1000

# calibration
run_calibration.py resnet50_v1.mlir \
  --dataset $REGRESSION_PATH/ILSVRC2012 \
  --input_num 200 \
  -o resnet50_v1_cali_table

# lowering to symetric int8
tpuc-opt resnet50_v1.mlir \
    --import-calibration-table='file=resnet50_v1_cali_table asymmetric=false' \
    --lowering="mode=INT8 asymmetric=false chip=bm1684x" \
    --save-weight \
    -o resnet50_v1_tpu_int8_sym.mlir


model_eval.py \
   --mlir_file resnet50_v1_tpu_int8_sym.mlir \
   --dataset /data/ILSVRC2012_img_val_with_subdir/ \
   --count 1000

# calibration
run_calibration.py resnet50_v1.mlir \
  --dataset $REGRESSION_PATH/ILSVRC2012 \
  --input_num 200 \
  --tune_num 30 \
  -o resnet50_v1_cali_table_tuned # --debug_cmd 'debug_log'

tpuc-opt resnet50_v1.mlir \
    --import-calibration-table='file=resnet50_v1_cali_table_tuned asymmetric=false' \
    --lowering="mode=INT8 asymmetric=false chip=bm1684x" \
    --save-weight \
    -o resnet50_v1_tpu_int8_sym_tuned.mlir

model_eval.py \
   --mlir_file resnet50_v1_tpu_int8_sym_tuned.mlir \
   --dataset /data/ILSVRC2012_img_val_with_subdir/ \
   --count 1000

popd

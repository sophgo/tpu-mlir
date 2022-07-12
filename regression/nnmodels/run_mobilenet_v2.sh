!/bin/bash
set -ex

mkdir -p mobilenet_v2
pushd mobilenet_v2

model_transform.py \
    --model_name mobilenet_v2 \
    --model_def  ${NNMODELS_PATH}/onnx_models/mobilenetv2-7.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ${REGRESSION_PATH}/image/cat.jpg \
    --test_result mobilenet_v2_top_outputs.npz \
    --mlir mobilenet_v2.mlir \


#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input mobilenet_v2_in_f32.npz \
  --test_reference mobilenet_v2_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model mobilenet_v2_1684x_f32.bmodel


#########################
# deploy to int8 bmodel
#########################
run_calibration.py mobilenet_v2.mlir \
  --dataset ${REGRESSION_PATH}/image \
  --input_num 2 \
  -o mobilenet_v2_cali_table

# to symmetric
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize INT8 \
  --calibration_table mobilenet_v2_cali_table \
  --chip bm1684x \
  --test_input mobilenet_v2_in_f32.npz \
  --test_reference mobilenet_v2_top_outputs.npz \
  --tolerance 0.96,0.73 \
  --correctness 0.99,0.95 \
  --model mobilenet_v2_1684x_int8_sym.bmodel

# to asymmetric
model_deploy.py \
  --mlir mobilenet_v2.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table mobilenet_v2_cali_table \
  --chip bm1684x \
  --test_input mobilenet_v2_in_f32.npz \
  --test_reference mobilenet_v2_top_outputs.npz \
  --tolerance 0.98,0.79 \
  --correctness 0.99,0.95 \
  --model mobilenet_v2_1684x_int8_asym.bmodel

popd

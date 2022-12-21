#!/bin/bash
set -ex

# all test (f32/f16/bf16/int8): run_model.sh mobilenet_v2 bm1684x all
# basic test (f32/int8): run_model.sh mobilenet_v2 bm1684x basic

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

model_name=$1
chip_name=$2
test_type=$3

if [ x$1 == x ]; then
  echo "Error: $0 model_name [bm1684x|cv183x]"
  exit 1
fi

if [ x$2 == x ]; then
  chip_name=bm1684x
fi

if [ x$test_type == x ]; then
  test_type=all
fi

cfg_file=$REGRESSION_PATH/config/${model_name}.cfg

if [ ! -f $cfg_file ]; then
  echo "Error: can't open config file ${cfg_file}"
  exit 1
fi

source $REGRESSION_PATH/chip.cfg
eval do_f32=\${${chip_name}_support_f32}
eval do_bf16=\${${chip_name}_support_bf16}
eval do_f16=\${${chip_name}_support_f16}
eval do_asymmetric=\${${chip_name}_support_asym}
eval do_symmetric=\${${chip_name}_support_sym}
eval model_type=\${${chip_name}_model_type}

# basic test don't run bf16/f16
if [ x${test_type} == xbasic ]; then
  do_f16=0
  do_bf16=0
fi

source ${cfg_file}

NET_DIR=$REGRESSION_PATH/regression_out/${model_name}_${chip_name}
mkdir -p $NET_DIR
pushd $NET_DIR

model_def_opt=
if [ x$model_path != x ] && [ -f $model_path ]; then
  model_def_opt="--model_def ${model_path}"
elif [ x$model_path2 != x ] && [ -f $model_path2 ]; then
  model_path=${model_path2}
  model_def_opt="--model_def ${model_path2}"
else
  echo "Error: can't find model file for ${model_name}"
  exit 1
fi

model_data_opt=
# caffemodel
if echo ${model_path} | grep -q -E '\.prototxt$'; then
  if [ x$model_data != x ] && [ -f $model_data ]; then
    model_data_opt="--model_data ${model_data}"
  else
    echo "Error: no caffemodel"
    exit 1
  fi
fi

# tflite model
do_cali=1
if echo ${model_path} | grep -q -E '\.tflite$'; then
  do_cali=0
  do_f32=0
  do_f16=0
  do_bf16=0
  do_symmetric=0
  do_asymmetric=1
fi

excepts_opt=
if [ x${excepts} != x ]; then
  excepts_opt="--excepts=${excepts}"
fi

input_shapes_opt=
if [ x${input_shapes} != x ]; then
  input_shapes_opt="--input_shapes=${input_shapes}"
fi

resize_dims_opt=
if [ x${resize_dims} != x ]; then
  resize_dims_opt="--resize_dims=${resize_dims}"
fi

keep_aspect_ratio_opt=
if [ x${keep_aspect_ratio} == x1 ]; then
  keep_aspect_ratio_opt="--keep_aspect_ratio"
fi

output_names_opt=
if [ x${output_names} != x ]; then
  output_names_opt="--output_names=${output_names}"
fi

mean_opt=
if [ x${mean} != x ]; then
  mean_opt="--mean=${mean}"
fi

scale_opt=
if [ x${scale} != x ]; then
  scale_opt="--scale=${scale}"
fi

pixel_format_opt=
if [ x${pixel_format} != x ]; then
  pixel_format_opt="--pixel_format=${pixel_format}"
fi

channel_format_opt=
if [ x${channel_format} != x ]; then
  channel_format_opt="--channel_format=${channel_format}"
fi

pad_value_opt=
if [ x${pad_value} != x ]; then
  channel_format_opt="--pad_value=${pad_value}"
fi

pad_type_opt=
if [ x${pad_type} != x ]; then
  channel_format_opt="--pad_type=${pad_type}"
fi

model_format_opt=
if [ x${model_format} != x ]; then
  model_format_opt="--model_format=${model_format}"
fi

test_input_opt=
test_result_opt=
test_innpz_opt=
test_reference_opt=
top_result=${model_name}_top_outputs.npz
if [ x${test_input} != x ]; then
  test_input_opt="--test_input=${test_input}"
  test_result_opt="--test_result=${top_result}"
  test_innpz_opt="--test_input=${model_name}_in_f32.npz"
  test_reference_opt="--test_reference=${top_result}"
fi

model_transform.py \
  --model_name ${model_name} \
  ${model_def_opt} \
  ${model_data_opt} \
  ${output_names_opt} \
  ${input_shapes_opt} \
  ${resize_dims_opt} \
  ${keep_aspect_ratio_opt} \
  ${mean_opt} \
  ${scale_opt} \
  ${pixel_format_opt} \
  ${channel_format_opt} \
  ${model_format_opt} \
  ${test_input_opt} \
  ${test_result_opt} \
  ${excepts_opt} \
  --mlir ${model_name}.mlir

#########################
# deploy to float bmodel
#########################
if [ ${do_f32} == 1 ]; then
  model_deploy.py \
    --mlir ${model_name}.mlir \
    --quantize F32 \
    --chip ${chip_name} \
    ${test_innpz_opt} \
    ${test_reference_opt} \
    ${excepts_opt} \
    --tolerance 0.99,0.99 \
    --model ${model_name}_${chip_name}_f32.${model_type}
fi

if [ ${do_f16} == 1 ]; then
  model_deploy.py \
    --mlir ${model_name}.mlir \
    --quantize F16 \
    --chip ${chip_name} \
    ${test_innpz_opt} \
    ${test_reference_opt} \
    ${excepts_opt} \
    --tolerance 0.95,0.85 \
    --model ${model_name}_${chip_name}_f16.${model_type}
fi

if [ ${do_bf16} == 1 ]; then
  model_deploy.py \
    --mlir ${model_name}.mlir \
    --quantize BF16 \
    --chip ${chip_name} \
    ${test_innpz_opt} \
    ${test_reference_opt} \
    ${excepts_opt} \
    --tolerance 0.95,0.85 \
    --model ${model_name}_${chip_name}_bf16.${model_type}
fi
#########################
# deploy to int8 bmodel
#########################

# only once
CALI_TABLE=${REGRESSION_PATH}/cali_tables/${model_name}_cali_table
QTABLE=${REGRESSION_PATH}/cali_tables/${model_name}_qtable
if [ ${do_cali} == 1 ] && [ ! -f ${CALI_TABLE} ]; then
  if [ x${dataset} == x ]; then
    echo "Error: ${model_name} has no dataset"
    exit 1
  fi
  run_calibration.py ${model_name}.mlir \
    --dataset ${dataset} \
    --input_num 100 \
    -o $CALI_TABLE
fi

cali_opt=
if [ -f ${CALI_TABLE} ]; then
  cali_opt="--calibration_table ${CALI_TABLE}"
fi

qtable_opt=
if [ x${use_quantize_table} == x1 ]; then
  if [ ! -f ${QTABLE} ]; then
    echo "Error: ${QTABLE} not exist"
    exit 1
  fi
  qtable_opt="--quantize_table ${QTABLE}"
fi

# to symmetric
if [ ${do_symmetric} == 1 ]; then

  tolerance_sym_opt=
  if [ x${int8_sym_tolerance} != x ]; then
    tolerance_sym_opt="--tolerance ${int8_sym_tolerance}"
  fi
  model_deploy.py \
    --mlir ${model_name}.mlir \
    --quantize INT8 \
    ${cali_opt} \
    ${qtable_opt} \
    --chip ${chip_name} \
    ${test_innpz_opt} \
    ${test_reference_opt} \
    ${tolerance_sym_opt} \
    ${excepts_opt} \
    --quant_input \
    --quant_output \
    --model ${model_name}_${chip_name}_int8_sym.${model_type}

fi #do_symmetric

# to asymmetric
if [ $do_asymmetric == 1 ]; then

  tolerance_asym_opt=
  if [ x${int8_asym_tolerance} != x ]; then
    tolerance_asym_opt="--tolerance ${int8_asym_tolerance}"
  fi
  model_deploy.py \
    --mlir ${model_name}.mlir \
    --quantize INT8 \
    --asymmetric \
    ${cali_opt} \
    ${qtable_opt} \
    --chip ${chip_name} \
    ${test_innpz_opt} \
    ${test_reference_opt} \
    ${tolerance_asym_opt} \
    ${excepts_opt} \
    --model ${model_name}_${chip_name}_int8_asym.${model_type}

fi #do_asymmetric

#########################
# app
#########################
if [ x${app} != x ] && [ x${chip_name} != xcv183x ]; then

  # by onnx
  ${app} \
    --input ${test_input} \
    --model ${model_path} \
    --output output_onnx.jpg

  # by f32 bmodel
  if [ x${do_f32} == x1 ]; then
    ${app} \
      --input ${test_input} \
      --model ${model_name}_${chip_name}_f32.${model_type} \
      --output output_f32.jpg
  fi

  # by int8 symmetric bmodel
  if [ x${do_symmetric} == x1 ]; then
    ${app} \
      --input ${test_input} \
      --model ${model_name}_${chip_name}_int8_sym.${model_type} \
      --output output_int8_sym.jpg
  fi

  if [ $do_asymmetric == 1 ]; then
    # by int8 asymmetric bmodel
    ${app} \
      --input ${test_input} \
      --model ${model_name}_${chip_name}_int8_asym.${model_type} \
      --output output_int8_asym.jpg
  fi

fi

popd

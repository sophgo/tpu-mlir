#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_name=$1
if [ x$1 == x ]; then
  echo "Error: $0 model_name"
  exit 1
fi

cfg_file=$REGRESSION_PATH/config/$1.cfg

if [ ! -f $cfg_file ]; then
  echo "Error: can't open config file ${cfg_file}"
  exit 1
fi

do_asymmetric=1

source ${cfg_file}

mkdir -p regression_out/${model_name}
pushd regression_out/${model_name}

model_def_opt=
if [ -f $model_path ]; then
  model_def_opt="--model_def ${model_path}"
elif [ -f $model_path2 ]; then
  model_path=${model_path2}
  model_def_opt="--model_def ${model_path2}"
else
  echo "Error: can't find model file for ${model_name}"
  exit 1
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
  ${output_names_opt} \
  ${input_shapes_opt} \
  ${resize_dims_opt} \
  ${keep_aspect_ratio_opt} \
  ${mean_opt} \
  ${scale_opt} \
  ${pixel_format_opt} \
  ${test_input_opt} \
  ${test_result_opt} \
  --mlir ${model_name}.mlir

#########################
# deploy to float bmodel
#########################
model_deploy.py \
  --mlir ${model_name}.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input ${model_name}_in_f32.npz \
  --test_reference ${top_result} \
  --tolerance 0.99,0.99 \
  --model ${model_name}_bm1684x_f32.bmodel

#########################
# deploy to int8 bmodel
#########################

# only once
CALI_TABLE=${REGRESSION_PATH}/cali_tables/${model_name}_cali_table
if [ ! -f ${CALI_TABLE} ]; then
  if [ x${dataset} == x ]; then
    echo "Error: ${model_name} has no dataset"
    exit 1
  fi
  run_calibration.py ${model_name}.mlir \
    --dataset ${dataset} \
    --input_num 100 \
    -o $CALI_TABLE
fi

# to symmetric
tolerance_sym_opt=
if [ x${int8_sym_tolerance} != x ]; then
  tolerance_sym_opt="--tolerance ${int8_sym_tolerance}"
fi
model_deploy.py \
  --mlir ${model_name}.mlir \
  --quantize INT8 \
  --calibration_table $CALI_TABLE \
  --chip bm1684x \
  ${test_innpz_opt} \
  ${test_reference_opt} \
  ${tolerance_sym_opt} \
  --correctness 0.99,0.90 \
  --model ${model_name}_bm1684x_int8_sym.bmodel

# to asymmetric
if [ x$do_asymmetric == x1 ]; then

tolerance_asym_opt=
if [ x${int8_asym_tolerance} != x ]; then
  tolerance_asym_opt="--tolerance ${int8_asym_tolerance}"
fi
model_deploy.py \
  --mlir ${model_name}.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table $CALI_TABLE \
  --chip bm1684x \
  ${test_innpz_opt} \
  ${test_reference_opt} \
  ${tolerance_asym_opt} \
  --correctness 0.99,0.90 \
  --model ${model_name}_bm1684x_int8_asym.bmodel
fi

#########################
# app
#########################
if [ x${app} != x ]; then

# by onnx
${app} \
  --input ${test_input} \
  --model ${model_path} \
  --output output_onnx.jpg

# by f32 bmodel
${app} \
  --input ${test_input} \
  --model ${model_name}_bm1684x_f32.bmodel \
  --output output_f32.jpg

# by int8 symmetric bmodel
${app} \
  --input ${test_input} \
  --model ${model_name}_bm1684x_int8_sym.bmodel \
  --output output_int8_sym.jpg

if [ x$do_asymmetric == x1 ]; then
# by int8 asymmetric bmodel
${app} \
  --input ${test_input} \
  --model ${model_name}_bm1684x_int8_asym.bmodel \
  --output output_int8_asym.jpg
fi

fi

popd

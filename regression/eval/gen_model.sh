#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_name=$1
if [ x$1 == x ]; then
  echo "Error: $0 model_name"
  exit 1
fi

cfg_file=$REGRESSION_PATH/config/$1.cfg
if [ x$2 != x ]; then
  cfg_file=$2
fi

if [ ! -f $cfg_file ]; then
  echo "Error: can't open config file ${cfg_file}"
  exit 1
fi

source ${cfg_file}

mkdir -p ${model_name}
pushd ${model_name}

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

pad_value_opt=
if [ x${pad_value} != x ]; then
  pad_value_opt="--pad_value=${pad_value}"
fi

pad_type_opt=
if [ x${pad_type} != x ]; then
  pad_type_opt="--pad_type=${pad_type}"
fi

debug_cmd_opt=
if [ x${debug_cmd} != x ]; then
  debug_cmd_opt="--debug_cmd=${debug_cmd}"
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
  ${pad_value_opt} \
  ${pad_type_opt} \
  ${debug_cmd_opt} \
  ${test_input_opt} \
  ${test_result_opt} \
  --mlir ${model_name}.mlir

# only once
CALI_TABLE=${REGRESSION_PATH}/cali_tables/${model_name}_cali_table
if [ x${specified_cali_table} != x ]; then
  CALI_TABLE=${specified_cali_table}
fi
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

# lowering to symetric int8
tpuc-opt ${model_name}.mlir \
    --import-calibration-table="file=${CALI_TABLE} asymmetric=false" \
    --convert-top-to-tpu="mode=INT8 asymmetric=false chip=bm1684x" \
    --canonicalize \
    -o ${model_name}_bm1684x_tpu_int8_sym.mlir

model_runner.py \
    --model ${model_name}_bm1684x_tpu_int8_sym.mlir \
    --input ${model_name}_in_f32.npz \
    --dump_all_tensors \
    --output ${model_name}_bm1684x_tpu_int8_sym_outputs.npz

# lowering to asymmetric int8
tpuc-opt ${model_name}.mlir \
    --import-calibration-table="file=${CALI_TABLE} asymmetric=true" \
    --convert-top-to-tpu="mode=INT8 asymmetric=true chip=bm1684x" \
    --canonicalize \
    -o ${model_name}_bm1684x_tpu_int8_asym.mlir

model_runner.py \
    --model ${model_name}_bm1684x_tpu_int8_asym.mlir \
    --input ${model_name}_in_f32.npz \
    --dump_all_tensors \
    --output ${model_name}_bm1684x_tpu_int8_asym_outputs.npz


npz_tool.py compare \
    ${model_name}_bm1684x_tpu_int8_sym_outputs.npz \
    ${model_name}_top_outputs.npz \
    --tolerance ${int8_sym_tolerance} -v

npz_tool.py compare \
    ${model_name}_bm1684x_tpu_int8_asym_outputs.npz \
    ${model_name}_top_outputs.npz \
    --tolerance ${int8_asym_tolerance} -v

popd

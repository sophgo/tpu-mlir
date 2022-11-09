#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_name=$1
full_test=$2
if [ x$1 == x ]; then
  echo "Error: $0 model_name"
  exit 1
fi

cfg_file=$REGRESSION_PATH/config/$1.cfg

if [ ! -f $cfg_file ]; then
  echo "Error: can't open config file ${cfg_file}"
  exit 1
fi

if [ x${full_test} == x0 ]; then
  do_bf16=1
  do_cali=1
  do_symmetric=1
else
  do_bf16=1
  do_cali=1
  do_symmetric=1
fi

source ${cfg_file}

NET_DIR=$REGRESSION_PATH/regression_out/${model_name}
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
  ${test_input_opt} \
  ${test_result_opt} \
  ${excepts_opt} \
  --mlir ${model_name}.mlir

#########################
# deploy to bf16 cvimodel
#########################
if [ ${do_bf16} == 1 ]; then
model_deploy.py \
  --mlir ${model_name}.mlir \
  --quantize BF16 \
  --chip cv183x \
  ${test_innpz_opt} \
  ${test_reference_opt} \
  ${excepts_opt} \
  --tolerance 0.95,0.85 \
  --model ${model_name}_cv183x_bf16.cvimodel
fi
#########################
# deploy to int8 cvimodel
#########################

# only once
CALI_TABLE=${REGRESSION_PATH}/config/${model_name}_cali_table
QTABLE=${REGRESSION_PATH}/config/${model_name}_qtable
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
if [ x${int8_sym_tolerance_18xx} != x ]; then
  tolerance_sym_opt="--tolerance ${int8_sym_tolerance_18xx}"
fi
model_deploy.py \
  --mlir ${model_name}.mlir \
  --quantize INT8 \
  ${cali_opt} \
  ${qtable_opt} \
  --chip cv183x \
  ${test_innpz_opt} \
  ${test_reference_opt} \
  ${tolerance_sym_opt} \
  ${excepts_opt} \
  --quant_output  \
  --model ${model_name}_cv183x_int8_sym.cvimodel

fi #do_symmetric
popd

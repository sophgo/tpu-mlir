#!/bin/bash
###########################################################
# usage: ./generic_models.sh  model_name \
#                             [model_zoo_path=/data/mlir-models] \
#                             [dataset_path=/data/dataset]
###############################################################


set -ex
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ x$1 == x ]; then
  echo "Error: $0 NET"
  exit 1
fi
NET=$1
CHIP_NAME=${SET_CHIP_NAME:-cv183x}
cfg_file=$REGRESSION_PATH/cv18xx_porting/config/model_config_18xx.sh

if [ ! -f $cfg_file ]; then
  echo "Error: can't open config file ${cfg_file}"
  exit 1
fi

if [ x$2 != x ]; then
  MODEL_PATH=$2
fi
MODEL_PATH=${MODEL_PATH:-/data/mlir-models}
if [ ! -d $MODEL_PATH ]; then
  echo "Error: model path not exist\n"
  exit 1
fi

if [ x$3 != x ]; then
  DATA_SET=$3
fi
DATA_SET=${DATA_SET:-/data/dataset}
if [ ! -d $DATA_SET ]; then
  echo "Error: model dataset not exist\n"
  exit 1
fi

export NET=${NET}
export MODEL_PATH=${MODEL_PATH}
export DATA_SET=${DATA_SET}
source ${cfg_file}

NET_DIR=$REGRESSION_PATH/cv18xx_porting/regression_out/${NET}_${CHIP_NAME}
mkdir -p $NET_DIR
pushd $NET_DIR

model_def_opt=
if [ x$MODEL_DEF != x ] && [ -f $MODEL_DEF ]; then
  model_def_opt="--model_def ${MODEL_DEF}"
else
  echo "Error: can't find model file for ${NET}"
  exit 1
fi

model_data_opt=
# caffemodel
if echo ${MODEL_DEF} | grep -q -E '\.prototxt$'; then
  if [ x$MODEL_DAT != x ] && [ -f $MODEL_DAT ]; then
    model_data_opt="--model_data ${MODEL_DAT}"
  else
    echo "Error: no caffemodel"
    exit 1
  fi
fi

excepts_opt=
if [ x${EXCEPTS} != x ]; then
  excepts_opt="--excepts=${EXCEPTS}"
fi

input_shapes_opt=
if [ x${INPUT_SHAPE} != x ]; then
  input_shapes_opt="--input_shapes=${INPUT_SHAPE}"
fi

resize_dims_opt=
if [ x${IMAGE_RESIZE_DIMS} != x ]; then
  resize_dims_opt="--resize_dims=${IMAGE_RESIZE_DIMS}"
fi

keep_aspect_ratio_opt=
if [ x${RESIZE_KEEP_ASPECT_RATIO} == x1 ]; then
  keep_aspect_ratio_opt="--keep_aspect_ratio"
fi

output_names_opt=
if [ x${OUTPUTS} != x ]; then
  output_names_opt="--output_names=${OUTPUTS}"
fi

mean_opt=
if [ x${MEAN} != x ]; then
  mean_opt="--mean=${MEAN}"
fi

scale_opt=
if [ x${INPUT_SCALE} != x ]; then
  scale_opt="--scale=${INPUT_SCALE}"
fi

pixel_format_opt=
if [ x${MODEL_CHANNEL_ORDER} != x ]; then
  pixel_format_opt="--pixel_format=${MODEL_CHANNEL_ORDER}"
fi

channel_format_opt=
if [ x${DATA_FORMAT} != x ]; then
  channel_format_opt="--channel_format=${DATA_FORMAT}"
fi

do_bf16=
if [ x${DO_QUANT_BF16} != x ];then
  do_bf16=${DO_QUANT_BF16}
fi

tolerance_top_opt=
if [ x${TOLERANCE_TOP} != x ];then
  tolerance_top_opt="--tolerance ${TOLERANCE_TOP}"
fi

test_input_opt=
test_result_opt=
test_innpz_opt=
test_reference_opt=
top_result=${NET}_top_outputs.npz
if [ x${IMAGE_PATH} != x ]; then
  test_input_opt="--test_input=${IMAGE_PATH}"
  test_result_opt="--test_result=${top_result}"
  test_innpz_opt="--test_input=${NET}_in_f32.npz"
  test_reference_opt="--test_reference=${top_result}"
fi

model_transform.py \
  --model_name ${NET} \
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
  ${tolerance_top_opt} \
  ${excepts_opt} \
  --mlir ${NET}.mlir

#########################
# deploy to bf16 cvimodel
#########################
tolerance_bf16_opt=
if [ x${TOLERANCE_BF16} != x ]; then
  tolerance_bf16_opt="--tolerance ${TOLERANCE_BF16}"
fi
if [ ${do_bf16} == 1 ]; then
model_deploy.py \
  --mlir ${NET}.mlir \
  --quantize BF16 \
  --chip ${CHIP_NAME} \
  ${test_innpz_opt} \
  ${test_reference_opt} \
  ${excepts_opt} \
  --tolerance 0.95,0.85 \
  --model ${NET}_${CHIP_NAME}_bf16.cvimodel
fi
#########################
# deploy to int8 cvimodel
#########################

do_cali=1
input_num=${INPUT_NUM}
# only once
if [ ${do_cali} == 1 ] && [ ! -f ${CALI_TABLE}_${INPUT_NUM} ]; then
  if [ x${DATA_SET} == x ]; then
    echo "Error: ${NET} has no dataset"
    exit 1
  fi
  run_calibration.py ${NET}.mlir \
    --dataset ${CALI_IMAGES} \
    --input_num ${INPUT_NUM} \
    -o ${CALI_TABLE}_${INPUT_NUM}
fi
CALI_TABLE=${CALI_TABLE}_${INPUT_NUM}

cali_opt=
if [ -f ${CALI_TABLE} ]; then
  cali_opt="--calibration_table ${CALI_TABLE}"
fi

mix_opt=
if [ ${MIX_PRECISION_TABLE} != '-'1 ]; then
  mix_opt="--quantize_table ${MIX_PRECISION_TABLE}"
fi

tolerance_sym_opt=
if [ x${TOLERANCE_INT8} != x ]; then
  tolerance_sym_opt="--tolerance ${TOLERANCE_INT8}"
fi
model_deploy.py \
  --mlir ${NET}.mlir \
  --quantize INT8 \
  ${cali_opt} \
  ${mix_opt} \
  --chip ${CHIP_NAME} \
  ${test_innpz_opt} \
  ${test_reference_opt} \
  ${tolerance_sym_opt} \
  ${excepts_opt} \
  --quant_input \
  --model ${NET}_${CHIP_NAME}_int8_sym.cvimodel
popd


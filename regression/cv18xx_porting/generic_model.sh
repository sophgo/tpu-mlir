#!/bin/bash
###########################################################
# usage: ./generic_models.sh  model_name \
#                             [model_zoo_path=/data/mlir-models] \
#                             [dataset_path=/data/dataset]
###############################################################


# set -ex
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

function usage {
    echo ""
    echo "usage: ./generic_model.sh -n model_name \\ "
    echo "                          [-m model_zoo_path=/data/mlir-models] \\ "
    echo "                          [-d dataset_path=/data/dataset] \\ "
    echo -e "                          -f \t\t#FusePreprocess \\ "
    echo -e "                          -p pixel_format \t#Set pixel_format \\ "
    echo -e "                          -a \t\t#AlignedInput \\"
    echo -e "                          -c \t\t#Set chip name \\"
}

# getopts
while getopts ":c:m:d:n:f:p:a:r:" opt
do
  case $opt in
  m)
    m=$OPTARG
    echo "set model path: $m";;
  d)
    d=$OPTARG
    echo "set dataset path: $d";;
  n)
    n=$OPTARG
    echo "set model name: $n";;
  f)
    f=1
    echo "set fuse preprocess: $f";;
  p)
    p=$OPTARG
    echo "set pixel format: $p";;
  a)
    a=1
    echo "set algined input: $a";;
  c)
    c=$OPTARG
    echo "set chip name: $c";;
  r)
    r=$OPTARG
    echo "set release path: $r";;
  ?)
    usage
    exit 1;;
  esac
done

if [ x$n == x ]; then
  usage
  exit 1
fi
NET=$n
CHIP_NAME=${SET_CHIP_NAME:-cv183x}
if [ x$c != x ]; then
  CHIP_NAME=$c
fi
cfg_file=$REGRESSION_PATH/cv18xx_porting/config/model_config_18xx.sh

if [ ! -f $cfg_file ]; then
  echo "Error: can't open config file ${cfg_file}"
  exit 1
fi

if [ x$m != x ]; then
  MODEL_PATH=$m
fi
MODEL_PATH=${MODEL_PATH:-/data/mlir-models}
if [ ! -d $MODEL_PATH ]; then
  echo "Error: model path not exist\n"
  exit 1
fi

if [ x$d != x ]; then
  DATA_SET=$d
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

NET_DIR=$REGRESSION_PATH/cv18xx_porting/regression_out/${CHIP_NAME}/${NET}

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

mkdir -p $NET_DIR
pushd $NET_DIR

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
if [ ${SET_CHIP_NAME}x = "cv181x"x ] ||
   [ ${SET_CHIP_NAME}x = "cv180x"x ];then
  do_bf16=0
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

#fuse_preprocess use image as input.
if [ x${f} != x ];then
  test_innpz_opt="--test_input=${IMAGE_PATH}"
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
echo model_transform.py \
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
cvimodel_name="${NET}_${CHIP_NAME}_bf16.cvimodel"
if [ x${TOLERANCE_BF16} != x ]; then
  tolerance_bf16_opt="--tolerance ${TOLERANCE_BF16}"
fi
if [ ${do_bf16} == 1 ]; then
  cmd="model_deploy.py \
    --mlir ${NET}.mlir \
    --quantize BF16 \
    --chip ${CHIP_NAME} \
    --compare_all "
  if [ x${f} != x ];then
    cmd=$cmd"--fuse_preprocess "
  fi
  if [ x${p} != x ];then
    cmd=$cmd"--customization_format=${p} "
  fi
  if [ x${a} != x ];then
    cmd=$cmd"--aligned_input "
  fi
  cmd=$cmd"${test_innpz_opt} \
    ${test_reference_opt} \
    ${excepts_opt} \
    ${tolerance_bf16_opt} \
    --model ${cvimodel_name}"
  echo $cmd
  eval $cmd
  # mv cvimodel to release path
  if [ x${r} != x ];then
    mv ${cvimodel_name} $r/
  fi
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

cvimodel_name="${NET}_${CHIP_NAME}_int8_sym.cvimodel"
cmd="model_deploy.py \
  --mlir ${NET}.mlir \
  --quantize INT8 \
  ${cali_opt} \
  ${mix_opt} \
  --chip ${CHIP_NAME} \
  --compare_all "
# fuse preprocess
if [ x${f} != x ];then
  cmd=$cmd"--fuse_preprocess "
fi
# pixel format
if [ x${p} != x ];then
  cmd=$cmd"--customization_format=${p} "
fi
if [ x${a} != x ];then
  cmd=$cmd"--aligned_input "
fi
cmd=$cmd"${test_innpz_opt} \
  ${test_reference_opt} \
  ${tolerance_sym_opt} \
  ${excepts_opt} \
  --quant_input \
  --model ${cvimodel_name}"
echo $cmd
eval $cmd
# mv cvimodel to release path
if [ x${r} != x ];then
  mv ${cvimodel_name} $r/
fi
popd


#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#set -xe

RELEASE_PATH=$1
#DEBUG=$2 #0/1
if [ x${RELEASE_PATH} == x ];then
  echo "Error, RELEASE_PATH not set."
  exit 1
fi
echo "RELEASE_PATH: $RELEASE_PATH"

dest_dir=$RELEASE_PATH
rm -rf $dest_dir
mkdir -p $dest_dir
tmp_dir=${REGRESSION_PATH}/regression_out/cvimodel_samples

function run_cvi_net() {
  local model=$1
  local chip=$2
  local quant_type=$3
  local fuse_preprocess=$4
  local cust_format=$5
  local aligned=$6
  local merge_weight=$7
  local dst_model=$8
  local info="${model} ${chip} ${quant_type} ${fuse_preprocess} ${cust_format} ${aligned} ${merge_weight}"
  local log_dir=${REGRESSION_PATH}/regression_out
  echo "======= run_models.sh ${info}====="
  echo "${REGRESSION_PATH}/run_model.sh $model $chip basic 0 $quant_type $fuse_preprocess $cust_format $algined $merge_weight"
  local out_log=${log_dir}/${model}_${chip}_${quant_type}_${fuse_preprocess}_${cust_format}_${aligned}_${merge_weight}.log
  ${REGRESSION_PATH}/run_model.sh $model $chip basic 0 $quant_type $fuse_preprocess $cust_format $aligned $merge_weight >$out_log 2>&1 | true

  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$info regression FAILED"
    echo "$info regression FAILED" >>${log_dir}/result_cvi_samples.log
    cat $out_log >>${log_dir}/fail.log
    exit 1
  else
    echo "$info regression PASSED"
    echo "$info regression PASSED" >>${log_dir}/result_cvi_samples.log
    rm ${out_log}
  fi

  local ori_file=${REGRESSION_PATH}/regression_out/${model}_${chip}/*.cvimodel
  local dst_file=${tmp_dir}/${dst_model}
  mv ${REGRESSION_PATH}/regression_out/${model}_${chip}/*.cvimodel ${tmp_dir}/${dst_model}
  rm -rf $REGRESSION_PATH/regression_out/${model}_${chip}
}

function pack_sample_cvimodels() {
  local chip=$1
  mkdir -p ${tmp_dir}
  pushd ${REGRESSION_PATH}/regression_out

  run_cvi_net mobilenet_v2_cvi ${chip} INT8_SYM 0 BGR_PLANAR 0 0 mobilenet_v2.cvimodel
  run_cvi_net mobilenet_v2_cvi ${chip} INT8_SYM 1 BGR_PLANAR 0 0 mobilenet_v2_fused_preprocess.cvimodel
  run_cvi_net mobilenet_v2_cvi ${chip} BF16 0 BGR_PLANAR 0 0 mobilenet_v2_bf16.cvimodel
  run_cvi_net mobilenet_v2_cvi ${chip} INT8_SYM 1 YUV420_PLANAR 1 0 mobilenet_v2_int8_yuv420.cvimodel
  run_cvi_net retinaface_mnet_cvi ${chip} INT8_SYM 1 RGB_PLANAR 0 0 retinaface_mnet25_600_fused_preprocess_with_detection.cvimodel
  run_cvi_net retinaface_mnet_cvi ${chip} INT8_SYM 1 RGB_PLANAR 1 0 retinaface_mnet25_600_fused_preprocess_aligned_input.cvimodel

  if [ ! x$chip = 'xcv181x' ] && [ ! x$chip = 'xcv180x' ]; then
    run_cvi_net yolov3_416_cvi ${chip} INT8_SYM 1 RGB_PLANAR 0 0 yolo_v3_416_fused_preprocess_with_detection.cvimodel
  fi

  if [ ! x$chip = 'xcv180x' ]; then
    run_cvi_net yolox_s_cvi ${chip} INT8_SYM 0 BGR_PLANAR 0 0 yolox_s.cvimodel
    run_cvi_net yolov5s_cvi ${chip} INT8_SYM 1 RGB_PLANAR 0 0 yolov5s_fused_preprocess.cvimodel
    run_cvi_net yolov5s_cvi ${chip} INT8_SYM 1 RGB_PLANAR 1 0 yolov5s_fused_preprocess_aligned_input.cvimodel
    run_cvi_net alphapose_res50_cvi ${chip} INT8_SYM 1 RGB_PLANAR 0 0 alphapose_fused_preprocess.cvimodel
    run_cvi_net arcface_res50_cvi ${chip} INT8_SYM 1 RGB_PLANAR 0 0 arcface_res50_fused_preprocess.cvimodel
    run_cvi_net arcface_res50_cvi ${chip} INT8_SYM 1 RGB_PLANAR 1 0 arcface_res50_fused_preprocess_aligned_input.cvimodel
  fi


  #gen merged cvimodel
  run_cvi_net mobilenet_v2_cvi ${chip} INT8_SYM 0 BGR_PLANAR 0 1 mobilenet_v2_bs1.cvimodel
  run_cvi_net mobilenet_v2_cvi_bs4 ${chip} INT8_SYM 0 BGR_PLANAR 0 1 mobilenet_v2_bs4.cvimodel
  model_tool --combine ${tmp_dir}/mobilenet_v2_bs1.cvimodel ${tmp_dir}/mobilenet_v2_bs4.cvimodel -o ${tmp_dir}/mobilenet_v2_bs1_bs4.cvimodel
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "model_tool --combine FAILED"
    exit 1
  fi
  rm ${tmp_dir}/mobilenet_v2_bs1.cvimodel ${tmp_dir}/mobilenet_v2_bs4.cvimodel

  #tar pack
  tar  zcvf ${dest_dir}/cvimodel_samples_${chip}.tar.gz  cvimodel_samples
  rm -rf ${tmp_dir}
  popd
}

pack_sample_cvimodels cv180x
pack_sample_cvimodels cv181x
pack_sample_cvimodels cv182x
pack_sample_cvimodels cv183x

# if [ ${DEBUG} = 0 ]; then
#  rm ${REGRESSION_PATH}/regression_out/*.log
# fi

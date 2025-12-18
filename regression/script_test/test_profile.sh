#!/bin/bash
# #!/bin/bash
# # test case: compile blazeface F16 bmodel with debug info for tpu_profile
# set -ex
#
# DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
#
# mkdir -p test_profile
# pushd test_profile
#
# model_transform.py \
#   --model_name blazeface \
#   --model_def ${REGRESSION_PATH}/model/blazeface.onnx \
#   --input_shapes [[1,3,128,128]] \
#   --test_input ${REGRESSION_PATH}/npz_input/blazeface_in.npz \
#   --test_result ${REGRESSION_PATH}/script_test/test_profile/blazeface_top_outputs.npz \
#   --debug \
#   --mlir ${REGRESSION_PATH}/script_test/test_profile/blazeface.mlir
#
# # deploy F16 bmodel with debug kept (for tpu_profile etc.)
# model_deploy.py \
#   --mlir ${REGRESSION_PATH}/script_test/test_profile/blazeface.mlir \
#   --quantize F16 \
#   --processor cv184x \
#   --test_input ${REGRESSION_PATH}/npz_input/blazeface_in.npz \
#   --test_reference ${REGRESSION_PATH}/script_test/test_profile/blazeface_top_outputs.npz \
#   --debug \
#   --model ${REGRESSION_PATH}/script_test/test_profile/blazeface_cv184x_f16.bmodel
#
# popd
#
# test case: compile yolov5s F16 bmodel with debug info for tpu_profile
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROFILE_DIR="${DIR}/../profile"

check_file_nonempty() {
  if [ ! -s "$1" ]; then
    echo "ERROR: missing or empty file: $1"
    exit 1
  fi
}

check_dir_nonempty() {
  if [ ! -d "$1" ]; then
    echo "ERROR: missing dir: $1"
    exit 1
  fi
  if [ -z "$(ls -A "$1")" ]; then
    echo "ERROR: empty dir: $1"
    exit 1
  fi
}

mkdir -p test_profile
pushd test_profile

tpu_profile.py --arch bm1684x --format layer "${PROFILE_DIR}/bmprofile_data-1_bm1684x/" "${PROFILE_DIR}/profile_bm1684x/" && tpu_profile.py --arch bm1684x --format html "${PROFILE_DIR}/bmprofile_data-1_bm1684x/" "${PROFILE_DIR}/profile_bm1684x/"

tpu_profile.py --arch bm1688 --format layer "${PROFILE_DIR}/bmprofile_data-1_bm1688/" "${PROFILE_DIR}/profile_bm1688/" && tpu_profile.py --arch bm1688 --format html "${PROFILE_DIR}/bmprofile_data-1_bm1688/" "${PROFILE_DIR}/profile_bm1688/"

tpu_profile.py "${PROFILE_DIR}/cdm_profile_data_dev0_bm1690" "${PROFILE_DIR}/profile_bm1690" --mode time --arch BM1690

tpu_profile.py "${PROFILE_DIR}/bmprofile_data-1_cv184x/" "${PROFILE_DIR}/profile_cv184x/" --mode time --arch cv184x

# tpu_profile.py --arch cv184x --format layer "${PROFILE_DIR}/bmprofile_data-1_cv184x/" "${PROFILE_DIR}/profile_cv184x/" && tpu_profile.py --arch cv184x --format html "${PROFILE_DIR}/bmprofile_data-1_cv184x/" "${PROFILE_DIR}/profile_cv184x/"

bm1684x_dir="${PROFILE_DIR}/profile_bm1684x"
check_file_nonempty "${bm1684x_dir}/layer.csv"
check_file_nonempty "${bm1684x_dir}/instruction.csv"
check_file_nonempty "${bm1684x_dir}/summary.csv"
check_file_nonempty "${bm1684x_dir}/mac_util.csv"
check_file_nonempty "${bm1684x_dir}/result.html"
check_file_nonempty "${bm1684x_dir}/profile_data.js"

bm1688_dir="${PROFILE_DIR}/profile_bm1688"
check_file_nonempty "${bm1688_dir}/PerfWeb/result.html"
check_file_nonempty "${bm1688_dir}/PerfWeb/profile_data.js"
check_dir_nonempty "${bm1688_dir}/PerfDoc"

bm1690_dir="${PROFILE_DIR}/profile_bm1690"
check_file_nonempty "${bm1690_dir}/PerfWeb/result.html"
check_file_nonempty "${bm1690_dir}/PerfWeb/profile_data.js"
check_dir_nonempty "${bm1690_dir}/PerfDoc"

cv184x_dir="${PROFILE_DIR}/profile_cv184x"
check_file_nonempty "${cv184x_dir}/PerfWeb/result.html"
check_file_nonempty "${cv184x_dir}/PerfWeb/profile_data.js"
check_dir_nonempty "${cv184x_dir}/PerfDoc"

popd

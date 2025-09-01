#!/bin/bash
# test case: test mlir_cut.py
set -ex

mkdir -p test_mlir_cut
pushd test_mlir_cut


# ===------------------------------------------------------------===
# yolov5s, 2-cores, bm1688
# donot use --time_fixed_subnet to generate/test multi-subnet models.
# ===------------------------------------------------------------===

model_transform.py \
       --model_name yolov5s \
       --model_def ${REGRESSION_PATH}/model/yolov5s.onnx \
       --input_shapes [[1,3,640,640]] \
       --mean 0.0,0.0,0.0 \
       --scale 0.0039216,0.0039216,0.0039216 \
       --keep_aspect_ratio \
       --pixel_format rgb \
       --output_names 350,498,646 \
       --test_input ${REGRESSION_PATH}/image/dog.jpg \
       --test_result yolov5s_top_outputs.npz \
       --mlir yolov5s.mlir

run_calibration.py yolov5s.mlir \
      --dataset $REGRESSION_PATH/dataset/COCO2017 \
      --input_num 1 \
      -o yolov5s_cali_table

model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize INT8 \
       --processor bm1688 \
       --num_core 2 \
       --calibration_table yolov5s_cali_table \
       --model yolov5s_1688_int8.bmodel \
       --debug \
    #    --time_fixed_subnet normal \

model_runner.py --model yolov5s_bm1688_int8_sym_tpu.mlir \
      --input yolov5s_in_f32.npz \
      --output yolov5s_tpu_outputs.npz \
      --dump_all

# ===------------------------------------------------------------===
# helper functions
# ===------------------------------------------------------------===
gen_random_tensor_names() {
    local ref_npz_file=$1
    local start=$2
    local end=$3
    local num=$4
    python3 -c "
import numpy as np
import random
import sys

def get_random_tensor_name(ref_npz_file, start=0, end=None, num=1):
    data = np.load(ref_npz_file)
    names = list(data.files)
    if end is None:
        end = len(names)
    available = names[start:end]
    return random.sample(available, min(num, len(available)))

selected = get_random_tensor_name('$ref_npz_file', $start, $end, $num)
print(','.join(selected))
"
}

gen_random_config_file() {
    local random_outputs=$(gen_random_tensor_names yolov5s_tpu_outputs.npz 10 -1 1)
    local config_file=$1
    python3 -c "
import json
import sys
output_names_str = '$random_outputs'
output_names = output_names_str.strip().split()
config = {
    'new_input_names': ['174_Mul'],
    'new_output_names': output_names,
    'assign_new_io_addrs': False,
    'remove_unused_local_ops': False,
    'put_storeop_near_producer': True
}
with open('$config_file', 'w') as f:
    json.dump(config, f, indent=2)
"
}

# ===------------------------------------------------------------===
# run tests
# ===------------------------------------------------------------===


ROUNDS=1
LAYERS=20

for ((i=1; i<=$ROUNDS; i++)); do

# ===------------------------------------------------------------===
# test cut top.mlir
# ===------------------------------------------------------------===

mlir_cut.py --mlir yolov5s.mlir --mode bt --num 3 \
      --output_names $(gen_random_tensor_names yolov5s_top_outputs.npz -20 -10 1) \
      --ref_data yolov5s_top_outputs.npz \
      --do_verify

mlir_cut.py --mlir yolov5s.mlir --mode ft --num 3 \
      --input_names $(gen_random_tensor_names yolov5s_top_outputs.npz 10 20 1) \
      --ref_data yolov5s_top_outputs.npz \
      --do_verify

# ===------------------------------------------------------------===
# test cut tpu.mlir
# ===------------------------------------------------------------===

mlir_cut.py --mlir yolov5s_bm1688_int8_sym_tpu.mlir --mode bt --num 3 \
      --output_names $(gen_random_tensor_names yolov5s_tpu_outputs.npz -20 -10 1) \
      --ref_data yolov5s_tpu_outputs.npz \
      --do_verify

mlir_cut.py --mlir yolov5s_bm1688_int8_sym_tpu.mlir --mode ft --num 3 \
      --input_names $(gen_random_tensor_names yolov5s_tpu_outputs.npz 10 20 1) \
      --ref_data yolov5s_tpu_outputs.npz \
      --do_verify

# ===------------------------------------------------------------===
# test cut final.mlir
# ===------------------------------------------------------------===

mlir_cut.py --mlir yolov5s_bm1688_int8_sym_final.mlir \
      --input_names $(gen_random_tensor_names yolov5s_tpu_outputs.npz 0 -20 $LAYERS) \
      --output_names $(gen_random_tensor_names yolov5s_tpu_outputs.npz -15 -1 3) \
      --ref_data yolov5s_tpu_outputs.npz \
      --do_verify

mlir_cut.py --mlir yolov5s_bm1688_int8_sym_final.mlir \
      --output_names $(gen_random_tensor_names yolov5s_tpu_outputs.npz 10 -1 $LAYERS) \
      --ref_data yolov5s_tpu_outputs.npz \
      --do_verify

gen_random_config_file custom_cfg.json
mlir_cut.py --mlir yolov5s_bm1688_int8_sym_final.mlir \
      --config_file custom_cfg.json \
      --ref_data yolov5s_tpu_outputs.npz \
      --do_verify

done

rm -rf *.npz *.onnx *.bmodel
popd

#!/bin/bash
# test case: test tdb functions with yolov5s on bm1684x
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

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

model_deploy.py \
       --mlir yolov5s.mlir \
       --quantize F16 \
       --processor bm1684x \
       --test_input yolov5s_in_f32.npz \
       --test_reference yolov5s_top_outputs.npz \
       --model yolov5s_1684x_f16.bmodel \
       --debug --compare_all


# bmodel_dis
bmodel_dis.py ./yolov5s_1684x_f16/compilation.bmodel --format reg > reg.log
bmodel_dis.py ./yolov5s_1684x_f16/compilation.bmodel --format mlir > mlir.log

# bmodel_checker
bmodel_checker.py ./yolov5s_1684x_f16 ./yolov5s_bm1684x_f16_tpu_outputs.npz --no_interactive

# bmodel_inference_combine
# .dat as input and .npz as ref
python $DIR/test_bmodel_dump_1.py
npz_tool.py compare ./yolov5s_bm1684x_f16_tpu_outputs.npz ./soc_infer/bmodel_infer_yolov5s_bm1684x_f16_tpu_outputs.npz
# .npz as input and .npz as ref
python $DIR/test_bmodel_dump_2.py
npz_tool.py compare ./yolov5s_bm1684x_f16_tpu_outputs.npz ./soc_infer/bmodel_infer_yolov5s_bm1684x_f16_tpu_outputs.npz
# .dat as input and .mlir as ref
python $DIR/test_bmodel_dump_3.py
npz_tool.py compare ./yolov5s_bm1684x_f16_tpu_outputs.npz ./soc_infer/bmodel_infer_yolov5s_bm1684x_f16_tpu_outputs.npz





#!/bin/bash
# test case: test training graph and inference graph of yolov5s and resnet50 on ILP layergroup
set -ex

mkdir -p test11
pushd test11
# test inference graph
python3 $PROJECT_ROOT/python/test/test_torch.py --case user_define_net --chip bm1690 --num_core 1 --mode f32 --debug
python3 $PROJECT_ROOT/python/test/test_torch.py --case user_define_net --chip bm1690 --num_core 8 --mode f32 --debug

# test training graph
#python3 $PROJECT_ROOT/python/tools/train/test/main.py --chip bm1690 --debug "const_name,disable_dot" --model resnet
#python3 $PROJECT_ROOT/python/tools/train/test/main.py --chip bm1690 --debug "const_name,disable_dot" --model yolo
#python3 $PROJECT_ROOT/python/tools/train/test/main.py --chip bm1690 --debug "const_name,disable_dot" --model test_model  --num_core 1
#python3 $PROJECT_ROOT/python/tools/train/test/main.py --chip bm1690 --debug "const_name,disable_dot" --model test_model  --num_core 8

popd


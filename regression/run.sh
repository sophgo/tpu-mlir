#!/bin/bash
set -ex
rm -rf tmp
./run_deploy.sh
./run_onnx.sh
./run_mobilenet_onnx.sh
./run_tflite.sh

#!/bin/bash
set -ex

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

$DIR/run_resnet18.sh
$DIR/run_yolov5s.sh
$DIR/run_resnet50_tf.sh

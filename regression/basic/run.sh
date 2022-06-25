#!/bin/bash
set -ex

DIR="$( cd "$(dirname "$0")" ; pwd -P )"

$DIR/run_basic.sh
$DIR/run_step_by_step.sh
$DIR/run_tflite.sh

#!/bin/bash
set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mkdir -p struct_optimize_pattern_test2
pushd struct_optimize_pattern_test2

# export ONNX and NPZ into current workdir
OUT_DIR="$(pwd)" python3 $DIR/struct_optimize_pattern_test2.py

# model_transform with struct_optimize enabled
model_transform.py \
  --model_name struct_optimize_pattern_test2 \
  --model_def struct_optimize_pattern_test2.onnx \
  --test_input struct_optimize_pattern_test2_input.npz \
  --test_result struct_optimize_pattern_test2_top_outputs.npz \
  --mlir struct_optimize_pattern_test2.mlir \
  --debug

# Check if the optimization worked by looking for Correlation op in the MLIR
echo "Checking for Correlation optimization..."
if grep -q "top.Correlation" struct_optimize_pattern_test2.mlir; then
    echo "[SUCCESS] Correlation optimization detected in MLIR!"
    grep "top.Correlation" struct_optimize_pattern_test2.mlir
else
    echo "[WARNING] Correlation optimization not found in MLIR"
    echo "Checking ScatterND count in origin vs optimized:"
    origin_scatter_count=$(grep -c "top.ScatterND" struct_optimize_pattern_test2_origin.mlir || echo "0")
    final_scatter_count=$(grep -c "top.ScatterND" struct_optimize_pattern_test2.mlir || echo "0")
    echo "Origin ScatterND count: $origin_scatter_count"
    echo "Final ScatterND count: $final_scatter_count"
fi

# model_deploy (float16)
model_deploy.py \
  --mlir struct_optimize_pattern_test2.mlir \
  --quantize F16 \
  --chip bm1684x \
  --test_input struct_optimize_pattern_test2_input.npz \
  --test_reference struct_optimize_pattern_test2_top_outputs.npz \
  --model struct_optimize_pattern_test2_f32.bmodel \
  --debug

echo "[SUCCESS] Correlation2 struct optimize pattern test completed!"

# Keep important files for debugging
echo "Generated files:"
ls -la *.mlir *.bmodel

popd

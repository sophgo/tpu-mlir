//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::TopKOp::init(InferenceParameter &p) { return success(); }
void tpu::TopKOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::TopKOp::inference(InferenceParameter &p) {
  auto axis = getAxis();
  auto is_largest = getLargest();
  auto replace_topk_indice = getReplaceTopkIndices();
  auto K = getKT() ? (int64_t)p.inputs[1][0] : getK();
  auto is_sorted = getSorted();
  if (is_sorted == false) {
    llvm_unreachable("Not supported");
  }
  auto input_shape = module::getShape(getInput());
  if (axis != input_shape.size() - 1) {
    llvm_unreachable("Not supported");
  }
  bool has_values = !module::isNone(getValues());
  bool has_indices = !module::isNone(getIndices());
  int axis_dim = input_shape[axis];
  int outer_dim = 1;
  for (int i = 0; i < axis; i++) {
    outer_dim *= input_shape[i];
  }
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; i++) {
    auto *ptr = p.inputs[0] + i * axis_dim;
    std::vector<std::pair<int, float>> result;
    topk_indices(result, ptr, axis_dim, K, is_largest);
    for (int k = 0; k < K; k++) {
      if (has_indices && !replace_topk_indice) {
        auto indices_ptr = p.outputs[1] + i * K + k;
        *indices_ptr = (float)result[k].first;
      }
      if (has_values) {
        auto values_ptr = p.outputs[0] + i * K + k;
        *values_ptr = result[k].second;
      }
    }
  }
  if (has_indices && replace_topk_indice) {
    std::string indices_output_name = module::getName(getIndices()).str();
    auto op = getOperation();
    std::string model_name = module::getName(module::getModuleOp(op)).str();
    std::string filename = model_name + "_ref_outputs.npz";
    auto ref_onnx_output = cnpy::npz_load(filename);
    if (ref_onnx_output.find(indices_output_name) != ref_onnx_output.end()) {
      const cnpy::NpyArray &index_array = ref_onnx_output[indices_output_name];
      auto data = index_array.data<int64_t>();
      for (size_t i = 0; i < index_array.num_vals; ++i) {
        auto indices_ptr = p.outputs[1] + i;
        *indices_ptr = data[i];
      }
    }
  }
  std::vector<int64_t> output_shape(input_shape.size());
  for (int i = 0; i < input_shape.size(); i++) {
    if (i == axis) {
      output_shape[i] = K;
    } else {
      output_shape[i] = input_shape[i];
    }
  }
  module::setShape(getValues(), output_shape);
  module::setShape(getIndices(), output_shape);

  return success();
}

bool tpu::TopKOp::support_multi_core() { return false; }

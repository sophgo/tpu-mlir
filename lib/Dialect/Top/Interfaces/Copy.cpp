//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
int64_t top::CopyOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::CopyOp::init(InferenceParameter &p) { return success(); }

void top::CopyOp::deinit(InferenceParameter &p) {}

LogicalResult top::CopyOp::inference(InferenceParameter &p) {
  float *input_data = p.inputs[0];
  float *output_data = p.outputs[0];
  // parse param
  auto shape = module::getI64Array(this->getShape());
  auto i_stride = module::getI64Array(this->getInputStride());
  auto o_stride = module::getI64Array(this->getOutputStride());
  std::vector<int64_t> shape_4;
  std::vector<int64_t> i_stride_4;
  std::vector<int64_t> o_stride_4;
  shape_4 = {1, 1, 1, 1};
  i_stride_4 = {0, 0, 0, 0};
  o_stride_4 = {0, 0, 0, 0};
  int num_dims = shape->size();
  ASSERT_THIS(num_dims <= 4);
  ASSERT_THIS(i_stride->size() == shape->size());
  ASSERT_THIS(o_stride->size() == shape->size());
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = shape->at(end);
    i_stride_4[idx] = i_stride->at(end);
    o_stride_4[idx] = o_stride->at(end);
  }
  // calculate
  for (int n = 0; n < shape_4[0]; n++) {
    for (int c = 0; c < shape_4[1]; c++) {
      for (int h = 0; h < shape_4[2]; h++) {
        for (int w = 0; w < shape_4[3]; w++) {
          int in_index = n * i_stride_4[0] + c * i_stride_4[1] +
                         h * i_stride_4[2] + w * i_stride_4[3];
          int out_index = n * o_stride_4[0] + c * o_stride_4[1] +
                          h * o_stride_4[2] + w * o_stride_4[3];
          output_data[out_index] = input_data[in_index];
        }
      }
    }
  }
  return success();
}

void top::CopyOp::shape_inference() {}

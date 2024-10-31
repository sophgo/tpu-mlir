//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ShapeReverseOp::init(InferenceParameter &p) { return success(); }
void tpu::ShapeReverseOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeReverseOp::inference(InferenceParameter &p) {
  auto output_shape = module::getShape(getOutput());
  int64_t _axis = getAxis();
  int on = output_shape[0];
  int oc = output_shape[1];
  int oh = output_shape.size() > 2 ? output_shape[2] : 1;
  int ow = output_shape.size() > 3 ? output_shape[3] : 1;
  int dim[] = {on, oc, oh, ow};
  int stride[] = {oc * oh * ow, oh * ow, ow, 1};
  for (int in = 0; in < on; in++) {
    for (int ic = 0; ic < oc; ic++) {
      for (int ih = 0; ih < oh; ih++) {
        for (int iw = 0; iw < ow; iw++) {
          int src_index[] = {in, ic, ih, iw};
          int dst_index[] = {in, ic, ih, iw};
          dst_index[_axis] = dim[_axis] - src_index[_axis] - 1;
          int src_offset = src_index[0] * stride[0] + src_index[1] * stride[1] +
                           src_index[2] * stride[2] + src_index[3] * stride[3];
          int dst_offset = dst_index[0] * stride[0] + dst_index[1] * stride[1] +
                           dst_index[2] * stride[2] + dst_index[3] * stride[3];
          p.outputs[0][dst_offset] = p.inputs[0][src_offset];
        }
      }
    }
  }
  return success();
}

bool tpu::ShapeReverseOp::support_multi_core() { return false; }
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <valarray>



int64_t top::SliceOp::getFLOPs() { return 0; }

LogicalResult top::SliceOp::init(InferenceParameter &p) { return success(); }
void top::SliceOp::deinit(InferenceParameter &p) {}

LogicalResult top::SliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = module::getNumElements(output());
  auto offset_v = module::getI64Array(offset());
  auto steps_v = module::getI64Array(steps());
  auto out_shape = module::getShape(output());
  auto in_shape = module::getShape(input());
  auto in_dims = in_shape.size();
  auto out_dims = out_shape.size();
  // just support the dims of input & input is equal.
  assert(in_dims == out_dims);
  // slice[range] -> (offset + stride)
  std::valarray<int64_t> in_stride_v(1, in_dims);
  std::valarray<int64_t> out_stride_v(1, out_dims);
  for (int i = in_stride_v.size() - 2; i >= 0; --i) {
    in_stride_v[i] *= in_stride_v[i + 1] * in_shape[i + 1];
    out_stride_v[i] *= out_stride_v[i + 1] * out_shape[i + 1];
  }
  auto in_offset_v = std::valarray<int64_t>(offset_v->data(), offset_v->size());
  auto in_offset = (in_offset_v * in_stride_v).sum();
  auto out_in_stride_v =
      std::valarray<int64_t>(steps_v->data(), steps_v->size());
  out_in_stride_v *= in_stride_v;

#pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
  for (int64_t i = 0; i < out_num_elem; ++i) {
    std::valarray<int64_t> out_it(1, out_dims);
    int64_t tmp = i;
    for (int j = 0; j < out_dims; j++) {
      out_it[j] = tmp / out_stride_v[j];
      tmp = tmp % out_stride_v[j];
    }
    p.outputs[0][i] = p.inputs[0][(out_it * out_in_stride_v).sum() + in_offset];
  }

  return success();
}

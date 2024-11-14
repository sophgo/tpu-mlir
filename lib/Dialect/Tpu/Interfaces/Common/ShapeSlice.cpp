//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <valarray>

using namespace tpu_mlir::backend;

LogicalResult tpu::ShapeSliceOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ShapeSliceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeSliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = module::getNumElements(getOutput());
  auto offset_v = module::getI64Array(getOffset());
  auto steps_v = module::getI64Array(getSteps());
  std::vector<int64_t> out_shape = module::getShape(getOutput());
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto in_dims = in_shape.size();
  auto out_dims = out_shape.size();
  const size_t slice_dims = offset_v->size();
  // just support the dims of input & input is equal.
  while (out_dims < in_dims) {
    out_shape.insert(out_shape.begin(), 1);
    out_dims++;
  }
  for (int i = 0; i < slice_dims; ++i) {
    if (offset_v->at(i) < 0) {
      offset_v->at(i) += in_shape[i];
    }
    offset_v->at(i) = steps_v->at(i) > 0
                          ? std::clamp(offset_v->at(i), 0L, in_shape[i])
                          : std::clamp(offset_v->at(i), 0L, in_shape[i] - 1);
  }

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

mlir::Type tpu::ShapeSliceOp::type_verify(uint64_t opd_idx,
                                          TypeCastMode &mode) {
  if (opd_idx == 0) {
    return type_verify_case_same(getOperation(), 0, mode);
  }
  return do_nothing(mode);
}

bool tpu::ShapeSliceOp::support_multi_core() { return false; }

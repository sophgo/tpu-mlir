//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::StridedSliceOp::init(InferenceParameter &p) { return success(); }
void tpu::StridedSliceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::StridedSliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = Module::getNumElements(output());
  auto out_shape = Module::getShape(output());
  auto in_shape = Module::getShape(input());
  auto in_dims = in_shape.size();
  auto out_dims = out_shape.size();

  auto input_ = p.inputs[0];
  auto output_ = p.outputs[0];
  auto starts_ = p.inputs[1];
  auto ends_ = p.inputs[2];
  auto strides_ = p.inputs[3];
  auto s_dims = Module::getShape(strides()).size();
  int32_t begin[8], end[8], step[8], input_shape[8];
  int32_t dims, b_mask, e_mask, shrink_mask;

  stride_slice_gen_params(in_shape.data(), in_dims, starts_, ends_, strides_, s_dims,
                          begin_mask(), end_mask(), ellipsis_mask(), new_axis_mask(),
                          shrink_axis_mask(), input_shape, &dims,
                          begin, end, step, &b_mask, &e_mask, &shrink_mask);
  for (int i = 0; i < dims; ++i) {
    begin[i] = StartForAxis(begin, step, b_mask, input_shape, i);
    end[i] = StopForAxis(end, step, e_mask, shrink_mask, input_shape, i, begin[i]);
  }

  //just support the dims of input & input is equal and Stride Slice at one axis now.
  assert(in_dims == out_dims);

  if (in_dims == 2) {
    #pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        int64_t o_index = i * out_shape[1] + j;
        int64_t i_index = (begin[0] + i * step[0]) * in_shape[1] + begin[1] + j * step[1];
        output_[o_index] = input_[i_index];
      }
    }
  } else if (in_dims == 3) {
    #pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        for (int k = 0; k < out_shape[2]; k++) {
          int64_t o_index = (i * out_shape[1] + j) * out_shape[2] + k;
          int64_t i_index = ((begin[0] + i * step[0]) * in_shape[1] + begin[1] + j * step[1]) * in_shape[2] +
                            begin[2] + k * step[2];
          output_[o_index] = input_[i_index];
        }
      }
    }
  } else if (in_dims == 4) {
    #pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        for (int k = 0; k < out_shape[2]; k++) {
          for (int z = 0; z < out_shape[3]; z++) {
            int64_t o_index = ((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3] + z;
            int64_t i_index = (((begin[0] + i * step[0]) * in_shape[1] + begin[1] + j * step[1]) * in_shape[2] +
                               begin[2] + k * step[2]) * in_shape[3] + begin[3] + z * step[3];
            output_[o_index] = input_[i_index];
          }
        }
      }
    }
  }

  return success();
}

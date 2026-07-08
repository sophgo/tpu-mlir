//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaShapeSliceOp(tpu::ShapeSliceOp op) {
  auto in_num = module::getNumElements(op.getInput());
  auto out_num = module::getNumElements(op.getOutput());

  std::vector<int32_t> in_vals(in_num);
  CHECK_CUDA(cudaMemcpy(in_vals.data(), getCudaData(op.getInput()),
                        in_num * sizeof(int32_t), cudaMemcpyDeviceToHost));

  auto in_shape_ref = module::getShape(op.getInput());
  auto out_shape_ref = module::getShape(op.getOutput());
  int in_dims = (int)in_shape_ref.size();
  int out_dims = (int)out_shape_ref.size();

  auto offset = module::getI64Array(op.getOffset());
  auto steps = module::getI64Array(op.getSteps());
  int slice_dims = (int)offset->size();

  std::vector<int64_t> in_shape(in_shape_ref.begin(), in_shape_ref.end());
  std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
  while (out_dims < in_dims) {
    out_shape.insert(out_shape.begin(), 1);
    out_dims++;
  }

  std::vector<int64_t> in_stride_v(in_dims, 1);
  std::vector<int64_t> out_stride_v(out_dims, 1);
  for (int i = in_dims - 2; i >= 0; --i) {
    in_stride_v[i] *= in_stride_v[i + 1] * in_shape[i + 1];
    out_stride_v[i] *= out_stride_v[i + 1] * out_shape[i + 1];
  }

  int64_t in_offset = 0;
  std::vector<int64_t> out_in_stride_v(slice_dims);
  for (int i = 0; i < slice_dims; ++i) {
    if ((*offset)[i] < 0)
      (*offset)[i] += in_shape[i];
    in_offset += (*offset)[i] * in_stride_v[i];
    out_in_stride_v[i] = (*steps)[i] * in_stride_v[i];
  }

  std::vector<int32_t> out_vals(out_num);
  for (int i = 0; i < out_num; ++i) {
    int64_t tmp = i;
    int64_t in_idx = in_offset;
    for (int j = 0; j < out_dims; ++j) {
      int64_t out_it = tmp / out_stride_v[j];
      tmp = tmp % out_stride_v[j];
      if (j < slice_dims)
        in_idx += out_it * out_in_stride_v[j];
    }
    out_vals[i] = in_vals[in_idx];
  }

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_vals.data(),
                        out_num * sizeof(int32_t), cudaMemcpyHostToDevice));
}

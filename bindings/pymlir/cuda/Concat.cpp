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

void py_cuda::cudaConcatOp(tpu::ConcatOp op) {
  auto out_shape = module::getShape(op.getOutput());
  int axis = op.getAxis();
  int outer_dim = 1, axis_dim = 1, inner_dim = 1;
  for (int i = 0; i < axis; i++) {
    outer_dim *= out_shape[i];
  }
  axis_dim = out_shape[axis];
  for (int i = axis + 1; i < out_shape.size(); i++) {
    inner_dim *= out_shape[i];
  }
  int tbytes = module::getDtypeSize(op.getOutput());
  auto dst = getCudaData(op.getOutput());
  if (module::isCV18xx() && module::isUniformQuantized(op.getOutput())) {
    auto nInputs = op.getInputs().size();
    auto multiplier_v = module::getI64Array(op.getMultipliers(), nInputs, 1);
    auto rshift_v = module::getI64Array(op.getRshifts(), nInputs, 0);
    int offset = 0;
    for (size_t i = 0; i < nInputs; i++) {
      auto in = op.getInputs()[i];
      auto in_shape = module::getShape(in);
      auto num_elem = module::getNumElements(in);
      auto src = getCudaData(in);
      int multiplier = multiplier_v->at(i);
      int rshift = rshift_v->at(i);
      if (multiplier == 1 && rshift == 0) {
        cuda::copyAxis(src, dst, outer_dim, axis_dim, inner_dim, offset,
                       in_shape[axis], tbytes);
        offset += in_shape[axis];
      } else {
        auto temp = cuda_malloc(module::getBytes(in));
        cuda::cvMulShiftInt8(src, temp.get(), multiplier, rshift, num_elem);
        cuda::copyAxis(temp.get(), dst, outer_dim, axis_dim, inner_dim, offset,
                       in_shape[axis], tbytes);
        offset += in_shape[axis];
      }
    }
  } else {
    int offset = 0;
    for (auto in : op.getInputs()) {
      auto src = getCudaData(in);
      auto in_shape = module::getShape(in);
      cuda::copyAxis(src, dst, outer_dim, axis_dim, inner_dim, offset,
                     in_shape[axis], tbytes);
      offset += in_shape[axis];
    }
  }
  auto out_type = module::getStorageType(op.getOutput());
  if (op.getDoRelu() && !out_type.isUnsignedInteger(8)) {
    int num_out = module::getNumElements(op.getOutput());
    cuda::doRelu(dst, num_out, getCudaType(op.getOutput()));
  }
}

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

void py_cuda::cudaGenericCpu(tpu::GenericCpuOp op) {
  if (op.getCpuOpName() != "quant") {
    UNREACHABLE_OP("Not Implemented", op);
  }
  if (!module::isUniformQuantized(op.getOutputs()[0])) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto param = op.getParam().value();
  float scale = param.get("scale").cast<FloatAttr>().getValueAsDouble();
  void *input = getCudaData(op.getInputs()[0]);
  void *output = getCudaData(op.getOutputs()[0]);
  int num_elems = module::getNumElements(op.getInputs()[0]);
  cudaF32ToInt8(input, output, scale, num_elems, true, CUDA_AWAY_FROM_ZERO);
}
